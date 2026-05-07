from __future__ import annotations

from flax import linen as nn
import jax
import jax.numpy as jnp
from tips.scenic.configs import tips_model_config
from tips.scenic.models import tips, vit

from crossformer.model.components.dpt import DPTConfig, DPTHead

VGG19_BLOCKS = (
    (64, 2),
    (128, 2),
    (256, 4),
    (512, 4),
    (512, 4),
)


def _upsample2x(x):
    b, h, w, c = x.shape
    return jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")


def _resize_like(x, ref):
    b, h, w, c = x.shape
    return jax.image.resize(ref, (b, h, w, c), method="nearest")


def _variant_out_size(variant: str, h: int, w: int) -> tuple[int, int]:
    if variant == "full":
        return h, w
    if variant == "half":
        return h // 2, w // 2
    if variant == "quarter":
        return h // 4, w // 4
    raise ValueError(f"unknown variant: {variant}")


class VGGBlock(nn.Module):
    ch: int
    depth: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth):
            x = nn.Conv(self.ch, (3, 3), padding="SAME", name=f"conv{i + 1}")(x)
            x = nn.relu(x)
        return x


class DeconvBlock(nn.Module):
    out_ch: int
    refine_ch: int | None = None

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(
            self.out_ch,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="up",
        )(x)
        x = nn.relu(x)
        if self.refine_ch is not None:
            x = nn.Conv(self.refine_ch, (3, 3), padding="SAME", name="refine")(x)
            x = nn.relu(x)
        return x


class UpsampleBlock(nn.Module):
    out_ch: int
    refine_ch: int
    relu_after_refine: bool = False

    @nn.compact
    def __call__(self, x):
        x = _upsample2x(x)
        x = nn.Conv(self.out_ch, (3, 3), padding="SAME", name="conv")(x)
        x = nn.relu(x)
        x = nn.Conv(self.refine_ch, (3, 3), padding="SAME", name="refine")(x)
        if self.relu_after_refine:
            x = nn.relu(x)
        return x


class HeatmapHead(nn.Module):
    num_keypoints: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (3, 3), padding="SAME", name="conv1")(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), padding="SAME", name="conv2")(x)
        x = nn.relu(x)
        logits = nn.Conv(self.num_keypoints, (3, 3), padding="SAME", name="out")(x)
        return jax.nn.sigmoid(logits)


class MaskHead(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (3, 3), padding="SAME", name="conv1")(x)
        x = nn.relu(x)
        logits = nn.Conv(1, (1, 1), padding="SAME", name="out")(x)
        return jax.nn.sigmoid(logits)


class SoftArgmaxPavlo(nn.Module):
    num_keypoints: int
    learned_beta: bool = True
    initial_beta: float = 1.0

    @nn.compact
    def __call__(self, heatmaps, size_mult: float = 1.0):
        b, h, w, k = heatmaps.shape
        x = nn.avg_pool(heatmaps, window_shape=(7, 7), strides=(1, 1), padding=[(3, 3), (3, 3)])
        x = jnp.transpose(x, (0, 3, 1, 2)).reshape(b, k, h * w)
        x = x - jnp.max(x, axis=-1, keepdims=True)
        if self.learned_beta:
            beta = self.param("beta", nn.initializers.constant(self.initial_beta), (self.num_keypoints,))
        else:
            beta = jnp.full((self.num_keypoints,), self.initial_beta, dtype=x.dtype)
        probs = jax.nn.softmax(x * beta[None, :, None], axis=-1).reshape(b, k, h, w)

        xs = jnp.arange(w, dtype=x.dtype) * size_mult
        ys = jnp.arange(h, dtype=x.dtype) * size_mult
        x_vals = jnp.sum(probs * xs[None, None, None, :], axis=(2, 3))
        y_vals = jnp.sum(probs * ys[None, None, :, None], axis=(2, 3))
        return jnp.stack((x_vals, y_vals), axis=-1)


class DreamHourglass(nn.Module):
    num_keypoints: int
    decoder: str = "upsample"
    deconv_decoder: bool = False
    full_output: bool = False
    skip_connections: bool = False
    output_scale: str = "quarter"
    internalize_spatial_softmax: bool = False
    learned_beta: bool = True
    initial_beta: float = 1.0

    def setup(self):
        if self.output_scale not in {"quarter", "half", "full"}:
            raise ValueError(f"unknown output_scale: {self.output_scale}")
        if self.decoder not in {"upsample", "deconv", "dpt"}:
            raise ValueError(f"unknown decoder: {self.decoder}")

    def _skip(self, x, ref):
        return x + ref if self.skip_connections else x

    def _encoder(self, x, stages):
        feats = []
        pools = []
        for i, (ch, depth) in enumerate(VGG19_BLOCKS, start=1):
            x = VGGBlock(ch, depth, name=f"layer_0_{i}_down")(x)
            feats.append(x)
            stages.append((f"enc{i}", x))
            if i < len(VGG19_BLOCKS):
                x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")
                pools.append(x)
                stages.append((f"pool{i}", x))
        return feats[0], pools[0], pools[1], pools[2], pools[3], feats[4]

    def _deconv_decoder(self, xs, stages):
        x_0_1, x_0_1_d, x_0_2_d, x_0_3_d, x_0_4_d, x_0_5 = xs
        x = self._skip(x_0_5, x_0_4_d)
        specs = (
            ("deconv_0_4", 256, 256, None, None),
            ("deconv_0_3", 128, 128, x_0_3_d, "quarter"),
            ("deconv_0_2", 64, 64, x_0_2_d, "half"),
            ("deconv_0_1", 64, None, x_0_1_d, "full"),
        )
        for name, out_ch, refine_ch, skip, stop in specs:
            x = self._skip(x, skip) if skip is not None else x
            x = DeconvBlock(out_ch, refine_ch, name=name)(x)
            stages.append((name, x))
            if self.output_scale == stop:
                return self._skip(x, x_0_1) if stop == "full" else x
        return x

    def _upsample_decoder(self, xs, stages):
        x_0_1, _, _, x_0_3_d, x_0_4_d, x_0_5 = xs
        x = self._skip(x_0_5, x_0_4_d)
        specs = (
            ("upsample_0_4", 256, 256, False, None, None),
            ("upsample_0_3", 128, 64, False, x_0_3_d, "quarter"),
            ("upsample_0_2", 64, 64, True, None, "half"),
            ("upsample_0_1", 64, 64, True, None, "full"),
        )
        for name, out_ch, refine_ch, relu, skip, stop in specs:
            x = self._skip(x, skip) if skip is not None else x
            x = UpsampleBlock(out_ch, refine_ch, relu_after_refine=relu, name=name)(x)
            stages.append((name, x))
            if self.output_scale == stop and not self.full_output:
                return self._skip(x, x_0_1) if stop == "full" else x
        return self._skip(x, x_0_1)

    def _dpt_decoder(self, xs, stages):
        _, x_0_1_d, x_0_2_d, x_0_3_d, _, x_0_5 = xs
        feats = (x_0_1_d, x_0_2_d, x_0_3_d, x_0_5)
        feats = tuple(nn.Conv(256, (1, 1), padding="SAME", name=f"dpt_proj_{i}")(feat) for i, feat in enumerate(feats))
        for i, feat in enumerate(feats):
            stages.append((f"dpt_proj_{i}", feat))

        cfg = DPTConfig(feature_dim=256, out_channels=64)
        x = DPTHead(cfg, name="dpt_head")(feats)
        out_h, out_w = _variant_out_size(self.output_scale, x.shape[1], x.shape[2])
        if x.shape[1:3] != (out_h, out_w):
            x = jax.image.resize(x, (x.shape[0], out_h, out_w, x.shape[-1]), method="bilinear")
        stages.append(("dpt_decoder", x))
        return x

    @nn.compact
    def __call__(self, x):
        stages = []
        enc = self._encoder(x, stages)
        if self.decoder == "dpt":
            x = self._dpt_decoder(enc, stages)
        elif self.decoder == "deconv" or self.deconv_decoder:
            x = self._deconv_decoder(enc, stages)
        else:
            x = self._upsample_decoder(enc, stages)

        heatmaps = HeatmapHead(self.num_keypoints, name="heads_0")(x)
        stages.append(("heatmaps", heatmaps))
        mask = MaskHead(name="mask_head")(x)
        if mask is not None:
            stages.append(("mask", mask))
        if not self.internalize_spatial_softmax:
            out = {"heatmaps": heatmaps, "mask": mask} if mask is not None else heatmaps
            return out, tuple((name, arr.shape) for name, arr in stages)

        keypoints = SoftArgmaxPavlo(
            self.num_keypoints,
            learned_beta=self.learned_beta,
            initial_beta=self.initial_beta,
            name="softmax",
        )(heatmaps)
        stages.append(("keypoints", keypoints))
        out = {"heatmaps": heatmaps, "keypoints": keypoints}
        if mask is not None:
            out["mask"] = mask
        return out, tuple((name, arr.shape) for name, arr in stages)


class DreamHourglassMultiStage(nn.Module):
    num_keypoints: int
    n_stages: int = 2
    decoder: str = "upsample"
    deconv_decoder: bool = False
    full_output: bool = False
    skip_connections: bool = False
    output_scale: str = "quarter"
    internalize_spatial_softmax: bool = False
    learned_beta: bool = True
    initial_beta: float = 1.0

    def setup(self):
        if not 1 <= self.n_stages <= 6:
            raise ValueError("DREAM supports 1 to 6 stages")

    @nn.compact
    def __call__(self, x):
        outputs = []
        shapes = []
        stage_input = x
        prev_heatmaps = None

        for i in range(self.n_stages):
            if i > 0:
                prev_in = _resize_like(x, prev_heatmaps)
                stage_input = jnp.concatenate([x, prev_in], axis=-1)

            out, stage_shapes = DreamHourglass(
                self.num_keypoints,
                decoder=self.decoder,
                deconv_decoder=self.deconv_decoder,
                full_output=self.full_output,
                skip_connections=self.skip_connections,
                output_scale=self.output_scale,
                internalize_spatial_softmax=self.internalize_spatial_softmax,
                learned_beta=self.learned_beta,
                initial_beta=self.initial_beta,
                name=f"stage{i + 1}",
            )(stage_input)
            outputs.append(out)
            shapes.extend((f"stage{i + 1}/{name}", shape) for name, shape in stage_shapes)
            prev_heatmaps = out["heatmaps"] if isinstance(out, dict) else out

        return tuple(outputs), tuple(shapes)


class DreamVGG(nn.Module):
    num_keypoints: int
    variant: str = "full"
    decoder: str = "auto"
    deconv_decoder: bool | None = None
    full_output: bool | None = None
    skip_connections: bool = False
    n_stages: int = 1
    internalize_spatial_softmax: bool = False
    learned_beta: bool = True
    initial_beta: float = 1.0

    def setup(self):
        if self.variant not in {"quarter", "half", "full"}:
            raise ValueError(f"unknown variant: {self.variant}")
        if self.decoder not in {"auto", "upsample", "deconv", "dpt"}:
            raise ValueError(f"unknown decoder: {self.decoder}")
        if not 1 <= self.n_stages <= 6:
            raise ValueError("DREAM supports 1 to 6 stages")

    def _decoder_config(self):
        decoder = ("deconv" if self.variant == "full" else "upsample") if self.decoder == "auto" else self.decoder
        deconv_decoder = decoder == "deconv"
        if self.deconv_decoder is not None and self.decoder == "auto":
            deconv_decoder = self.deconv_decoder
            decoder = "deconv" if deconv_decoder else "upsample"

        full_output = self.full_output
        if full_output is None:
            full_output = self.variant == "full" and not deconv_decoder
        return decoder, deconv_decoder, full_output

    @nn.compact
    def __call__(self, x):
        decoder, deconv_decoder, full_output = self._decoder_config()
        return DreamHourglassMultiStage(
            self.num_keypoints,
            n_stages=self.n_stages,
            decoder=decoder,
            deconv_decoder=deconv_decoder,
            full_output=full_output,
            skip_connections=self.skip_connections,
            output_scale=self.variant,
            internalize_spatial_softmax=self.internalize_spatial_softmax,
            learned_beta=self.learned_beta,
            initial_beta=self.initial_beta,
            name="dream",
        )(x)


class DreamTIPS(nn.Module):
    num_keypoints: int
    variant: str = "quarter"
    decoder: str = "auto"
    tips_variant: str = "tips_v2_b14"
    freeze_encoder: bool = True

    def setup(self):
        if self.variant not in {"quarter", "half", "full"}:
            raise ValueError(f"unknown variant: {self.variant}")
        if self.decoder not in {"auto", "dpt"}:
            raise ValueError(f"TIPS supports decoder='auto' or decoder='dpt', got {self.decoder}")

    def _dpt_decoder(self, x, cfg, out_h: int, out_w: int, stages):
        readouts, _ = TipsReadoutEncoder(
            variant=cfg.variant,
            pooling=cfg.pooling,
            num_cls_tokens=cfg.num_cls_tokens,
            posembs=tuple(cfg.positional_embedding.shape),
            freeze=self.freeze_encoder,
            name="tips",
        )(x, train=False)
        feats = tuple(
            nn.Conv(256, (1, 1), padding="SAME", name=f"tips_dpt_proj_{i}")(feat) for i, feat in enumerate(readouts)
        )
        for i, feat in enumerate(feats):
            stages.append((f"tips_dpt_proj_{i}", feat))

        y = DPTHead(DPTConfig(feature_dim=256, out_channels=64), name="tips_dpt_head")(feats)
        if y.shape[1:3] != (out_h, out_w):
            y = jax.image.resize(y, (y.shape[0], out_h, out_w, y.shape[-1]), method="bilinear")
        stages.append(("tips_dpt_decoder", y))
        return y

    @nn.compact
    def __call__(self, x):
        cfg = tips_model_config.get_config(self.tips_variant)
        b, h, w, _ = x.shape
        out_h, out_w = _variant_out_size(self.variant, h, w)
        stages = []

        if self.decoder == "dpt":
            y = self._dpt_decoder(x, cfg, out_h, out_w, stages)
            heatmaps = HeatmapHead(self.num_keypoints, name="heads_0")(y)
            stages.append(("heatmaps", heatmaps))
            mask = MaskHead(name="mask_head")(y)
            stages.append(("mask", mask))
            return {"heatmaps": heatmaps, "mask": mask}, tuple((name, arr.shape) for name, arr in stages)

        enc = tips.VisionEncoder(
            variant=cfg.variant,
            pooling=cfg.pooling,
            num_cls_tokens=cfg.num_cls_tokens,
            posembs=tuple(cfg.positional_embedding.shape),
            name="tips",
        )
        spatial, _ = enc(x, train=False)
        if self.freeze_encoder:
            spatial = jax.lax.stop_gradient(spatial)

        stages = [("tips_spatial", spatial)]

        y = nn.LayerNorm(name="tips_ln")(spatial)
        y = nn.Conv(256, (1, 1), padding="SAME", name="tips_adapter")(y)
        y = nn.relu(y)
        y = jax.image.resize(y, (b, out_h, out_w, 256), method="bilinear")
        stages.append(("tips_resize", y))

        y = nn.Conv(128, (3, 3), padding="SAME", name="decode_conv1")(y)
        y = nn.relu(y)
        y = nn.Conv(64, (3, 3), padding="SAME", name="decode_conv2")(y)
        y = nn.relu(y)
        heatmaps = HeatmapHead(self.num_keypoints, name="heads_0")(y)
        stages.append(("heatmaps", heatmaps))
        return heatmaps, tuple((name, arr.shape) for name, arr in stages)


class TipsReadoutEncoder(nn.Module):
    variant: str
    pooling: str = "tok"
    num_cls_tokens: int = 2
    posembs: tuple[int, int] = (16, 16)
    readout_layers: tuple[int, ...] = ()
    freeze: bool = False

    def setup(self):
        self.encoder = vit.ViT(
            variant=self.variant,
            num_cls_tokens=self.num_cls_tokens,
            posembs=self.posembs,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            stochastic_depth=0.0,
        )
        self.patches = self.encoder.patches

    def _layers(self) -> tuple[int, ...]:
        if self.readout_layers:
            return self.readout_layers
        n = self.encoder.num_layers
        return (n // 4 - 1, n // 2 - 1, (3 * n) // 4 - 1, n - 1)

    def _pool_features(self, x):
        if self.pooling != "tok":
            raise ValueError(f"Invalid pooling: {self.pooling}")
        return x[:, self.num_cls_tokens :, :], x[:, : self.num_cls_tokens, :]

    def _reshape_spatial(self, x, h: int, w: int):
        fh = h // self.patches[0]
        fw = w // self.patches[1]
        b, n, c = x.shape
        if n != fh * fw:
            raise ValueError(f"Invalid shape: {x.shape}")
        return x.reshape(b, fh, fw, c)

    @nn.compact
    def __call__(self, x, *, train: bool):
        x = vit.maybe_center_pad(x, patch_h=self.patches[0], patch_w=self.patches[1])
        h, w = x.shape[1:3]
        x = self.encoder.token_fn(x)

        readouts = []
        readout_layers = set(self._layers())
        for i, block in enumerate(self.encoder.transformer.encoder_blocks):
            x = block(x, deterministic=not train)
            if i in readout_layers:
                y = self.encoder.norm(x)
                patch, _ = self._pool_features(y)
                patch = self._reshape_spatial(patch, h, w)
                readouts.append(patch)

        x = self.encoder.norm(x)
        _, cls = self._pool_features(x)
        if self.freeze:
            readouts = [jax.lax.stop_gradient(y) for y in readouts]
            cls = jax.lax.stop_gradient(cls)
        return tuple(readouts), cls
