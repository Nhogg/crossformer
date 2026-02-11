from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

from einops import rearrange
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.diffusion import (
    cosine_beta_schedule,
    create_diffusion_model,
)
from crossformer.model.components.transformer import MAPHead
from crossformer.utils.mytyping import PRNGKey

from .adj.cart import get_fwd_kin_fn, get_jac_fn


class ActionHead(ABC):
    """Action prediction modules that take in the transformer token outputs and predict actions.

    Each action head here does chunked action prediction: i.e. at every timestep, it tries to predict the next
    `action_horizon` actions into the future from that timestep.  Setting `action_horizon=1` corresponds to
    the typical action prediction setup.
    """

    @abstractmethod
    def loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        raise NotImplementedError

    @abstractmethod
    def predict_action(
        self,
        transformer_outputs: dict[str, TokenGroup],
        argmax: bool = False,
        sample_shape: tuple[int, ...] = (),
        rng: PRNGKey | None = None,
        temperature: float = 1.0,
        train: bool = False,
        embodiment_action_dim: int | None = None,
    ) -> Array:
        """Predict the action for the last timestep in the window. Returns shape
        (*sample_shape, batch_size, action_horizon, action_dim).
        """
        raise NotImplementedError


def masked_mean(x, mask):
    mask = jnp.broadcast_to(mask, x.shape)
    return jnp.mean(x * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)


def continuous_loss(
    pred_value: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
    loss_type: str = "mse",
) -> Array:
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if loss_type == "mse":
        loss = jnp.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = jnp.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    loss = masked_mean(loss, mask)

    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)

    sign_deltas = jnp.logical_or(
        jnp.logical_and(ground_truth_value > 0, pred_value <= 0),
        jnp.logical_and(ground_truth_value <= 0, pred_value > 0),
    )
    lsign = masked_mean(sign_deltas, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
        "lsign": lsign,
    }


class ContinuousActionHead(nn.Module, ActionHead):
    """Predicts continuous actions (as opposed to discretized).

    Continuous actions are predicted by tanh squashing the model output to [-max_action, max_action], and then
    optimized using a standard regression loss.

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str
    pool_strategy: str = "mean"
    action_horizon: int = 1
    action_dim: int = 7
    clip_pred: bool = True
    max_action: float = 5.0
    loss_type: str = "mse"
    num_preds: int = 0
    loss_weight: float = 1.0
    constrain_loss_dims: bool = True

    def setup(self):
        if self.pool_strategy == "use_map":
            self.map_head = MAPHead()

        num_preds = self.num_preds if self.num_preds else self.action_horizon * self.action_dim
        self.mean_proj = nn.Dense(num_preds)

    def __call__(self, transformer_outputs: dict[str, TokenGroup], train: bool = True) -> jax.Array:
        """
        Returns:
            mean: Predicted actions w/ shape (batch_size, window_size, action_horizon, action_dim)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.pool_strategy == "use_map":  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        elif self.pool_strategy == "mean":  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        elif self.pool_strategy == "pass":
            embeddings = token_group.tokens
        else:
            raise ValueError(f"{self.pool_strategy} not implemented!")

        if len(embeddings.shape) == 3:
            # Implies embeddings is (batch_size, window_size, embedding_size)
            mean = self.mean_proj(embeddings)
            mean = rearrange(mean, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim)
        else:
            # Assumes embeddings is (batch_size, window_size, H, embedding_size)
            assert embeddings.shape[-2] == self.action_horizon
            mean = self.mean_proj(embeddings)

        if self.clip_pred:
            mean = jnp.tanh(mean / self.max_action) * self.max_action

        return mean

    def loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        action_head_mask: ArrayLike | None = None,
        train: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        """Computes the loss for the action regression objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, window_size, action_horizon, action_dim)
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension
            action_head_mask: boolean array (batch,) which is True if the action space corresponds to this specific head

        Returns:
            loss: float
            metrics: dict
        """
        if self.constrain_loss_dims:
            # when using separate heads we can constrain the loss to the action dimensions and action horizon specific to this head
            actions = actions[:, :, : self.action_horizon, : self.action_dim]
            action_pad_mask = action_pad_mask[:, :, : self.action_horizon, : self.action_dim]

        # (batch, window_size, action_horizon, action_dim)
        mean = self(transformer_outputs, train=train)

        if action_head_mask is None:
            action_head_mask = jnp.ones(mean.shape[0], dtype=bool)

        # combine the timestep pad mask with the action pad mask and the action head mask
        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask & action_head_mask[:, None, None, None]

        loss, metrics = continuous_loss(mean, actions, mask, loss_type=self.loss_type)
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: dict[str, TokenGroup],
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> jax.Array:
        """Convenience methods for predicting actions for the final timestep in the window."""
        # only get the last timestep in the window
        mean = self(transformer_outputs, train=train)  # [:, -1]
        return jnp.broadcast_to(mean, sample_shape + mean.shape)


class L1ActionHead(ContinuousActionHead):
    loss_type: str = "l1"


########
# FROM OCTO
# Diffusion Action Head
########


class MSEActionHead(ContinuousActionHead):
    loss_type: str = "mse"
    pool_strategy: str = "use_map"


class DiffusionActionHead(nn.Module):
    """Predicts actions uses a diffusion process.

    Only a single pass through the transformer is done to obtain an action embedding at each timestep. The
    action is then predicted using a diffusion process conditioned on this embedding. The diffusion model
    architecture is an MLP with residual connections (see `octo.model.components.diffusion`).

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str
    pool_strategy: str = "mean"  # replaces use_map
    action_horizon: int = 1  # replaces pred_horizon
    action_dim: int = 7
    clip_pred: bool = True
    max_action: float = 5.0
    loss_type: str = "mse"
    num_preds: int = 0
    loss_weight: float = 1.0
    constrain_loss_dims: bool = True

    # diffusion-specific config with sane defaults
    time_dim: int = 32
    num_blocks: int = 3
    dropout_rate: float = 0.1
    hidden_dim: int = 256
    use_layer_norm: bool = True
    diffusion_steps: int = 20

    def setup(self):
        if self.pool_strategy == "use_map":
            self.map_head = MAPHead()

        # create the diffusion model (score network)
        self.diffusion_model = create_diffusion_model(
            self.action_dim * self.action_horizon,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
        )

        # create beta schedule
        self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
        self.alphas = 1 - self.betas
        self.alpha_hats = jnp.array([jnp.prod(self.alphas[: i + 1]) for i in range(self.diffusion_steps)])

    def __call__(
        self,
        transformer_outputs: dict[str, TokenGroup],
        time: ArrayLike | None = None,
        noisy_actions: ArrayLike | None = None,
        train: bool = True,
    ) -> jax.Array:
        """Performs a single forward pass through the diffusion model."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )

        if self.pool_strategy == "use_map":  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        elif self.pool_strategy == "mean":  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        elif self.pool_strategy == "pass":
            embeddings = token_group.tokens
        else:
            raise ValueError(f"{self.pool_strategy} not implemented!")

        # Now, embeddings is (batch_size, window_size, embedding_size)

        # time and noisy_actions are None during initialization, so we replace them with a dummy array
        if (time is None or noisy_actions is None) and not self.is_initializing():
            raise ValueError("Must provide time and noisy_actions when calling diffusion action head")
        elif self.is_initializing():
            time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
            noisy_actions = jnp.zeros(
                (*embeddings.shape[:2], self.action_dim * self.action_horizon),
                dtype=jnp.float32,
            )

        pred_eps = self.diffusion_model(embeddings, noisy_actions, time, train=train)
        return pred_eps

    def loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        action_head_mask: ArrayLike | None = None,
        train: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        """Computes the loss for the diffusion objective.

        Args:
            actions: shape (batch_size, window_size, action_horizon, action_dim)
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension
            action_head_mask: boolean array (batch,) which is True if the action space corresponds to this specific head

        Returns:
            loss: float
            metrics: dict
        """

        batch_size, window_size = timestep_pad_mask.shape
        ### dont need to do chunking here like octo

        if self.constrain_loss_dims:
            # when using separate heads we can constrain the loss to the action
            # dimensions and action horizon specific to this head

            # pred_eps = pred_eps[:, :, : self.action_horizon, : self.action_dim]
            # noise = noise[:, :, : self.action_horizon, : self.action_dim]
            actions = actions[:, :, : self.action_horizon, : self.action_dim]
            action_pad_mask = action_pad_mask[:, :, : self.action_horizon, : self.action_dim]

        # fold action_dim and action_horizon into one dimension
        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

        # piggy-back on the dropout rng chain for diffusion rng
        rng = self.make_rng("dropout")
        time_key, noise_key = jax.random.split(rng)
        time = jax.random.randint(time_key, (batch_size, window_size, 1), 0, self.diffusion_steps)
        noise = jax.random.normal(noise_key, actions_flat.shape)

        alpha_hat = self.alpha_hats[time]
        alpha_1 = jnp.sqrt(alpha_hat)
        alpha_2 = jnp.sqrt(1 - alpha_hat)
        noisy_actions = alpha_1 * actions_flat + alpha_2 * noise

        pred_eps = self(transformer_outputs, train=train, time=time, noisy_actions=noisy_actions)

        if action_head_mask is None:
            action_head_mask = jnp.ones(pred_eps.shape[0], dtype=bool)

        # combine the timestep pad mask with the action pad mask and the action head mask
        mask = rearrange(
            (
                timestep_pad_mask[:, :, None, None]  # dimension reduced because noise is b w (h a)
                & action_pad_mask
                & action_head_mask[:, None, None, None]
            ),
            "b w h a -> b w (h a)",
        )

        loss, metrics = continuous_loss(pred_eps, noise, mask, loss_type=self.loss_type)
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: dict[str, TokenGroup],
        rng: PRNGKey,
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> jax.Array:
        """Convenience methods for predicting actions for the final timestep in the window."""
        module, variables = self.unbind()

        def scan_fn(carry, time):
            current_x, rng = carry
            input_time = jnp.broadcast_to(time, (*current_x.shape[:-1], 1))

            eps_pred = module.apply(variables, transformer_outputs, input_time, current_x, train=train)

            alpha_1 = 1 / jnp.sqrt(self.alphas[time])
            alpha_2 = (1 - self.alphas[time]) / (jnp.sqrt(1 - self.alpha_hats[time]))
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(key, shape=current_x.shape)
            current_x = current_x + (time > 0) * (jnp.sqrt(self.betas[time]) * z)

            current_x = jnp.clip(current_x, -self.max_action, self.max_action)

            return (current_x, rng), ()

        def sample_actions(rng):
            rng, key = jax.random.split(rng)
            batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]

            (actions_flat, _), () = jax.lax.scan(
                scan_fn,
                (
                    jax.random.normal(
                        key,
                        (
                            batch_size,
                            window_size,
                            self.action_horizon * self.action_dim,
                        ),
                    ),
                    rng,
                ),
                jnp.arange(self.diffusion_steps - 1, -1, -1),
            )

            actions = rearrange(
                actions_flat,
                "b w (h a) -> b w h a",
                h=self.action_horizon,
                a=self.action_dim,
            )
            # only get the last timestep in the window
            # return actions[:, -1]
            return actions

        n_samples = int(np.prod(sample_shape))
        actions = jax.vmap(sample_actions)(jax.random.split(rng, n_samples))
        actions = actions.reshape(sample_shape + actions.shape[1:])
        return actions


def sample_tau(key: jax.Array, shape=(), s: float = 0.999) -> jnp.ndarray:
    """
    Sample timesteps τ ~ Beta((s-τ)/s; alpha=1.5, beta=1)

    Args:
        key: JAX PRNGKey
        shape: output shape, e.g. (N,) for N samples
        s: cutoff parameter, default 0.999

    Returns:
        τ samples in [0, s]
    """
    alpha, beta = 1.5, 1.0
    # sample x ~ Beta(alpha, beta) on [0,1]
    x = jax.random.beta(key, alpha, beta, shape=shape)
    # map back: τ = s * (1 - x)
    tau = s * (1.0 - x)
    return tau


class FlowMatchingActionHead(ContinuousActionHead):
    """Flow-matching head that predicts conditional action velocities."""

    time_dim: int = 32
    num_blocks: int = 3
    dropout_rate: float = 0.1
    hidden_dim: int = 256
    use_layer_norm: bool = True
    flow_steps: int = 10
    base_std: float = 1.0

    def setup(self):
        if self.pool_strategy == "use_map":
            self.map_head = MAPHead()

        self.flow_model = create_diffusion_model(
            self.action_dim * self.action_horizon,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
        )

    def _embed(self, transformer_outputs: dict[str, TokenGroup], train: bool) -> jax.Array:
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            "Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        print(transformer_outputs.keys())
        print("token_group.tokens", token_group.tokens.shape)

        if self.pool_strategy == "use_map":
            return self.map_head(token_group, train=train)[:, :, 0]
        if self.pool_strategy == "mean":
            return token_group.tokens.mean(axis=-2)
        if self.pool_strategy == "pass":
            return token_group.tokens
        raise ValueError(f"{self.pool_strategy} not implemented!")

    def __call__(
        self,
        transformer_outputs: dict[str, TokenGroup],
        time: ArrayLike | None = None,
        a_t: ArrayLike | None = None,
        train: bool = True,
    ) -> jax.Array:
        embeddings = self._embed(transformer_outputs, train=train)
        print("embeddings", embeddings.shape)
        print("\n" * 10)

        if (time is None or a_t is None) and not self.is_initializing():
            raise ValueError("Must provide time and current action when calling flow head")
        if self.is_initializing():
            time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
            a_t = jnp.zeros(
                (*embeddings.shape[:2], self.action_dim * self.action_horizon),
                dtype=jnp.float32,
            )

        if a_t.ndim == 4:
            a_t = rearrange(a_t, "b w h a -> b w (h a)")

        return self.flow_model(embeddings, a_t, time, train=train)

    def flow_loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        action_head_mask: ArrayLike | None = None,
        train: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        if self.constrain_loss_dims:
            actions = actions[:, :, : self.action_horizon, : self.action_dim]
            action_pad_mask = action_pad_mask[:, :, : self.action_horizon, : self.action_dim]

        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

        rng = self.make_rng("dropout")
        base_key, time_key = jax.random.split(rng)
        base = self.base_std * jax.random.normal(base_key, actions_flat.shape)

        # deprecated in favor of beta distributed time
        # time = jax.random.uniform(time_key, (*actions_flat.shape[:2], 1))
        time = sample_tau(time_key, shape=(*actions_flat.shape[:2], 1), s=0.99)

        blended = time * actions_flat + (1.0 - time) * base
        target = actions_flat - base

        pred = self(
            transformer_outputs,
            time=time,
            a_t=blended,
            train=train,
        )

        if action_head_mask is None:
            action_head_mask = jnp.ones(pred.shape[0], dtype=bool)

        mask = rearrange(
            (timestep_pad_mask[:, :, None, None] & action_pad_mask & action_head_mask[:, None, None, None]),
            "b w h a -> b w (h a)",
        )

        loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")
        return loss, metrics

    def loss(self, *args, **kwargs) -> tuple[Array, dict[str, Array]]:
        return self.flow_loss(*args, **kwargs)

    def predict_action(
        self,
        transformer_outputs: dict[str, TokenGroup],
        rng: PRNGKey,
        train: bool = True,
        *args,
        sample_shape: tuple[int, ...] = (),
        **kwargs,
    ) -> jax.Array:
        module, variables = self.unbind()

        def sample_actions(rng):
            rng, key = jax.random.split(rng)
            tokens = transformer_outputs[self.readout_key].tokens
            batch_size, window_size = tokens.shape[:2]
            a_t = self.base_std * jax.random.normal(
                key,
                (
                    batch_size,
                    window_size,
                    self.action_horizon * self.action_dim,
                ),
            )

            dt = 1.0 / max(self.flow_steps, 1)

            def scan_fn(a_t, step):
                t = (step + 0.5) * dt
                time = jnp.full((*a_t.shape[:2], 1), t, dtype=a_t.dtype)
                velocity = module.apply(
                    variables,
                    transformer_outputs,
                    time,
                    a_t,
                    train=train,
                )
                updated = a_t + dt * velocity
                if self.clip_pred:
                    updated = jnp.clip(updated, -self.max_action, self.max_action)
                return updated, ()

            steps = jnp.arange(self.flow_steps)
            a_t, _ = jax.lax.scan(scan_fn, a_t, steps)

            actions = rearrange(
                a_t,
                "b w (h a) -> b w h a",
                h=self.action_horizon,
                a=self.action_dim,
            )
            if self.clip_pred:
                actions = jnp.clip(actions, -self.max_action, self.max_action)
            return actions

        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        samples = jax.vmap(sample_actions)(jax.random.split(rng, n_samples))
        samples = samples.reshape(sample_shape + samples.shape[1:])
        return samples


def cart_flow_loss(qt: jax.Array, q1: jax.Array, t: jax.Array, qvh: jax.Array, fk: Callable, J_fn: Callable):
    """
    Use q to denote joint space and x to denote Cartesian/task space.
    D = dof dimension, M = Cartesian/task dimension (e.g., 3 or 6).
    """

    J = J_fn(qt)
    xvh = jnp.einsum("...md,...d->...m", J, qvh)
    xt = fk(qt)
    x1 = fk(q1)
    dx = x1 - xt
    xv = dx / (dt := (1.0 - t))

    return xvh, xv


def _cart_flow_adjustor(
    batch,
    flows,
    timestep_pad_mask,
    action_pad_mask,
    action_head_mask,
    fk: Callable,
    J_fn: Callable,
):
    # actions_flat = rearrange(actions, "b w h a -> b w (h a)")
    q1 = batch["action"]
    qt = rearrange(flows["blended"], "b w (h a) -> b w h a", h=q1.shape[-2], a=q1.shape[-1])
    q0 = rearrange(flows["base"], "b w (h a) -> b w h a", h=q1.shape[-2], a=q1.shape[-1])
    qvh = rearrange(flows["pred"], "b w (h a) -> b w h a", h=q1.shape[-2], a=q1.shape[-1])
    qt = qt[..., :-1]  # no gripper
    q1 = q1[..., :-1]  # no gripper
    q0 = q0[..., :-1]  # no gripper
    qvh = qvh[..., :-1]  # no gripper
    t = jnp.expand_dims(flows["time"], axis=-1)

    if True:
        x1 = fk(q1)
        xt = fk(qt)
        x0 = fk(q0)
        xvh = fk(q0 + qvh) - fk(q0)  # task space displacement
        xv = x1 - x0

    pred, target = xvh, xv  # pred, target = cart_flow_loss(qt, q1, t, qvh, fk, J_fn)
    if action_head_mask is None:
        action_head_mask = jnp.ones(pred.shape[0], dtype=bool)

    mask = timestep_pad_mask[:, :, None, None] & action_pad_mask & action_head_mask[:, None, None, None]
    mask = rearrange(mask[..., : pred.shape[-1]], "b w h a -> b w (h a)")
    pred, target = rearrange(pred, "b w h a -> b w (h a)"), rearrange(target, "b w h a -> b w (h a)")

    loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")
    return loss, metrics


# robot = pk.Robot.from_urdf(adj.cart.urdf)
class AdjFlowHead(FlowMatchingActionHead):
    """for FLow Adjustments"""

    wf: float = 0.5  # flow loss weight
    wa: float = 0.5  # adjustor loss weight
    # robot: pk.Robot = pk.Robot.from_urdf(adj.cart.urdf)
    # robot_fn: Callable= lambda :

    def setup(self):
        super().setup()
        self.adjustor = partial(
            _cart_flow_adjustor,
            fk=get_fwd_kin_fn(robot, pad_gripper=True),
            J_fn=get_jac_fn(robot, pad_gripper=True),
        )

    def do_flow(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        if self.constrain_loss_dims:
            actions = actions[:, :, : self.action_horizon, : self.action_dim]
            action_pad_mask = action_pad_mask[:, :, : self.action_horizon, : self.action_dim]

        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

        rng = self.make_rng("dropout")
        base_key, time_key = jax.random.split(rng)
        base = self.base_std * jax.random.normal(base_key, actions_flat.shape)

        # deprecated in favor of beta distributed time
        # time = jax.random.uniform(time_key, (*actions_flat.shape[:2], 1))
        time = sample_tau(time_key, shape=(*actions_flat.shape[:2], 1), s=0.99)

        blended = time * actions_flat + (1.0 - time) * base
        target = actions_flat - base

        pred = self(
            transformer_outputs,
            time=time,
            a_t=blended,
            train=train,
        )

        return {
            "pred": pred,
            "target": target,
            "time": time,
            "base": base,
            "blended": blended,
            "action_pad_mask": action_pad_mask,
        }

    def flow_loss(
        self,
        pred,
        target,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        action_head_mask: ArrayLike | None = None,
        train: bool = True,
    ):
        if action_head_mask is None:
            action_head_mask = jnp.ones(pred.shape[0], dtype=bool)

        mask = rearrange(
            (timestep_pad_mask[:, :, None, None] & action_pad_mask & action_head_mask[:, None, None, None]),
            "b w h a -> b w (h a)",
        )

        loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")
        return loss, metrics

    def loss(self, *, embeddings, batch, train=True) -> tuple[Array, dict[str, Array]]:
        timestep_pad_mask = batch["observation"]["timestep_pad_mask"].astype(bool)
        action_pad_mask = batch["action_pad_mask"].astype(bool)
        action_head_mask = batch["action_head_masks"]["single_arm"].astype(bool)

        # pprint(action_head_mask)
        # pprint(batch['info']['loss'])
        # pprint((spec(batch['info']['loss']),spec(action_head_mask)))

        flows = self.do_flow(embeddings, batch["action"], action_pad_mask=action_pad_mask, train=train)
        use_flow_head = action_head_mask & batch["mask"]["only_adjustment"]
        lf, fmetrics = self.flow_loss(
            flows["pred"],
            flows["target"],
            timestep_pad_mask=timestep_pad_mask,
            action_pad_mask=flows["action_pad_mask"],
            # if only adjustment then zero out flow loss
            action_head_mask=use_flow_head,
            train=train,
        )
        la, ametrics = self.adjustor(
            batch,
            flows,
            timestep_pad_mask=timestep_pad_mask,
            action_pad_mask=flows["action_pad_mask"],
            action_head_mask=action_head_mask,
        )
        l = (lf * self.wf) + (la * self.wa)
        return l, {"flow": fmetrics, "adj": ametrics, "combined": {"lf": lf, "la": la, "total": l}}
