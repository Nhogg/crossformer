# CrossFormer Codebase Insights

## Overview
The CrossFormer codebase implements a transformer-based policy for cross-embodied robotics, trained on 900K trajectories from 20 diverse robot embodiments (manipulation, navigation, locomotion, aviation). It builds on the Octo framework, emphasizing modularity, scalability, and adaptability across different robot morphologies.

## Technologies & Frameworks
- **JAX/Flax Ecosystem**: Full implementation for efficient training/inference on TPUs/GPUs, enabling differentiable computations and multi-host scaling.
  - Key files: `crossformer/model/crossformer_model.py`, `crossformer/model/crossformer_module.py`
- **Google Grain**: Migration to JAX-first data loading for scalable, deterministic multi-host training with `jax.Array` compatibility.
  - Key files: `crossformer/data/grain/README.md`, `crossformer/data/grain/pipelines.py`
- **ArrayRecord (AREC) Datasets**: Custom utilities for efficient, compressed dataset building.
  - Key files: `crossformer/data/arec/README.md`, `crossformer/data/arec/arec.py`
- **Hydra-Based CN Configs**: Modular configuration system using Python dataclasses for experiment sweeps.
  - Key files: `crossformer/cn/README.md`, `crossformer/cn/base.py`

## Architectural Patterns
- **Modular Attention Structure**: Blockwise-causal design with configurable attention rules (e.g., CAUSAL, CURRENT) using fnmatch patterns on token groups.
  - Key files: `crossformer/model/components/block_transformer.py`
- **Token Group Abstraction**: Semantic groups (PrefixGroup, TimestepGroup) for multimodal inputs with associated masks/encodings.
  - Key files: `crossformer/model/components/base.py`, `crossformer/model/components/block_transformer.py`
- **Config-Driven Modularity**: Components instantiated from ModuleSpec configs with `create` methods.
  - Key files: `crossformer/utils/spec.py`, `crossformer/model/crossformer_module.py`
- **Feature-Based Organization**: Code organized by features (e.g., model/components) with shared infra layers.
  - Reference: `AGENTS.md`

## Key Components
- **CrossFormerTransformer**: Core transformer sequencing task, observation, and readout tokens with cross-modal attention.
  - Key file: `crossformer/model/crossformer_module.py:21-315`
- **Tokenizers**: Modular encoders for inputs, including ImageTokenizer with VIT, FiLM conditioning, and TokenLearner.
  - Key files: `crossformer/model/components/tokenizers.py`, `crossformer/model/components/vit_encoders.py`
- **Action Heads**: Abstractions for prediction, including DiffusionActionHead (denoising diffusion) and FlowActionHead (normalizing flows).
  - Key file: `crossformer/model/components/action_heads.py`
- **Readouts**: Special tokens for independent readout of embeddings (e.g., actions, values).
  - Key file: `crossformer/model/crossformer_module.py`

## Algorithms & Novel Approaches
- **Diffusion for Action Generation**: Denoising diffusion process conditioned on embeddings, using cosine schedules and MLP-ResNet score networks.
  - Key files: `crossformer/model/components/diffusion.py`, `crossformer/model/components/action_heads.py`
- **Cross-Modal Attention**: Task tokens repeated per timestep for attention between tasks and observations.
  - Key file: `crossformer/model/crossformer_module.py:216-238`
- **Blockwise-Causal Transformer**: Custom attention masks with group-based rules, dynamic assembly/disassembly.
  - Key file: `crossformer/model/components/block_transformer.py`
- **FiLM Conditioning and Token Learning**: Feature-wise modulation for tasks, TokenLearner for efficient token reduction.
  - Key files: `crossformer/model/components/tokenizers.py`, `crossformer/model/components/film_conditioning_layer.py`

## Notable Code Structures
- **Attention Rule Patterns**: Declarative fnmatch-based rules for flexible attention control.
  - Key file: `crossformer/model/components/block_transformer.py`
- **Pytree-Based Handling**: JAX pytrees for nested data, custom flattening for token management.
  - Key file: `crossformer/model/components/block_transformer.py`
- **Config Store with CNMeta**: Dynamic Hydra config registration via metaclass.
  - Key file: `crossformer/cn/base.py`
- **JAX-First Transforms**: Pure Python/JAX rewrites of data transforms for Grain.
  - Key files: `crossformer/data/traj_transforms.py`, `crossformer/data/obs_transforms.py`

This codebase integrates modern ML frameworks with innovative robotics algorithms, focusing on generalist policies for diverse embodiments.