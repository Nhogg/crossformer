# CrossFormer UML Diagrams

This directory contains UML diagrams documenting the CrossFormer architecture.

## Overview

**CrossFormer** is a transformer-based robot policy trained on 900K trajectories across 20 different robot embodiments. It's a unified model for manipulation, navigation, locomotion, and aviation tasks.

## Diagrams

### 1. `crossformer_architecture.puml` - Class Diagram
A comprehensive class diagram showing:
- **High-Level API**: `CrossFormerModel` - main entry point for training/inference
- **Model Module**: `CrossFormerModule` and `CrossFormerTransformer` - core architecture
- **Tokenizers**: Vision and language tokenization components
- **Vision Encoders**: ResNet26, PatchEncoder, etc.
- **Transformer Components**: BlockTransformer with flexible attention patterns
- **Action Heads**: L1, Diffusion, and Value heads
- **Data Pipeline**: OXE dataset integration

### 2. `data_flow.puml` - Sequence Diagram
Shows the data flow through the system:
- **Data Loading**: Trajectory and frame transforms
- **Forward Pass**: Tokenization → Transformer → Action prediction
- **Training**: Loss computation and gradient updates

### 3. `model_components.puml` - Component Diagram
High-level component view showing:
- System architecture and package organization
- Component dependencies
- Infrastructure integration (HuggingFace, Orbax, W&B)

## How to View

### Option 1: Online Viewers
1. Copy the `.puml` file contents
2. Paste into [PlantUML Online Editor](http://www.plantuml.com/plantuml/uml/)
3. View the rendered diagram

### Option 2: VS Code
1. Install the "PlantUML" extension
2. Open any `.puml` file
3. Press `Alt+D` to preview

### Option 3: Command Line (requires PlantUML)
```bash
# Install PlantUML (requires Java)
sudo apt-get install plantuml

# Render to PNG
plantuml crossformer_architecture.puml
plantuml data_flow.puml
plantuml model_components.puml

# This creates .png files you can view
```

### Option 4: Docker
```bash
# Render all diagrams using Docker
docker run --rm -v $(pwd):/data plantuml/plantuml *.puml
```

## Key Architecture Highlights

### 1. Multi-Modal Conditioning
- **Language**: Natural language task descriptions
- **Goal Images**: Visual goal specifications
- Both processed through separate tokenizers and combined in the transformer

### 2. Blockwise-Causal Attention
The `BlockTransformer` implements flexible attention patterns:
- **PrefixGroup** (task tokens): Visible to all but don't attend to observations
- **TimestepGroup** (observation tokens): Causal attention across timesteps
- **Readout tokens**: Extract action predictions from embeddings

### 3. Cross-Embodiment Support
Multiple action head types for different robots:
- `single_arm`: 7-DoF manipulation
- `bimanual`: Dual-arm manipulation
- `nav`: Mobile base navigation
- `quadruped`: Legged locomotion

### 4. Modular Tokenization
Different tokenizers for different input modalities:
- **ImageTokenizer**: Encodes camera images (RGB, depth, etc.)
- **LowdimObsTokenizer**: Encodes proprioceptive state (joint positions, velocities)
- **TextProcessor**: Encodes language instructions

### 5. Data Pipeline
Built on Open X-Embodiment (OXE) datasets:
- Trajectory-level transforms: chunking, goal relabeling, task augmentation
- Frame-level transforms: image augmentation, normalization
- Transitioning from TensorFlow to JAX-native Google Grain pipeline

## Model Statistics

- **Parameters**: 130M
- **Training Data**: 900K robot trajectories
- **Embodiments**: 20 different robots
- **Pre-trained Model**: Available at `hf://rail-berkeley/crossformer`
- **Base Framework**: JAX + Flax

## Directory Structure

```
crossformer/
├── model/
│   ├── crossformer_model.py      # High-level API
│   ├── crossformer_module.py     # Core module
│   └── components/               # Building blocks
│       ├── tokenizers.py         # Input tokenization
│       ├── block_transformer.py  # Attention mechanism
│       ├── action_heads.py       # Output heads
│       └── vit_encoders.py       # Vision encoders
├── data/
│   ├── dataset.py                # Dataset utilities
│   ├── oxe/                      # OXE integration
│   └── grain/                    # New JAX data pipeline
└── utils/                        # Training utilities
```

## References

- Built on top of the [Octo](https://github.com/rail-berkeley/octo) codebase
- Uses [Open X-Embodiment](https://robotics-transformer-x.github.io/) datasets
- Implements ideas from various transformer and vision-language papers

## Development

For development setup and installation instructions, see:
- `../README.md` - Main project README
- `../INSTALL.md` - Installation guide
- `../docs/` - Detailed documentation

---

**Note**: These diagrams are automatically generated from the codebase structure as of 2026-01-22. For the most up-to-date architecture details, refer to the source code.
