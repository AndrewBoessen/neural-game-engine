# Neural Game Engine

_Neural network approach for modeling interactive game environments using a VQ-VAE and Spatio-Temporal Transformer. Trained on Atari Skiing gameplay data._

| ![SkiingGIF](./assets/original.gif) | ![Generated](./assets/game.gif) |
| :---------------------------------: | :-----------------------------: |
|              Original               |          AI Generated           |

## Install

1. Clone Repo

```
git clone https://github.com/AndrewBoessen/neural-game-engine.github
cd neural-game-engine
```

2. Create Conda Environment

```
conda create -n engine python=3.10
conda activate engine
```

3. Install Dependencies

```
pip install -r requirements.txt
```

## Load Checkpoints and Data

### Pretrained Model Checkpoints

Download model checkpoints and move to root directory

|                                          VQ-VAE Checkpoint                                          |                                    Neural Game Engine Checkpoint                                    |
| :-------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: |
| [Download Here](https://drive.google.com/file/d/1xIec8GLG2CwhUb2dGMrpVb2z1NoUrjHJ/view?usp=sharing) | [Download Here](https://drive.google.com/file/d/1exsjhvskQ48hqWKFC-quVvV3ftBuZ4cW/view?usp=sharing) |

### Gameplay Dataset

The model is trained on a dataset of ~33,000 frames and evaluated on a set of ~8,000 frames.

To train the transformer game-engine model, the dataset is preprocessed and tokenized

https://github.com/user-attachments/assets/b91b7eef-b018-4bfa-99d8-05ac5def9104

#### Download Datasets

|                                          Gameplay Dataset                                           |                                            Token Dataset                                            |
| :-------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: |
| [Download Here](https://drive.google.com/file/d/1mr900bK0xpwiQskSB4KJvwtrbwtnEJcY/view?usp=sharing) | [Download Here](https://drive.google.com/file/d/19UJVwnnpArB_rG6F4Jn3TTqxKfn04mhD/view?usp=sharing) |
|                                    Extract to `/gameplay_data/`                                     |                                      Extract to `/token_data/`                                      |

## Play

An interactive game script is available that generates frames based on user input

https://github.com/user-attachments/assets/58f30664-9cac-4037-8458-acb96b8ba519

Example gameplay recording. Running on Nvidia RTX 4070 at 15fps

### Run Interactive Game Environment

Follow installation instructions above to install token data and model checkpoints

```
python play.py
```

## Architecture

The Neural Game Engine architecture leverages a combination of a Vector Quantized Variational Auto-Encoder (VQ-VAE) for image tokenization and a Spatio-Temporal Transformer (ST-Transformer) for modeling game dynamics. This design enables the simulation of interactive game environments by capturing the causal relationships between user actions and game state transitions.

![neural game engine](./assets/architecture.png)

*Interactive Game Engine. An RL agent is used to create a dataset consisting of observation, action pairs. The observations are encoded into states with the VAE encoder E. The sequential model (ST-Transformer) takes the encoded state, action pairs and predicts the next state s_t+1. The state to be predicted, initially represented as a mask token, \[MASK\], is iteratively generated in a non-auto regressive manner with MaskGIT and bidirectional attention. Predicted states are projected to pixel space with the VAE decoder D.*

### Data Collection (RL Agent)
A reinforcement learning (RL) agent is employed to collect gameplay data, generating a dataset of observation-action pairs. The agent interacts with the environment and records its trajectories, simulating diverse scenarios. This dataset, consisting of approximately 33,000 training frames and 8,000 validation frames, serves as the foundation for training both the image tokenizer and the game engine.

### Image Tokenizer (VQ-VAE)
The VQ-VAE encodes game frames into a discrete latent space, forming a tokenized representation of the image. It uses:
- **Encoder**: Converts 256×256 RGB images into a 16×16 grid of tokens, reducing spatial complexity.
- **Codebook**: Contains 512 unique tokens used for quantization.
- **Decoder**: Reconstructs images from the tokenized representations.
This compression allows the ST-Transformer to operate on discrete tokens rather than raw pixel data, enabling efficient sequence modeling.

### Game Engine (ST-Transformer)
The ST-Transformer predicts future game states based on past states and user actions. It processes sequences of state-action token pairs, capturing both spatial and temporal dependencies:
- **Spatial Attention**: Models relationships between tokens within a single frame.
- **Temporal Attention**: Models dependencies across multiple frames.
- **MaskGIT Algorithm**: Uses bidirectional attention and iterative refinement to generate states in parallel, reducing computation time while maintaining visual fidelity.

### State Prediction and Interactivity
The predicted tokens are fed back into the VQ-VAE decoder to generate the next game frame. User actions influence the prediction pipeline, ensuring real-time interaction and causality. The iterative non-autoregressive process minimizes quality degradation over time, though edge cases (e.g., object collisions) remain challenging.

This architecture demonstrates the feasibility of neural networks in simulating dynamic, visually coherent game environments with real-time responsiveness. For further details, refer to the [full technical report](./assets/technical_report.pdf).

