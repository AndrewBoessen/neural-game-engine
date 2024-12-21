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

### Gameplay Dataset

The model is trained on a dataset of ~33,000 frames and evaluated on a set of ~8,000 frames.

To train the transformer game-engine model, the dataset is preprocessed and tokenized

#### Download Datasets

|                                          Gameplay Dataset                                           |                                            Token Dataset                                            |
| :-------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: |
| [Download Here](https://drive.google.com/file/d/1mr900bK0xpwiQskSB4KJvwtrbwtnEJcY/view?usp=sharing) | [Download Here](https://drive.google.com/file/d/19UJVwnnpArB_rG6F4Jn3TTqxKfn04mhD/view?usp=sharing) |
|                                    Extract to `/gameplay_data/`                                     |                                      Extract to `/token_data/`                                      |
