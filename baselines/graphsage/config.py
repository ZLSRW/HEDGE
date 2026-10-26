from dataclasses import dataclass

@dataclass
class HParams:
    lr: float = 0.001
    weight_decay: float = 0.0005
    epsilon: float = 0.0001
    epochs: int = 300
    batch_size: int = 64
    early_stop_patience: int = 40
    hidden_dim: int = 64
    dropout: float = 0.3
    num_networks: int = 1
    focal_gamma: float = 0.0

class TrainConfig:

    def __init__(self):
        self.hparams = HParams()
