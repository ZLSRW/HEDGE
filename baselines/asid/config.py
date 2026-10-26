from dataclasses import dataclass

@dataclass
class HParams:
    lr: float = 0.01
    weight_decay: float = 0.0001
    epochs: int = 500
    batch_size: int = 0
    early_stop_patience: int = 40
    hidden_dim: int = 64
    dropout: float = 0.0
    num_networks: int = 1
    focal_gamma: float = 0.0

class TrainConfig:

    def __init__(self):
        self.hparams = HParams()
