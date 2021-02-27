from typing import NamedTuple, Optional

import numpy as np


class ModelParameters(NamedTuple):
    encoder_layers: np.ndarray = np.array([32, 32, 32])
    decoder_layers: Optional[np.ndarray] = np.array([])
    mirror_encoder: bool = True


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AppState(metaclass=Singleton):
    def __init__(self):
        super().__init__()
        self.model_parameters: ModelParameters = ModelParameters()


if __name__ == '__main__':
    AppState()