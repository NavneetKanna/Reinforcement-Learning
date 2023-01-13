from dlgrad_main.dlgrad.mlp import MLP
from dlgrad_main.dlgrad.afu import ReLU
import numpy as np

        
class DQNNet:
    def __init__(self, input_shape: int, output_shape: int) -> None:
        self.fc1 = MLP(input_shape, 200, bias=True)
        self.fc2 = MLP(200, output_shape, bias=True)

    def forward(self, data):
        x = self.fc1(data)
        x = ReLU(x)
        x = self.fc2(x)

        return x


