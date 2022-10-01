import numpy as np
from tensor import Tensor

class Layer:
    def __init__(self):
        self.weigths = None
        self.bias = None

    def __call__(self, x: Tensor):
        pass

class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weigths = Tensor(np.random.randn(in_features, out_features))
        self.bias = Tensor(np.random.randn(out_features))
        
    def __call__(self, x: Tensor):
        return x.dot(self.weigths) + self.bias

class Conv2d(Layer):
    pass
