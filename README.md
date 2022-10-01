# Neural Network Library from Scratch
Custom neural network library from scratch inspired by [PyTorch](https://github.com/pytorch/pytorch) and [Micrograd](https://github.com/karpathy/micrograd) to learn more about how neural network frameworks work under the hood. 

## Tensor
Implements main datatype with built-in autograd engine. Tensor datatype is wrapper around numpy array.

### Usage
```python
from tensor import Tensor

x = Tensor.randn(shape=(16, 16))
    f = Tensor.ones(shape=(3, 3))
    y = x.convolution(f)
    z = y.tanh()

    # shape [14, 14]
    print(z.shape)

    w = Tensor.ones((14, 3))

    out = (z.dot(w)).relu()

    # shape [14, 3]
    print(out.shape)

    out.backward()

    print(x)
```

## NN
Implements some neural network layers. Currently, only Linear layer is implemented.