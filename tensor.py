import numpy as np
from scipy.signal import correlate2d

class Tensor:
    """
    Engine for autograd
    """

    def __init__(self, data: np.ndarray | list, dtype: np.dtype = np.float32, _children: tuple=(), _op: str =""):
        """
        Custom engine for autograd. Tensor is a wrapper class around numpy array, which can perform autodifferentiation.

        :param np.ndarray | list data: Input data
        :param np.dtype dtype: data type, defaults to np.float32
        :param tuple _children: container for children nodes, defaults to ()
        :param str _op: operation used to produce this tensor, defaults to ""
        """
        self.data = np.array(data, dtype=dtype)
        # initializing gradient as zero
        self.grad = np.zeros_like(self.data)

        self.shape = self.data.shape

        # backward function
        self._backward = lambda: None
        # children nodes
        self._children = set(_children)
        # operation used to produce this tensor
        self._op = _op

    def __repr__(self):
        return f"Tensor(\ndata:\n{self.data}\ngrad:\n{self.grad}\n)"

    # INITIALIZATIONS
    @staticmethod
    def randn(shape):
        return Tensor(np.random.randn(*shape))

    @staticmethod
    def rand(shape):
        return Tensor(np.random.rand(*shape))

    @staticmethod
    def zeros(shape):
        return Tensor(np.zeros(shape))

    @staticmethod
    def ones(shape):
        return Tensor(np.ones(shape))

    # ELEMENT-WISE OPERATIONS
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += output.grad

            # if other has different dims we squeeze output grad using sum
            dim_diff = output.grad.ndim - other.grad.ndim
            other.grad += output.grad if dim_diff == 0 else np.sum(output.grad, axis=tuple(i for i in range(dim_diff)))

        output._backward = _backward
        return output

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        output = Tensor(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * output.grad

            other_grad = self.data * output.grad

            # if other has different dims we squeeze output grad using sum
            dim_diff = output.grad.ndim - other.grad.ndim
            other.grad += other_grad if dim_diff == 0 else np.sum(other_grad, axis=tuple(i for i in range(dim_diff)))            

        output._backward = _backward
        return output


    def __neg__(self):
        return self * -np.ones_like(self.data)
    
    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        output = Tensor(self.data**other, _children=(self,), _op=f'pow')

        def _backward():
            self.grad += (other * self.data**(other-1)) * output.grad

        output._backward = _backward
        return output

    def sqrt(self, other):
        assert isinstance(other, (int, float))

        return self ** (1.0/other)
    
    def __truediv__(self, other):
        return self * other ** -1

    def exp(self):
        output = Tensor(np.exp(self.data), _children=(self,), _op=f'exp')

        def _backward():
            self.grad += output.data * output.grad

        output._backward = _backward
        return output

    def log(self):
        output = Tensor(np.log(self.data), _children=(self,), _op=f'log')

        def _backward():
            self.grad += (1.0 / self.data) * output.grad

        output._backward = _backward
        return output

    def sum(self):
        output = Tensor(np.sum(self.data), _children=(self, ), _op="sum")

        def _backward():
            self.grad += np.ones_like(self.grad) * output.grad

        output._backward = _backward
        return output

    def mean(self):
        output = Tensor(np.mean(self.data), _children=(self, ), _op="mean")

        def _backward():
            self.grad += (1/np.sum(self.data.shape)) * np.ones_like(self.grad) * output.grad

        output._backward = _backward
        return output

    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self ** -1

    # MATRIX OPERATIONS
    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.shape[-1] == other.data.shape[-2]

        output = Tensor(self.data @ other.data, _children=(self, other), _op="dot")

        def _backward():
            self.grad += output.grad @ other.data.T
            other.grad += self.data.T @ output.grad

        output._backward = _backward   
        return output

    def convolution(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(correlate2d(self.data, other.data, mode="valid"), _children=(self, other), _op="convolution")

        def _backward():
            self.grad += correlate2d(np.rot90(other.data, k=2), output.grad, mode="full")
            other.grad += correlate2d(self.data, output.grad, mode="valid")

        output._backward = _backward
        return output

    # ACTIVATION FUNCTIONS
    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    def tanh(self):
        exp_x = self.exp()
        exp_neg_x = (-self).exp()

        return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

    def relu(self):
        output = Tensor(np.maximum(self.data, 0), _children=(self, ), _op="relu")

        def _backward():
            self.grad +=  np.maximum(output.data, 0) * output.grad

        output._backward = _backward
        return output

    # BACKWARD METHOD
    def backward(self):
        """
        Backward Pass

        1. Builds computation graph
        2. Sets gradients to 1 for final output
        3. Performs backpropagation
        """

        # building computation graph
        graph = []
        visited = set()

        def build_graph(node):
            if node not in visited:
                visited.add(node)

                for child in node._children:
                    build_graph(child)
                
                graph.append(node)
            
        build_graph(self)

        # setting gradients for output
        self.grad = np.ones_like(self.data)

        # backpropagation
        for node in reversed(graph):
            node._backward()
