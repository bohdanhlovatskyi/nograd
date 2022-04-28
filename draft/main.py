import numpy as np 
import matplotlib.pyplot as plt

import torch

import mnist

class Tensor:

    def __init__(self, data, requires_grad: bool = False, depends_on = None) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.depends_on = depends_on if depends_on else []
        self.grad: 'Tensor' = None
        if self.requires_grad:
            self.zero_grad()

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: 'Tensor' = None):

        if grad is None:
            grad = Tensor(np.ones(self.data.shape))
        

        self.grad.data = self.grad.data + grad.data
        
        for dep in self.depends_on:
            backward_grad = dep.backward(grad.data)
            dep.ctx.backward(Tensor(backward_grad))

    def __str__(self) -> str:
        return f'{self.data}'

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return _matmul(self, other)
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        return _matmul(self, other)

    def relu(self):
        return _relu(self)

    def dot(self, other: 'Tensor') -> 'Tensor':
        return self @ other

    def softmax(self) -> 'Tensor':
        return _softmax(self)

    def mean(self):
        return _mean(self)
    
    def cross_entropy(self, true_result):
        return _cross_entropy(self, true_result)
    
class Func:

    def __init__(self, ctx, op) -> None:
        self.ctx = ctx
        self.backward = op

def _matmul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    print(t1.shape, t2.shape)
    data = t1.data @ t2.data
    if data.shape == ():
        data = np.array([data]).reshape((1, 1))
    depends_on = []

    if t1.requires_grad:
        def matmul_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        depends_on.append(Func(t1, matmul_fn1))

    if t2.requires_grad:
        def matmul_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        depends_on.append(Func(t2, matmul_fn2))

    return Tensor(data, t1.requires_grad or t2.requires_grad, depends_on)

def _relu(t: 'Tensor') -> 'Tensor':
    data = np.maximum(0, t.data)
    depends_on = []

    if t.requires_grad:
        def relu_fn(grad: np.ndarray):
            return grad * (data >= 0)
        
        depends_on.append(Func(t, relu_fn))
    
    return Tensor(data, t.requires_grad, depends_on)


def _softmax(t: 'Tensor') -> 'Tensor':
    max_value = np.max(t.data)
    temp = t.data - max_value
    data = np.exp(temp)
    divide_by = np.sum(temp)
    data = temp / divide_by
    depends_on = []

    if t.requires_grad:
        def softmax_fn(grad: np.ndarray):
            der = temp * (divide_by - temp)
            return grad * der / (divide_by ** 2) 

        depends_on.append(Func(t, softmax_fn))
    
    return Tensor(data, t.requires_grad, depends_on)

def _mean(t: 'Tensor') -> 'Tensor':
    data = np.sum(t.data) / t.data.shape
    depends_on = []
    if t.requires_grad:
        def mean_fn(grad: np.ndarray):
            return np.ones(t.data.shape) / t.data.shape

        depends_on.append(Func(t, mean_fn))

    return Tensor(data, t.requires_grad, depends_on)

def _cross_entropy(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    data = - np.mean((t2.data * np.log(t1.data) + (1-t2.data) * np.log(1-t1.data)))
    depends_on = []
    if t1.requires_grad:
        def cross_entropy_fn(grad: np.ndarray):
            print(grad.shape, t1.shape, t2.shape)
            temp = t2.data / t1.data.T - (1 - t2.data) / (1 - t1.data).T
            return -grad * temp / t1.shape[1]
        
        depends_on.append(Func(t1, cross_entropy_fn))
    return Tensor(data, t1.requires_grad, depends_on)

class SGD:
    def __init__(self, params, lr: float = 0.01) -> None:
        self.lr = lr
        self.params = params # list of tensors

    def step(self) -> None:
        for parameter in self.params:
            parameter.data -= parameter.grad * self.lr

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros(param.grad.shape)

class Net:
  def __init__(self):
    self.l1 = Tensor(np.random.rand(784, 128), requires_grad=True)
    self.l2 = Tensor(np.random.rand(128, 10), requires_grad=True)

  def forward(self, x: 'Tensor'):
    return x.dot(self.l1).relu().dot(self.l2).softmax()


def example():
    SHOW_TORCH = True 
    if SHOW_TORCH:
        print("torch example")
        a = torch.tensor([2., 3.], requires_grad=True)
        b = torch.tensor([6., 2.], requires_grad=True)

        z = a.matmul(b)
        z.backward()

        print(a.grad)
        print(b.grad)

    print("our example")
    print()

    an = np.array([2., 3.]).reshape((2, 1))
    bn = np.array([6., 2.]).reshape((2, 1))
    a = Tensor(an.T, requires_grad=True)
    b = Tensor(bn, requires_grad=True)

    z = a @ b
    z.backward()

    print(a.grad)
    print(b.grad)

if __name__ == "__main__":
    # load dataset
    # mnist.init()
    
    Xt, yt, Xv, Yv = mnist.load()

    # create model
    model = Net()
    optim = SGD([model.l1, model.l2], lr=0.001)

    # train loop
    iteration = 0
    for x, yv in zip(Xt, yt):
        x, y = Tensor(x.reshape(1, 784)),\
               Tensor(np.eye(10)[yv, :].reshape(10, 1))
        out = model.forward(x)
        loss = out.cross_entropy(y)
        assert loss.requires_grad

        print(iteration, loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
        iteration += 1
        if iteration == 100:
            break

    iteration = 0
    for x, yv in zip(Xv, Yv):
        x, y = Tensor(x.reshape(1, 784)),\
               Tensor(np.eye(10)[yv, :].reshape(10, 1))
        out = model.forward(x)
        print("prediction: ", out)
        plt.imshow(x.data.reshape(28, 28))
        plt.show()
        iteration += 1
        if iteration == 2:
            break
