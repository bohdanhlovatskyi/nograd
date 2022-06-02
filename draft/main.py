import numpy as np 
import matplotlib.pyplot as plt

import copy 
import torch

import mnist

DISPLAY_GRAPH = False

class Tensor:

    def __init__(self, data, requires_grad: bool = False, depends_on = None) -> None:
        self.data = data.astype(np.float64)
        self.requires_grad = requires_grad
        self.depends_on = depends_on if depends_on else []
        self.grad: 'Tensor' = None
        if self.requires_grad:
            self.zero_grad()

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self): 
        return Tensor(self.data.T, self.requires_grad, self.depends_on)

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: 'Tensor' = None):

        if grad is None or not grad.data.any():
            grad = Tensor(np.ones(self.data.shape))
        
        self.grad.data += grad.data
        
        for dep in self.depends_on:
            backward_grad = dep.backward(grad.data)
            if DISPLAY_GRAPH:
                print(dep.backward.__name__)
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

    def mse(self, other) -> 'Tensor':
        return _MSE(self, other)

    def sigmoid(self) -> 'Tensor':
        return _sigmoid(self)

    def mean(self):
        return _mean(self)
    
    def cross_entropy(self, true_result):
        return _cross_entropy(self, true_result)
    
class Func:

    def __init__(self, ctx, op) -> None:
        self.ctx = ctx
        self.backward = op

def _matmul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
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
            tmp = grad * (data >= 0)
            return tmp
        
        depends_on.append(Func(t, relu_fn))
    
    return Tensor(data, t.requires_grad, depends_on)


def _softmax(t: 'Tensor') -> 'Tensor':
    max_value = np.max(t.data)
    temp = t.data - max_value
    data = np.exp(temp)
    divide_by = np.sum(data)
    data = data / divide_by
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
    delta = 0.0001
    data = - np.mean((t2.data.T * np.log(t1.data+delta) + (1-t2.data).T * np.log(1-t1.data+delta)))
    depends_on = []
    if t1.requires_grad:
        def cross_entropy_fn(grad: np.ndarray):
            temp = t2.data.T / (t1.data+delta) - (1 - t2.data).T / (1 - t1.data + delta)
            return -data*grad*temp
        
        depends_on.append(Func(t1, cross_entropy_fn))
    return Tensor(data, t1.requires_grad, depends_on)

def _MSE(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    temp = t2.data.T - t1.data
    data = np.mean(temp * temp)
    depends_on = []

    if t1.requires_grad:
        def MSE_fn(grad: np.ndarray):
            return -2*grad*temp/t1.shape[1]

        depends_on.append(Func(t1, MSE_fn))
    
    return Tensor(data, t1.requires_grad, depends_on)


def _sigmoid(t: 'Tensor') -> 'Tensor':
    sig = lambda x: 1 / (1 + np.exp(-x))
    data = sig(t.data)
    depends_on = []

    if t.requires_grad:
        def sigmoid_fn(grad: np.ndarray):
            return grad * data * (1 - data)
        
        depends_on.append(Func(t, sigmoid_fn))
    
    return Tensor(data, t.requires_grad, depends_on)

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

class MNISTNet:
  def __init__(self):
    self.l1 = Tensor(np.random.rand(784, 128), requires_grad=True)
    self.l2 = Tensor(np.random.rand(128, 10), requires_grad=True)

  def forward(self, x: 'Tensor'):
    return x.dot(self.l1).relu().dot(self.l2).sigmoid()

class XORNet :
    def __init__(self) -> None:
        self.l1 = Tensor(np.random.rand(2, 2), requires_grad=True)
        self.l2 = Tensor(np.random.rand(2, 1), requires_grad=True)

    def forward(self, x: 'Tensor'):
        # print(x.shape, self.l1.shape, self.l2.shape)

        w = x @ self.l1
        w = w.sigmoid()
        w = w @ self.l2
        w = w.sigmoid()

        return w

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


def test():
    x, y = np.array([1, 0, 0, 0, 1.]).reshape((1, 5)), np.array([1., 0.]).reshape((1, 2))
    X, Y = Tensor(x), Tensor(y)
    x, y = torch.Tensor(x), torch.Tensor(y)

    lll = np.random.rand(5, 4)
    L1 = Tensor(lll, requires_grad=True)
    l1 = torch.Tensor(copy.deepcopy(lll))

    lllll = np.random.rand(4, 2)
    L2 = Tensor(lllll, requires_grad=True)
    l2 = torch.Tensor(copy.deepcopy(lllll))
    
    x = x @ l1
    print("[first mul]: torch: ", x)
    X = Tensor(X.data @ L1.data)
    print("our: ", X.data)

    x = torch.relu(x)
    print("[first relu]: torch", x)
    X = X.relu()
    print("our: ", X.data)

    x = x @ l2
    print("[second mul]:  torch: ", x)
    X = Tensor(X.data @ L2.data)
    print("our: ", X.data)    

    ''' softmax is ok
    x = torch.softmax(x, dim=1)
    print("torch: ", x)
    X = X.softmax()
    print("out: ", X.data)
    '''

    x = torch.sigmoid(x)
    print("[sig]: torch: ", x)
    X = X.sigmoid()
    print("out: ", X.data)

    loss = torch.nn.MSELoss()
    print("[loss]: torch: ", loss(x, y))
    print("our: ", X.mse(Y))

    l, ol = loss(x, y), X.mse(Y)
    l.backward(torch.ones_like(l))
    ol.backward()

    print()

if __name__ == "__main__":
    # test() 

    # exit()
    # load dataset
    # mnist.init()


    # Xt, yt, Xv, Yv = mnist.load()

    # create models
    model = XORNet()
    optim = SGD([model.l1, model.l2], lr=0.0001)
    Xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    ys = np.array([[0,], [1,], [1,], [0,]])

    # train loop
    iteration = 0
    epochs = 10
    # for x, yv in zip(Xt, yt):
    for i in range(epochs):
 
        # x, y = Tensor(x.reshape(1, 784)),\
        #        Tensor(np.eye(10)[yv, :].reshape(10, 1))            
        # x.data /= 255.
        X = Tensor(Xs[i % 4].reshape((1, 2)))
        y = Tensor(ys[i%4].reshape((1,  1)))

        out = model.forward(X)
        loss = out.mse(y)

        assert loss.requires_grad
        print(iteration, loss)

        if iteration == 0:
            optim.zero_grad()
        loss.backward()
        optim.step()

        iteration += 1            
        if iteration == 20:
            break

    # iteration = 0
    # for x, yv in zip(Xv, Yv):
    #     x, y = Tensor(x.reshape(1, 784)),\
    #            Tensor(np.eye(10)[yv, :].reshape(10, 1))
    #     out = model.forward(x)
    #     print("prediction: ", out, np.argmax(out))
    #     plt.imshow(x.data.reshape(28, 28))
    #     plt.show()
    #     iteration += 1
    #     if iteration == 2:
    #         break

    out = model.forward(Tensor(Xs[0]))
    print(out)

    out = model.forward(Tensor(Xs[1]))
    print(out)
    
    out = model.forward(Tensor(Xs[2]))
    print(out)

    out = model.forward(Tensor(Xs[3]))
    print(out)
