from cmath import nan
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
    temp = t2.data - t1.data
    data = np.mean(temp * temp)
    depends_on = []

    if t1.requires_grad:
        def MSE_fn(grad: np.ndarray):
            return -2*grad*temp/t1.shape[1]

        depends_on.append(Func(t1, MSE_fn))
    
    return Tensor(data, t1.requires_grad, depends_on)


def _sigmoid(t: 'Tensor') -> 'Tensor':
    data = np.zeros(t.data.shape)
    for i in range(t.data.shape[1]):
        val = t.data[0, i]
        if val < 0:
            data[0, i] = np.exp(val) / (1 + np.exp(val))
        else:
            data[0, i] = 1 / (1 + np.exp(-val))
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
    epsilon = 0.12
    self.l1 = Tensor(np.random.rand(784, 25)*epsilon*2 - epsilon, requires_grad=True)
    self.l2 = Tensor(np.random.rand(25, 10)*epsilon*2 - epsilon, requires_grad=True)

  def forward(self, x: 'Tensor'):
    return x.dot(self.l1).sigmoid().dot(self.l2).sigmoid()

class XORNet :
    def __init__(self) -> None:
        self.l1 = Tensor(np.random.rand(2, 2), requires_grad=True)
        self.l2 = Tensor(np.random.rand(2, 1), requires_grad=True)

    def forward(self, x: 'Tensor'):

        w = x @ self.l1
        w = w.softmax()
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

def test_relu():
    SHOW_TORCH = True 
    if SHOW_TORCH:
        print("torch example")
        a = torch.tensor([2., 3.], requires_grad=True)
        m = torch.nn.ReLU()
        b = m(a)
        z = a.matmul(b)
        z.backward()

        print(a.grad)
        print(b.grad)

    print("our example")
    print()

    an = np.array([2., 3.]).reshape((2, 1))
    bn = np.array([6., 2.]).reshape((2, 1))
    a = Tensor(an.T, requires_grad=True)
    b = a.relu()
    z = a @ b.T
    z.backward()

    print(a.grad)
    print(b.grad)

def test_mse():
    loss = torch.nn.MSELoss()
    a = torch.tensor([1.3643746, 2.37467237], requires_grad=True)
    b = torch.tensor([7.398497899, 4.43876743829])
    out = loss(a, b)
    out.backward()
    print(out)
    print(a.grad)

    an = np.array([1.3643746, 2.37467237]).reshape((2, 1))
    bn = np.array([7.398497899, 4.43876743829]).reshape((2, 1))
    a = Tensor(an, requires_grad=True)
    b = Tensor(bn)
    s = a.mse(b)
    print(s)
    s.backward()
    print(a.grad)

def test_matmul_and_mse():
    print("torch example")
    loss = torch.nn.MSELoss()
    a = torch.tensor([2., 3.], requires_grad=True)
    b = torch.tensor([6., 2.], requires_grad=True)

    z = a.matmul(b)
    true = torch.tensor([19.])
    out = loss(z, true)
    out.backward()
    print(out)
    print(a.grad)
    print(b.grad)

    print("our example")
    print()

    an = np.array([2., 3.]).reshape((2, 1))
    bn = np.array([6., 2.]).reshape((2, 1))
    a = Tensor(an.T, requires_grad=True)
    b = Tensor(bn, requires_grad=True)

    z = a @ b
    true = Tensor(np.array([19.]))
    out = z.mse(true)
    out.backward()
    print(out)
    print(a.grad)
    print(b.grad)

def test_xor():
    print("torch example")
    loss = torch.nn.MSELoss()
    first_layer = np.matrix('2. 3.; 4. 5')
    second_layer = np.matrix('6.; 7.')
    x = np.matrix('0., 1.')
    gt = np.matrix('1.')
    input = torch.tensor(x)
    first_layer_torch = torch.tensor(first_layer, requires_grad=True)
    second_layer_torch = torch.tensor(second_layer, requires_grad=True)

    w = input.matmul(first_layer_torch)
    w = w.matmul(second_layer_torch)
    true = torch.tensor(gt)
    out = loss(w, true)
    out.backward()
    print(out)
    print(first_layer_torch.grad)
    print(second_layer_torch.grad)

    print("our example")
    print()

    input_our = Tensor(x)
    first_layer_our = Tensor(first_layer, requires_grad=True)
    second_layer_our = Tensor(second_layer, requires_grad=True) 
    w = input_our @ first_layer_our
    w = w @ second_layer_our
    true = Tensor(gt)
    out = w.mse(true)
    out.backward()
    print(out)
    print(first_layer_our.grad)
    print(second_layer_our.grad)


# if __name__ == "__main__":
#     # test_mse()
#     # test_matmul_and_mse()
#     # test_xor()
#     # exit(0)
#     # test() 
#     # example()
#     # exit()
#     # test_relu()
#     # load dataset
#     # mnist.init()


#     # Xt, yt, Xv, Yv = mnist.load()

#     # create models
#     model = XORNet()
#     optim = SGD([model.l1, model.l2], lr=0.1)
#     Xs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
#     ys = np.array([[0.,], [1.,], [1.,], [0.,]])

#     # # train loop
#     epochs = 100000
#     # # for x, yv in zip(Xt, yt):
#     for i in range(epochs):

#         # X = Tensor(Xs[i % 4].reshape((1, 2)))
#         # y = Tensor(ys[i%4].reshape((1,  1)))

#         X = Tensor(Xs)
#         y = Tensor(ys)
#         out = model.forward(X)
#         loss = out.mse(y)
#         if np.isnan(loss.data):
#             exit(0)

#         assert loss.requires_grad
#         # print(i, loss)

#         optim.zero_grad()
#         loss.backward()
#         print(f"iteration #{i}: loss: {loss}\n l1_grad: {model.l1.grad}, l2_grad: {model.l2.grad}, l1_value: {model.l1}, l2_value: {model.l2}")
#         optim.step()

#     # iteration = 0
#     # for x, yv in zip(Xv, Yv):
#     #     x, y = Tensor(x.reshape(1, 784)),\
#     #            Tensor(np.eye(10)[yv, :].reshape(10, 1))
#     #     out = model.forward(x)
#     #     print("prediction: ", out, np.argmax(out))
#     #     plt.imshow(x.data.reshape(28, 28))
#     #     plt.show()
#     #     iteration += 1
#     #     if iteration == 2:
#     #         break

#     out = model.forward(Tensor(Xs[0]))
#     print(Xs[0], out)

#     out = model.forward(Tensor(Xs[1]))
#     print(Xs[1], out)
    
#     out = model.forward(Tensor(Xs[2]))
#     print(Xs[2], out)

#     out = model.forward(Tensor(Xs[3]))
#     print(Xs[3], out)


if __name__ == "__main__":
    # load dataset
    # mnist.init()

    Xt, yt, Xv, Yv = mnist.load()
    # create model
    model = MNISTNet()
    optim = SGD([model.l1, model.l2], lr=30)

    # train loop
    for iteration in range(50):
        x, y = Tensor(Xt),\
               Tensor(np.eye(10)[yt, :])
        print(y)
        out = model.forward(x)
        loss = out.mse(y)
        assert loss.requires_grad

        print(iteration, loss, np.linalg.norm(model.l1.data), np.linalg.norm(model.l1.grad.data))
        optim.zero_grad()
        loss.backward()
        optim.step()

    iteration = 0
    for x, yv in zip(Xv, Yv):
        x, y = Tensor(x.reshape(1, 784)),\
               Tensor(np.eye(10)[yv, :].reshape(10, 1))
        out = model.forward(x)
        print("prediction: ", out)
        print('true result: ', y)
        plt.imshow(x.data.reshape(28, 28))
        plt.show()
        iteration += 1
        if iteration == 10:
            break