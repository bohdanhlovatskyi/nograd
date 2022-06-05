from cmath import nan
import numpy as np 
import matplotlib.pyplot as plt

import copy
from pyparsing import str_type 
import torch


from tqdm import tqdm

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

    def toposort(self):
        def _toposort(node, visited, nodes):
            visited.add(node)
            if node.depends_on:
                [_toposort(i.ctx, visited, nodes) for i in node.depends_on if i.ctx not in visited]
                nodes.append(node)
            return nodes
        return _toposort(self, set(), [])

    def backward(self, grad: 'Tensor' = None):

        if grad is None or not grad.data.any():
            grad = Tensor(np.ones(self.data.shape))
        
        self.grad.data += grad.data

        toposort = list(reversed(self.toposort()))
        for tensor in toposort:
            for dep in tensor.depends_on:
                backward_grad = dep.backward(tensor.grad.data)
                dep.ctx.grad.data += backward_grad

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
    
    def cross_entropy(self, true_result) -> 'Tensor':
        return _cross_entropy(self, true_result)
    
    def conv2D(self, filt: 'Tensor', bias: 'Tensor', padding="same", stride=(1, 1)) -> 'Tensor':
        return _conv2D(self, filt, bias, padding, stride)
    
    def pool(self, kernel_shape, stride=(1, 1), mode="max") -> 'Tensor':
        return _pool(self, kernel_shape, stride, mode)

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
            jacobian = np.diag(data[0]) - data.T @ data
            res = grad @ jacobian
            return res

        depends_on.append(Func(t, softmax_fn))
    
    return Tensor(data, t.requires_grad, depends_on)

def _cross_entropy(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    data = -t2.data @ np.log(t1.data.T)
    depends_on = []

    if t1.requires_grad:
        def cross_entropy_fn(grad: np.ndarray):
            return (t1.data - t2.data) * grad
        
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

def _conv2D(tensor: 'Tensor', filt: 'Tensor', bias: 'Tensor', padding="same", stride=(1, 1)) -> 'Tensor':
    """
    tensor - input tensor of shape (m, h_prev, w_prev, c_prev)
    filt - filter for the convolution of shape (kh, kw, c_prev, c_new)
    bias - bias of shape (1, 1, 1, c_new)
    padding - string same or none
    stride - tuple of (sh, sw)
    """
    m, h_prev, w_prev, c_prev = tensor.shape
    kh, kw, _, c_new = filt.shape
    sh, sw = stride

    if padding == "same":
        ph = np.ceil(((sh*h_prev) - sh + kh - h_prev) / 2)
        ph = int(ph)
        pw = np.ceil(((sw*w_prev) - sw + kw - w_prev) / 2)
        pw = int(pw)
    else:
        ph = 0
        pw = 0
    
    data = tensor.data
    data = np.pad(data, [(0, 0), (ph, ph), (pw, pw), (0, 0)], mode='constant', constant_values=0)

    out_h = int(((h_prev - kh + (2*ph)) / (stride[0])) + 1)
    out_w = int(((w_prev - kw + (2*pw)) / (stride[1])) + 1)
    output_conv = np.zeros((m, out_h, out_w, c_new))
    m_range = np.arange(0, m)
	
    for i in range(out_h):
        for j in range(out_w):
            for f in range(c_new):
                output_conv[m_range, i, j, f] = np.sum(np.multiply(
                    data[m_range,
                    i*(stride[0]):kh+(i*(stride[0])),
                    j*(stride[1]):kw+(j*(stride[1]))],
                    filt.data[:, :, :, f]), axis=(1, 2, 3)) + bias.data[0, 0, 0, f]
    
    depends_on = []

    # grad is of shape (m, h_out, w_out, c_new)
    if bias.requires_grad:
        def bias_grad_fn(grad: np.ndarray):
            db = np.sum(grad, axis = (0, 1, 2), keepdims=True)
            return db

        depends_on.append(Func(bias, bias_grad_fn))

    if filt.requires_grad:
        def filt_grad_fn(grad: np.ndarray):
            dfilt = np.zeros_like(filt.data)
            for i in range(m):
                for h in range(out_h):
                    for w in range(out_w):
                        for f in range(c_new):
                            dfilt[:, :, :, f] += data[i,h*(stride[0]):(h*(stride[0]))+kh, w*(stride[1]):(w*(stride[1]))+kw, :] * grad[i, h, w, f]
            return dfilt
        
        depends_on.append(Func(filt, filt_grad_fn))
    
    if tensor.requires_grad:
        def tensor_grad_fn(grad: np.ndarray):
            dx = np.zeros(data.shape)
            for i in range(m):
                for h in range(out_h):
                    for w in range(out_w):
                        for f in range(c_new):
                            dx[i, h*(stride[0]):(h*(stride[0]))+kh, w*(stride[1]):(w*(stride[1]))+kw, :] += grad[i, h, w, f] * filt.data[:, :, :, f]
            
            if padding == "same":
                dx = dx[:, ph:-ph, pw:-pw, :]
            
            return dx
        
        depends_on.append(Func(tensor, tensor_grad_fn))

    return Tensor(output_conv, tensor.requires_grad or bias.requires_grad or filt.requires_grad, depends_on)

def _pool(tensor, kernel_shape, stride=(1, 1), mode="max") -> 'Tensor':
    m, h_prev, w_prev, c_prev = tensor.shape
    kh, kw = kernel_shape
    data = tensor.data

    out_h = int(((h_prev - kh) / (stride[0])) + 1)
    out_w = int(((w_prev - kw) / (stride[1])) + 1)
    output_pool = np.zeros((m, out_h, out_w, c_prev))
    m_range = np.arange(0, m)

    for i in range(out_h):
        for j in range(out_w):
            if mode == "max":
                output_pool[m_range, i, j] = np.max(data[m_range, i*(stride[0]):kh+(i*(stride[0])),j*(stride[1]):kw+(j*(stride[1]))], axis=(1,2))
            if mode == "avg":
                output_pool[m_range, i, j] = np.mean(data[m_range, i*(stride[0]):kh+(i*(stride[0])),j*(stride[1]):kw+(j*(stride[1]))], axis=(1,2))

    depends_on = []

    return Tensor(output_pool, tensor.requires_grad, depends_on)

class SGD:
    def __init__(self, params, lr: float = 0.01, alpha: float = 0.001 ) -> None:
        self.lr = lr
        self.alpha = alpha
        self.params = params # list of tensors

    def step(self) -> None:
        for parameter in self.params:
            parameter.data -= (parameter.grad + self.alpha * parameter.data) * self.lr

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros(param.grad.shape)

class MNISTNet:
  def __init__(self):
    w = torch.empty(785, 25)
    torch.nn.init.xavier_uniform_(w, gain=torch.nn.init.calculate_gain('sigmoid'))
    self.l1 = Tensor(np.array(w.data), requires_grad=True)

    w = torch.empty(25, 10)
    torch.nn.init.xavier_uniform_(w, gain=torch.nn.init.calculate_gain('sigmoid'))
    self.l2 = Tensor(np.array(w.data), requires_grad=True)

  def forward(self, x: 'Tensor'):
    x.data /= 255.0
    return x.dot(self.l1).relu().dot(self.l2).softmax()

def validation(model, Xv, Yv):
    iteration = 0
    tr = 0
    for x, y in tqdm(zip(Xv, Yv)):
        x, y = Tensor(x.reshape(1, 785)),\
               Tensor(np.eye(10)[y, :].reshape(10, 1))
        out = model.forward(x)

        if np.argmax(out.data) == np.argmax(y.data):
            tr += 1
        iteration += 1

    print(f"Accuracy: {tr/iteration}")

def Mnist(load: bool = False):
    # load dataset
    if load:
        mnist.init()

    # Accuracy: 0.8727 for 10 epochs

    Xt, yt, Xv, Yv = mnist.load()
    Xt = np.append(np.ones((60000, 1)), Xt, axis=1)
    Xv = np.append(np.ones((10000, 1)), Xv, axis=1)
    # create model
    model = MNISTNet()
    optim = SGD([model.l1, model.l2], lr=0.01)

    # train loop
    iteration = 0
    for epoch in range(15):
        loss = None
        for x, y in tqdm(zip(Xt, yt)):
            iteration += 1
            x, y = Tensor(x.reshape(1, 785)),\
                Tensor(np.eye(10)[y, :].reshape(1, 10))

            out = model.forward(x)
            loss = out.cross_entropy(y)
            assert loss.requires_grad

            optim.zero_grad()
            loss.backward()
            optim.step()

        validation(model, Xv, Yv)
        print(loss)

def test_forward_conv():
    truth = np.random.rand(1, 1, 3, 3)
    m = torch.nn.Conv2d(1, 1, 3)
    filt = m.weight.data.numpy()
    bias = m.bias.data.numpy()
    l = np.random.rand(5, 5)
    to_torch = l.reshape(1, 1, 5, 5)
    tens = torch.Tensor(to_torch)
    tens.requires_grad = True
    out = m(tens)
    torch_truth = torch.Tensor(truth)
    loss = torch.nn.MSELoss()
    torch_loss = loss(out, torch_truth)
    print(torch_loss)
    torch_loss.backward()
    # print(out)
    # print(m.bias.grad)
    # print(tens.grad)
    print(m.weight.grad)

    to_our = l.reshape(1, 5, 5, 1)
    tens = Tensor(to_our, requires_grad=True)
    filt = filt.reshape(3, 3, 1, 1)
    filt = Tensor(filt, requires_grad=True)
    bias = bias.reshape(1, 1, 1, 1)
    bias = Tensor(bias, requires_grad=True)
    out = tens.conv2D(filt, bias, padding="valid")
    truth = truth.reshape(1, 3, 3, 1)
    truth = Tensor(truth)
    our_loss = out.mse(truth)
    print(our_loss)
    our_loss.backward()
    # print(out.data)
    # print(bias.grad.data)
    # print(tens.grad.data)
    print(filt.grad.data)

def test_pool():
    truth = np.random.rand(1, 2, 2, 2)
    m = torch.nn.MaxPool2d(2, stride=2)
    l = np.random.rand(1, 2, 4, 4)
    tens = torch.Tensor(l)
    tens.requires_grad = True
    out = m(tens)
    torch_truth = torch.Tensor(truth)
    loss = torch.nn.MSELoss()
    torch_loss = loss(out, torch_truth)
    print(torch_loss)
    torch_loss.backward()
    print(out)

    to_our = l.transpose(0, 2, 3, 1)
    tens = Tensor(to_our, requires_grad=True)
    out = tens.pool((2, 2), (2, 2))
    truth = truth.transpose(0, 2, 3, 1)
    truth = Tensor(truth)
    our_loss = out.mse(truth)
    print(our_loss)
    our_loss.backward()
    print(out)

if __name__ == "__main__":
    # Mnist()
    # test_forward_conv()
    test_pool()
