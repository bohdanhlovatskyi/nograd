import numpy as np 


import torch

class Tensor:

    def __init__(self, data, requires_grad: bool = False, depends_on = None) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.depends_on = depends_on if depends_on else []
        self.grad: 'Tensor' = None
        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: 'Tensor' = None):
        if grad is None:
            grad = Tensor(np.ones(self.data.shape))

        self.grad.data = self.grad.data + grad.data

        for dep in self.depends_on:
            backward_grad = dep.backward(grad.data)
            dep.ctx.backward(backward_grad)

    def __str__(self) -> str:
        return f'{self.data}'

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return _matmul(self, other)

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
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            print(grad.shape, t2.data.T.shape, t2.data.shape)
            return grad @ t2.data.T

        depends_on.append(Func(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        depends_on.append(Func(t2, grad_fn2))

    return Tensor(data, t1.requires_grad or t2.requires_grad, depends_on)


SHOW_TORCH = True 

if __name__ == "__main__":

    if SHOW_TORCH:
        a = torch.tensor([2., 3.], requires_grad=True)
        b = torch.tensor([6., 2.], requires_grad=True)

        z = a.matmul(b)
        z.backward()

        print(a.grad)
        print(b.grad)

    print()
    print()

    an = np.array([2., 3.]).reshape((2, 1))
    bn = np.array([6., 2.]).reshape((2, 1))
    a = Tensor(an.T, requires_grad=True)
    b = Tensor(bn, requires_grad=True)

    z = a @ b
    z.backward()

    print(a.grad)
    print(b.grad)
