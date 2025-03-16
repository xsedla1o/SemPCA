import functools

import torch


def lr_lambda(epoch, decay, decay_step):
    return decay ** (epoch // decay_step)


class Optimizer:
    def __init__(self, parameter, lr=2e-3):
        self.optim = torch.optim.Adam(parameter, lr=lr, betas=(0.9, 0.9), eps=1e-12)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=functools.partial(lr_lambda, decay=0.75, decay_step=1000),
        )

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_last_lr()


class SGDOptimizer:
    def __init__(self, parameter):
        self.optim = torch.optim.SGD(parameter, lr=2e-3)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=functools.partial(lr_lambda, decay=0.75, decay_step=1000),
        )

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_last_lr()
