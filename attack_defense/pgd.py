import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


class AttackPGD(nn.Module):
    def __init__(self, model, loss_fn, rand_init, eps, num_steps, step_size):
        super(AttackPGD, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.rand_init = rand_init
        self.step_size = step_size
        self.epsilon = eps
        self.num_steps = num_steps

    def forward(self, inputs, targets):
        x = inputs.detach()
        if self.rand_init:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = self.loss_fn(logits, targets)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            #x = torch.clamp(x, 0, 1)

        return x