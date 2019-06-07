import torch
import torch.nn as nn
import torch.nn.functional as F
batch_size, n_classes = 1, 3
x = torch.randn(batch_size, n_classes)
print(x.shape)
print(x)
target = torch.randint(n_classes, size=(batch_size,), dtype=torch.long)
print(target)
def softmax(x):
    return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)
def nl(input, target):
    return -input[range(target.shape[0]), target].log().mean()

pred = softmax(x)
print(pred)
loss=nl(pred, target)
print(loss)
pred = F.log_softmax(x, dim=-1)
loss = F.nll_loss(pred, target)
print(loss)
print(F.cross_entropy(x, target))