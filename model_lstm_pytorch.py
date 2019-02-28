import torch
import torchvision
import torch.nn as nn


class NNet(torch.nn.Module):
    def __init__(self, layer_shapes, activation_functions):
        super(NNet, self).__init__()
        assert len(layer_shapes) == len(activation_functions) + 1
        self.layer_shapes = layer_shapes
        self.activation_functions = activation_functions

        linear_functions = list()
        for i in range(len(self.layer_shapes)-1):
            linear_functions.append(torch.nn.Linear(
                    self.layer_shapes[i], self.layer_shapes[i+1]))

        self.linear_functions = linear_functions

    def parameters(self):
        parameters = list()
        for function in self.linear_functions:
            parameters = parameters+list(function.parameters())

        return parameters

    def forward(self, x):
        assert x.shape[1] == self.layer_shapes[0]
        y = x
        for i in range(len(self.layer_shapes)-1):
            lin = self.linear_functions[i](y)
            y = self.activation_functions[i](lin)
        return y


batch_size = 100
epochs = 500
learning_rate = 0.001

train_set = torchvision.datasets.MNIST(root=
                                       '../../data',
                                       train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

test_set = torchvision.datasets.MNIST(root=
                                      '../../data',
                                      train=False,
                                      transform=torchvision.transforms.ToTensor(),
                                      download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=False)

model = NNet([784, 16, 10], [torch.nn.Tanh(),
                                  torch.nn.Softmax(dim=1)])

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_items = list()

for t in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss_items.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()