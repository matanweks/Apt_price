import numpy as np
import torch
import matplotlib.pyplot as plt
import robust_loss_pytorch.general
from tqdm import tqdm


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 100)
        self.fc2 = torch.nn.Linear(100, 200)
        self.fc3 = torch.nn.Linear(200, 1)
        # self.m = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        # x = self.m(x)
        x = torch.nn.functional.relu(self.fc2(x))
        # x = self.m(x)
        x = self.fc3(x)
        return x[:, None][:, 0]


class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x[:, None])[:, 0]


def plot_regression(regression, color='r', label='w', figure=0, linspace=[0, 0.2], x=0, y=0, errorbar=True, Percent=1, down=False):
    # A helper function for plotting a regression module.
    x_plot = np.linspace(linspace[0], linspace[1], 1000)
    x_plot = torch.Tensor(x_plot)
    x_plot = x_plot.view(-1, 1)
    plt.figure(figure)
    y_plot = regression(x_plot).detach().numpy()
    if down:
        down = -1
    else:
        down = 1
    if errorbar:
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        y_i = regression(x)
        y_tag = y_i.cpu().detach().numpy()[:, 0]
        loss = y_i - y
        loss = loss.cpu().detach().numpy()[:, 0]
        x_fill = np.sort(x.cpu().detach().numpy()[:, 0] * Percent)
        y_fill_down = down*np.sort(down*(y_tag-np.abs(loss)))
        y_fill_up = down*np.sort(down*(y_tag+np.abs(loss)))
        plt.fill_between(x_fill, y_fill_down, y_fill_up, label=label, alpha=0.5, color=color, interpolate=True)
        plt.plot(x_plot * Percent, y_plot, color=color)
    else:
        plt.plot(x_plot * Percent, y_plot, color=color)


def fit_adaptive(x, y, epochs=500):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    x = x.view(-1, 1)
    y = y.view(-1, 1)

    regression = Net()
    adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        num_dims=1, float_dtype=np.float32, device='cpu')
    params = list(regression.parameters()) + list(adaptive.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)

    for epoch in tqdm(range(epochs)):
        y_i = regression(x)
        loss = torch.mean(adaptive.lossfun(y_i - y))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if np.mod(epoch, 100) == 0:
        #     print('{:<4}: loss={:03f}  alpha={:03f}  scale={:03f}'.format(
        #         epoch, loss.data, adaptive.alpha()[0, 0].data, adaptive.scale()[0, 0].data))

    return regression


def fit(x, y, epochs=2000):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    x = x.view(-1, 1)
    y = y.view(-1, 1)

    regression = Net()
    params = regression.parameters()
    optimizer = torch.optim.Adam(params, lr=0.01)

    for epoch in tqdm(range(epochs)):
        x = x.view(-1, 1)
        y_i = regression(x)

        # Hijacking the general loss to compute MSE.
        loss = torch.mean(robust_loss_pytorch.general.lossfun(
            y_i - y, alpha=torch.Tensor([2.]), scale=torch.Tensor([0.1])))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if np.mod(epoch, 100) == 0:
        #     print('{:<4}: loss={:03f}'.format(epoch, loss.data))
    return regression
