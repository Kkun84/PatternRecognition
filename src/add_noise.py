import torch

class AddNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return x + self.mean + torch.rand(x.shape) * self.std


class AddDots:
    def __init__(self, n, value):
        self.n = n
        self.value = value

    def __call__(self, x):
        x = x.clone()
        for _ in range(self.n):
            i = torch.randint(0, x.shape[0], [1])[0]
            j = torch.randint(0, x.shape[1], [1])[0]
            x[i, j] = self.value
        return x
