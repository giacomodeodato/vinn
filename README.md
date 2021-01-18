# vinn
A pytorch module to implement Bayesian neural networks with variational inference.

The standard layer implementation uses <i>Bayes by Backprop</i> \[Blundell et al., 2015\] and the local reparameterization trick \[Kingma, Salimans and Welling, 2015\] to accelerate the forward pass. The KL divergence is computed in closed form if possible, and using the Monte Carlo approximation otherwise.

## Model definition
This module is ment to be used as a drop-in replacement of ```torch.nn```. Below is an example of a standard neural network architecture and the correspondig Bayesian implementation using ```vinn```.
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 6, 5)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(6, 16, 5)
      self.fc1 = nn.Linear(256, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 256)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x
```
```python
import torch.nn as nn
import torch.nn.functional as F

import vinn

class BayesianNet(vinn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.conv1 = vinn.Conv2d(1, 6, 5)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = vinn.Conv2d(6, 16, 5)
      self.fc1 = vinn.Linear(256, 120)
      self.fc2 = vinn.Linear(120, 84)
      self.fc3 = vinn.Linear(84, 10)

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 256)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x
```

## Training

## Uncertainty estimation

## References

**\[Blundell et al., 2015\]** "Weight Uncertainty in Neural Network." International Conference on Machine Learning.

**\[Kingma, Salimans and Welling, 2015\]** “Variational dropout and the local reparameterization trick”. Advances in Neural Information Processing Systems.
