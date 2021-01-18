# vinn
A pytorch module to implement Bayesian neural networks with variational inference.

The standard layer implementation uses <i>Bayes by Backprop</i> \[Blundell et al., 2015\] and the local reparameterization trick \[Kingma, Salimans and Welling, 2015\] to accelerate the forward pass. The KL divergence is computed in closed form if possible, and using the Monte Carlo approximation otherwise.

## Usage

### Prior and Posterior distributions
Prior and posterior distribution can be set as arguments in the layer declaration. The module supports distributions from `torch.distributions` that have a loc and scale parameters. The default implementation uses a Normal posterior distribution as well as a Normal prior initialized with `loc=0` and `scale=1`.
```python
from torch.distributions import Normal

linear_layer = Linear(4, 5, posterior=Normal, prior=Normal(0, 1))
```

### Model definition
This module is ment to be used as a drop-in replacement of ```torch.nn```. Below is an example of a Bayesian neural network implementation.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import vinn

class Net(vinn.Module): # class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = vinn.Conv2d(1, 6, 5) # self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = vinn.Conv2d(6, 16, 5) # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = vinn.Linear(256, 120) # self.fc1 = nn.Linear(256, 120)
        self.fc2 = vinn.Linear(120, 84) # self.fc2 = nn.Linear(120, 84)
        self.fc3 = vinn.Linear(84, 10) # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### Training
Bayesian neural networks implemented using variational inference can be trained by optimizing the Evidence Lower BOund (ELBO):
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=ELBO%20%3D%20%5Cmathbb%7BE%7D_%7Bq(%5Cmathbf%7Bw%7D%3B%20%5Ctheta)%7D%5C%7B%5Cmbox%7Blog%20%7Dp(%5Cmathcal%7BD%7D%7C%5Cmathbf%7Bw%7D)%5C%7D%20-%20KL%5C%7Bq(%5Cmathbf%7Bw%7D%3B%20%5Ctheta)%7C%7Cp(%5Cmathbf%7Bw%7D)%5C%7D" width=400>
</p>

This is equivalent to minimizing the negative log likelihood (cross entropy loss) plus the KL divergence between the posterior approximation and the prior distribution. Such loss can be scaled to very large datasets using stochastic variational inference \[Hoffman et al., 2013\] as follows:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=ELBO%5C%20loss%20%3D%20-%20%5Cmathbb%7BE%7D_%7Bq(%5Cmathbf%7Bw%7D%3B%20%5Ctheta)%7D%5C%7B%5Ctext%7Blog%20%7Dp(%5Cmathcal%7BD%7D_i%7C%5Cmathbf%7Bw%7D)%5C%7D%20%2B%20%5Cbeta%20KL%5C%7Bq(%5Cmathbf%7Bw%7D%3B%20%5Ctheta)%7C%7Cp(%5Cmathbf%7Bw%7D)%5C%7D%2C%5Cqquad%20%5Cbeta%20%3D%201%2FM" width=650>
</p>

where *i* is the mini-batch index and *M* is the number of mini-batches. This can be implemented as follows:
```python
net = Net()
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
beta = torch.tensor(1.0/len(data_loader))

for epoch in range(n_epochs):
    for i, (inputs, targets) in enumerate(data_loader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = net(inputs)

        # build elbo loss
        nll = criterion(outputs, targets)
        kl = net.kl
        elbo = nll + beta*kl

        # backprop pass and optimize
        elbo.backward()
        optimizer.step()
```

By default, `vinn` layers have a `kl` attribute as well as models extending `vinn.Module`.
> :warning: The negative log likelihood function (i.e. `CrossEntropyLoss`) must have `reduction="sum"` in order to follow the correct ELBO loss implementation. Otherwise, `beta` needs to be scaled accordingly.

### Uncertainty estimation

## References

**\[Blundell et al., 2015\]** "Weight Uncertainty in Neural Network". International Conference on Machine Learning.

**\[Hoffman et al., 2013\]** "Stochastic variational inference". The Journal of Machine Learning Research.

**\[Kingma, Salimans and Welling, 2015\]** "Variational dropout and the local reparameterization trick". Advances in Neural Information Processing Systems.
