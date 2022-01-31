<div align="center">
  <img height="150px" src="docs/source/_static/img/probtorch-logo.png"></a>
</div>

Probabilistic Torch is library for deep generative models that extends [PyTorch](http://pytorch.org). It is similar in spirit and design goals to [Edward](http://edwardlib.org) and [Pyro](https://github.com/uber/pyro), sharing many design characteristics with the latter.

The design of Probabilistic Torch is intended to be as PyTorch-like as possible. Probabilistic Torch models are written just like you would write any PyTorch model, but make use of three additional constructs:

1. A library of *reparameterized* [distributions](https://pytorch.org/docs/stable/distributions.html) that implement methods for sampling and evaluation of the log probability mass and density functions (now available in PyTorch)

2. A [Trace](https://github.com/probtorch/probtorch/blob/master/probtorch/stochastic.py#L119) data structure, which is both used to instantiate and store random variables.

3. Objective functions that approximate the lower bound on the log marginal likelihood using [Monte Carlo](https://github.com/probtorch/probtorch/blob/master/probtorch/objectives/montecarlo.py) and [Importance-weighted](https://github.com/probtorch/probtorch/blob/master/probtorch/objectives/importance.py) estimators.

This repository accompanies the NIPS 2017 paper:

```latex
@inproceedings{siddharth2017learning,
    title = {Learning Disentangled Representations with Semi-Supervised Deep Generative Models},
    author = {Siddharth, N. and Paige, Brooks and van de Meent, Jan-Willem and Desmaison, Alban and Goodman, Noah D. and Kohli, Pushmeet and Wood,
Frank and Torr, Philip},
    booktitle = {Advances in Neural Information Processing Systems 30},
    editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
    pages = {5927--5937},
    year = {2017},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7174-learning-disentangled-representations-with-semi-supervised-deep-generative-models.pdf}
}
```


# Contributors

(in order of joining)

- Jan-Willem van de Meent
- Siddharth Narayanaswamy
- Brooks Paige
- Alban Desmaison
- Alican Bozkurt
- Amirsina Torfi
- Babak Esmaeili
- Eli Sennesh


# Installation

1. Install PyTorch [[instructions](https://github.com/pytorch/pytorch)]

2. Install this repository from source
```
pip install git+git://github.com/probtorch/probtorch
```

3. Refer to the `examples/` subdirectory for [Jupyter](http://jupyter.org) notebooks that illustrate usage.

4. To build and read the API documentation, please do the following
```
git clone git://github.com/probtorch/probtorch
cd probtorch/docs
pip install -r requirements.txt
make html
open build/html/index.html
```


# Mini-Tutorial: Semi-supervised MNIST

Models in Probabilistic Torch define variational autoencoders. Both the encoder and the decoder model can be implemented as standard PyTorch models that subclass `nn.Module`.

In the `__init__` method we initialize network layers, just as we would in a PyTorch model. In the `forward` method, we additionally initialize a `Trace` variable, which is a write-once dictionary-like object. The `Trace` data structure implements methods for instantiating named random variables, whose values and log probabilities are stored under the specifed key.

Here is an implementation for the encoder of a standard semi-supervised VAE, as introduced by Kingma and colleagues [1]

```python
import torch
import torch.nn as nn
import probtorch

class Encoder(nn.Module):
    def __init__(self, num_pixels=784, num_hidden=50, num_digits=10, num_style=2):
        super(self.__class__, self).__init__()
        self.h = nn.Sequential(
                    nn.Linear(num_pixels, num_hidden),
                    nn.ReLU())
        self.y_log_weights = nn.Linear(num_hidden, num_digits)
        self.z_mean = nn.Linear(num_hidden + num_digits, num_style)
        self.z_log_std = nn.Linear(num_hidden + num_digits, num_style)

    def forward(self, x, y_values=None, num_samples=10):
        q = probtorch.Trace()
        x = x.expand(num_samples, *x.size())
        if y_values is not None:
            y_values = y_values.expand(num_samples, *y_values.size())
        h = self.h(x)
        y = q.concrete(logits=self.y_log_weights(h), temperature=0.66,
                       value=y_values, name='y')
        h2 = torch.cat([y, h], -1)
        z = q.normal(loc=self.z_mean(h2),
                     scale=torch.exp(self.z_log_std(h2)),
                     name='z')
        return q
```

In the code above, the method `q.concrete` samples or observes from a Concrete/Gumbel-Softmax relaxation of the discrete distribution, depending on whether supervision values `y_values` are provided. The method `q.normal` samples from a univariate normal.

The resulting trace `q` now contains two entries `q['y']` and `q['z']`, which are instances of a `RandomVariable` class, which stores both the value and the log probability associated with the variable. The stored values are now used to condition execution of the decoder model:
```python

def binary_cross_entropy(x_mean, x, EPS=1e-9):
    return - (torch.log(x_mean + EPS) * x +
              torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)

class Decoder(nn.Module):
    def __init__(self, num_pixels=784, num_hidden=50, num_digits=10, num_style=2):
        super(self.__class__, self).__init__()
        self.num_digits = num_digits
        self.h = nn.Sequential(
                   nn.Linear(num_style + num_digits, num_hidden),
                   nn.ReLU())
        self.x_mean = nn.Sequential(
                        nn.Linear(num_hidden, num_pixels),
                        nn.Sigmoid())

    def forward(self, x, q=None):
        if q is None:
            q = probtorch.Trace()
        p = probtorch.Trace()
        y = p.concrete(logits=torch.zeros(x.size(0), self.num_digits),
                       temperature=0.66,
                       value=q['y'], name='y')
        z = p.normal(loc=0.0, scale=1.0, value=q['z'], name='z')
        h = self.h(torch.cat([y, z], -1))
        p.loss(binary_cross_entropy, self.x_mean(h), x, name='x')
        return p
```
The model above can be used both for conditioned forward execution, but also for generation. The reason for this is that `q[k]` returns `None` for variable names `k` that have not been instantiated.

To train the model components above, probabilistic Torch provides objectives that compute an estimate of a lower bound on the log marginal likelihood, which can now be maximized with standard PyTorch optimizers
```python
from probtorch.objectives.montecarlo import elbo
from random import rand
# initialize model and optimizer
enc = Encoder()
dec = Decoder()
optimizer =  torch.optim.Adam(list(enc.parameters())
                              + list(dec.parameters()))
# define subset of batches that will be supervised
supervise = [rand() < 0.01 for _ in data]
# train model for 10 epochs
for epoch in range(10):
    for b, (x, y) in data:
        x = Variable(x)
        if supervise[b]:
            y = Variable(y)
            q = enc(x, y)
        else:
            q = enc(x)
        p = dec(x, q)
        loss = -elbo(q, p, sample_dim=0, batch_dim=1)
        loss.backward()
        optimizer.step()
```

For a more details, see the Jupyter notebooks in the `examples/` subdirectory.

# References

[1] Kingma, Diederik P, Danilo J Rezende, Shakir Mohamed, and Max Welling. 2014. “Semi-Supervised Learning with Deep Generative Models.” http://arxiv.org/abs/1406.5298.
