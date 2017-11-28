<div align="center">
  <img height="150px" src="docs/source/_static/img/probtorch-logo.png"></a>
</div>

Probabilistic Torch is library for deep generative models that extends [PyTorch](http://pytorch.org). It is similar to [Edward](http://edwardlib.org) and [Pyro](https://github.com/uber/pyro), sharing many design characteristics with the latter. 

Probababilistic Torch is very light weight: It currently implements the minimal data structures needed to train variational autoencoders:

- A library of [distribution primitives](https://github.com/probtorch/probtorch/tree/master/probtorch/distributions) for sampling and evaluation of the log probability mass and density functions

- A [Trace](https://github.com/probtorch/probtorch/blob/master/probtorch/stochastic.py#L119) data structure, which is both used to instantiate and store random variables.

- Objective functions for [Monte Carlo](https://github.com/probtorch/probtorch/blob/master/probtorch/objectives/montecarlo.py) and [Importance-weighted](https://github.com/probtorch/probtorch/blob/master/probtorch/objectives/importance.py) approximation of a lower bound on the log marginal likelihood.


# Installation

1. Install PyTorch [[instructions](https://github.com/pytorch/pytorch)]
2. Clone this repository
```
git clone git@github.com:probtorch/probtorch.git
```

3. Refer to the `examples/` subdirectory for [Jupyter](http://jupyter.org) notebooks that illustrate usage. 

4. To read documentation, please do the following 
```
cd docs
pip install -r requirements.txt
make html
open build/html/index.html
```


# Contributors 

(in order of joining)

- Jan-Willem van de Meent
- Siddharth Narayanaswamy 
- Brooks Paige
- Alban Desmaison
- Alican Bozkurt
- Amirsina Torfi


# Citing

Please cite the NIPS 2017 paper "Learning Disentangled Representations with Semi-Supervised Deep Generative Models"

```latex
@inproceedings{narayanaswamy2017learning,
    title = {Learning Disentangled Representations with Semi-Supervised Deep Generative Models},
    author = {Narayanaswamy, Siddharth and Paige, T. Brooks and van de Meent, Jan-Willem and Desmaison, Alban and Goodman, Noah and Kohli, Pushmeet and Wood, Frank and Torr, Philip},
    booktitle = {Advances in Neural Information Processing Systems 30},
    editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
    pages = {5927--5937},
    year = {2017},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7174-learning-disentangled-representations-with-semi-supervised-deep-generative-models.pdf}
}
```



