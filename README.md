# CausalTensor
 CausalTensor is a python package for doing causal inference and policy evaluation using panel data. The package achieves 13K downloads by 2025-03.  

[![PyPI Version](https://badge.fury.io/py/causaltensor.svg)](https://pypi.org/project/causaltensor/)
[![Documentation Status](https://readthedocs.org/projects/causaltensor/badge/?version=latest)](https://causaltensor.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/causaltensor)](https://pepy.tech/project/causaltensor)

## What is CausalTensor
CausalTensor is a suite of tools for addressing questions like "What is the impact of strategy X to outcome Y" given time-series data colleting from multiple units. Answering such questions has wide range of applications from econometrics, operations research, business analytics, polictical science, to healthcare. Please visit our [complete documentation](https://causaltensor.readthedocs.io/) for more information. 

## Installing CausalTensor
CausalTensor is compatible with Python 3 or later and also depends on numpy. The simplest way to install CausalTensor and its dependencies is from PyPI with pip, Python's preferred package installer.

    $ pip install causaltensor

Note that CausalTensor is an active project and routinely publishes new releases. In order to upgrade CausalTensor to the latest version, use pip as follows.

    $ pip install -U causaltensor
    
## Using CausalTensor
We have implemented the following estimators including the traditional method Difference-in-Difference and recent proposed methods such as Synthetic Difference-in-Difference, Matrix Completion with Nuclear Norm Minimization, and De-biased Convex Panel Regression.  

| Estimator      | Reference |
| ----------- | ----------- |
| [Difference-in-Difference (DID)](https://en.wikipedia.org/wiki/Difference_in_differences) | [Implemented through two-way fixed effects regression.](http://web.mit.edu/insong/www/pdf/FEmatch-twoway.pdf)       |
| [De-biased Convex Panel Regression (DC-PR)](https://arxiv.org/abs/2106.02780) | Vivek Farias, Andrew Li, and Tianyi Peng. "Learning treatment effects in panels with general intervention patterns." Advances in Neural Information Processing Systems 34 (2021): 14001-14013. |
| [Synthetic Control (OLS SC)](http://www.jstor.org/stable/3132164)   | Abadie, Alberto, and Javier Gardeazabal. “The Economic Costs of Conflict: A Case Study of the Basque Country.” The American Economic Review 93, no. 1 (2003): 113–32. |
| [Synthetic Difference-in-Difference (SDID)](https://arxiv.org/pdf/1812.09970.pdf)   | Dmitry Arkhangelsky, Susan Athey, David A. Hirshberg, Guido W. Imbens, and Stefan Wager. "Synthetic difference-in-differences." American Economic Review 111, no. 12 (2021): 4088-4118. |
| [Matrix Completion with Nuclear Norm Minimization (MC-NNM)](https://arxiv.org/abs/1710.10251)| Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi. "Matrix completion methods for causal panel data models." Journal of the American Statistical Association 116, no. 536 (2021): 1716-1730. |

Please visit our [documentation](https://causaltensor.readthedocs.io/) for the usage instructions. Or check the following simple demo as a tutorial:

- [Panel Data Example](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/Panel_Data_Example.ipynb)
    - [Panel Data Example with old API](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/Panel%20Data%20Example.ipynb)
- [Panel Data with Multiple Treatments](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/Panel_Regression_with_Multiple_Interventions.ipynb)
- [MC-NNM with covariates and missing data](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tests/MCNNM_test.ipynb)