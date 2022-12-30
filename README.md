# CausalTensor
 CausalTensor is a python package for doing causal inference and policy evaluation using panel data. 


## What is CausalTensor
CausalTensor is a suite of tools for addressing questions like "What is the impact of strategy X to outcome Y" given time-series data colleting from multiple units. Answering such questions has wide range of applications from econometrics, operations research, business analytics, polictical science, to healthcare. Please visit our complete documentation for more information. 

## Installing CausalTensor
CausalTensor is compatible with Python 3 or later and also depends on numpy. The simplest way to install CausalTensor and its dependencies is from PyPI with pip, Python's preferred package installer.

    $ pip install causaltensor

Note that CausalTensor is an active project and routinely publishes new releases. In order to upgrade CausalTensor to the latest version, use pip as follows.

    $ pip install -U causaltensor
    
## Using CausalTensor
We have implemented the following estimators including the traditional method Difference-in-Difference and recent proposed methods such as Synthetic Difference-in-Difference, Matrix Completion with Nuclear Norm Minimization, and De-biased Convex Panel Regression.  

| Estimator      | Reference |
| ----------- | ----------- |
| Difference-in-Difference (DID)      | [Implemented through two-way fixed effects regression](http://web.mit.edu/insong/www/pdf/FEmatch-twoway.pdf)       |
| Synthetic Difference-in-Difference (SDID)   | https://arxiv.org/pdf/1812.09970.pdf |
| Matrix Completion with Nuclear Norm Minimization (MC-NNM)| https://arxiv.org/abs/1710.10251 |
| De-biased Convex Panel Regression (DC-PR) | https://arxiv.org/abs/2106.02780 |
