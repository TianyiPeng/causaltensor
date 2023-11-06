Usage
=====

.. _installation:

Installation
------------

CausalTensor is compatible with Python 3 or later and also depends on numpy. The simplest way to install CausalTensor and its dependencies is from PyPI with pip, Python's preferred package installer.

.. code-block:: console

   $ pip install causaltensor

Note that CausalTensor is an active project and routinely publishes new releases. In order to upgrade CausalTensor to the latest version, use pip as follows.

.. code-block:: console

   $ pip install -U causaltensor

Tutorial
----------------
For a basic panel data problem, we require two matrices as inputs 

1. :math:`O \in R^{N \times T}`: :math:`O` is the outcome matrix where :math:`O_{ij}` represents the outcome of the i-th unit at time j
2. :math:`Z \in R^{N \times T}`: :math:`Z` is the intervention matrix where :math:`Z_{ij}` indicates whether the i-th unit used the intervention or not at time j.

Given such two matrices, the problem is to ask **"what is the impact of the intervention to the outcome**"? 

Please check `Panel Data Example <https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/Panel%20Data%20Example.ipynb>`_ 
for a simple demo. 

Check :doc:`api` for various methods for solving such a problem.