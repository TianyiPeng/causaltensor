API
#######


Difference-in-Difference
*****************************

Difference-in-difference (DID) is a popular method to address panel data problems. 
We use a two-way fixed effects regression to estimate the average treatment effect on the treated entries (ATT). 
In particular, we solve the following regression by linear regression

.. math::
   \min \sum_{ij} (O_{ij} - a_i - b_j - \tau Z_{ij})^2

where :math:`a_{i}, b_{j}` are unknown fixed effects and :math:`\tau` is the ATT. 

To use DID, simply call

.. code-block:: python

   M, tau = DID(O, Z)

with two return parameters `M` and `tau`. Here :math:`M_{ij}=a_{i}+b_{j}`` is the estimated ideal outcome; and `tau` is the estimated ATT. 

De-biased Convex Panel Regression
**********************************************************

The second method is De-biaeed Convex Panel Regression (DC-PR) proposed by [FariasLiPeng22]_. 
Note that an issue of the DID model is that, $a_i+b_j$ are often too simple to describe the complex reality of the outcome. As a fix, 
a low-rank factor model to generalize :math:`a_i+b_j` has been advocated. 

The idea in [FariasLiPeng22]_ is to firstly solve the following low-rank regression problem by replacing :math:`a_i+b_j` in DID by a low-rank matrix :math:`M`

.. math::
   \hat{M}, \hat{\tau} = \arg\min \sum_{ij} (O_{ij}-M_{ij}-\tau Z_{ij})^2 + \lambda \|M\|_{*}

where $\|M\|_{*}$ is the nuclear norm to penalize the low-rankness of the matrix and $\lambda$ is a tunning parameter. The second step of [2] is to mitigate the bias induced by the regularization parameter (it also reflects the interaction between $\hat{M}$ and $Z$):

.. math::
   \tau^{d} = \hat{\tau} - \lambda \frac{<Z, \hat{U}\hat{V}^{\top}>}{\|P_{\hat{T}^{\perp}}(Z)\|_{F}^2}.

To use DC-PR, call

.. code-block:: python
   
   M, tau, M_raw, tau_raw = DC_PR_auto_rank(O, Z)

where `M`, `tau` are the de-biased versions and `M_raw` and `tau_raw` are the optimizers for the first step. This function helps to find the proper rank for :math:`M` (but not very stable, and may be updated later). You can also use

.. code-block:: python

   M, tau, M_raw, tau_raw = DC_PR_with_suggested_rank(O, Z, suggest_r = r)

if you have an estimation of the rank of :math:`M` by yourself. 

In addition, we also provide a formula to estimate the empirical standard deviation of DC-PR when noises are (heterogenoues) independent sub-Gaussian. See [FariasLiPeng22]_ for further details. 

.. code-block:: python

   std = std_debiased_convex(O, Z, M_raw, tau_raw)


Synthetic Difference-in-Difference
**********************************************************


The second method is called synthetic difference-in-difference (SDID) proposed by [Arkhangelsky21]_. Readers can read [Arkhangelsky21]_ for more details. To use SDID, simply call

.. code-block:: python

   tau = SDID(O, Z)

where `tau` is the estimation of SDID. 

Matrix Completion with Nuclear Norm Minimization
**********************************************************



The third method is based on matrix completion method proposed by [Athey21]_. The idea is to solve the following matrix completion problem, only using the outcome data without intervention (i.e., :math:`Z_{ij}=0`)

.. math::
   \hat{M}, \hat{a}, \hat{b} = \arg\min \sum_{ij, Z_{ij}=0} (O_{ij}-M_{ij} - a_i - b_j)^2 + \lambda \|M\|_{*}

where :math:`\|M\|_{*}` is the nuclear norm that penalizes the low-rankness of the matrix (here :math:`a_{i}` and :math:`b_{j}` are used to improve the empirical performance, as suggested by [Athey21]_). 

After :math:`\hat{M}, \hat{a}, \hat{b}` are obtained, the ATT :math:`\hat{\tau}` can be estimated simply by 

.. math::

   \hat{\tau} = \frac{\sum_{ij, Z_{ij}=1} (O_{ij} - \hat{M}_{ij} - \hat{a}_i - \hat{b}_{j})}{\sum_{ij, Z_{ij}=1} 1}.


To use this method (referred to as matrix completion with nuclear norm minimization, or MC-NNM), when you have an estimation of the rank of the matrix :math:`M` (e.g., by checking the spectrum), call

.. code-block:: python

   M, a, b, tau = MC_NNM_with_suggested_rank(O, 1-Z, suggest_r = r)

where `M`, `a`, `b` are the optimizers and `tau` is the estimated ATT. 

We also provide a function to help you find the right parameter $\lambda$ or rank by cross-validation:

.. code-block:: python

   M, a, b, tau = MC_NNM_with_cross_validation(O, 1-Z)
