Synthetic DGP Study
===================

**Notebook:** ``tutorials/guides/02_synthetic_dgp.ipynb``

This tutorial uses ``causaltensor.synthetic.generate`` to create panels with
fully controlled parameters, allowing isolation of each factor that affects
estimation accuracy.

Topics covered
--------------

1. **DGP tour** -- visualise baseline ``M = UV^T``, treatment mask ``Z``, and
   observed outcomes ``O = M + tau*Z + noise``.
2. **Single-run benchmark** -- compare all estimators on one synthetic dataset.
3. **Convergence study** -- sweep N and T independently to confirm that
   relative error decreases as the panel grows.
4. **Rank misspecification** -- vary the true rank of ``M`` while keeping
   estimator settings fixed; observe how methods degrade.
5. **Noise sensitivity** -- sweep the noise scale (sigma) to understand
   each method's signal-to-noise requirements.

Key function
------------

.. code-block:: python

   from causaltensor.synthetic import generate

   O, Z, tau_true = generate(
       N=50, T=80,
       rank=3,
       noise_scale=1.0,
       treatment_pattern="Block",
       treatment_level=0.3,
       seed=0,
   )
