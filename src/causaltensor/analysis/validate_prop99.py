"""Prop99 vs synthdid, MCPanel, pysyncon. python -m causaltensor.analysis.validate_prop99"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pysyncon import Dataprep, RobustSynth

from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.utils.common import get_fit_result_from_method
from causaltensor.utils.panel import default_raw_datasets_path

OUT = Path(__file__).resolve().parent / "results" / "validation"
METHODS = ("OLS_DID", "SC", "SDID", "MC_NNM_CV", "RSC")
RANK, LAMBDA = 2, 0.1
RSCRIPT = next(Path(r"C:\Program Files\R").glob("R-*/bin/Rscript.exe"))

R = r"""
out <- commandArgs(TRUE)[1]
Y <- as.matrix(read.csv(file.path(out, "prop99_Y.csv"), check.names=FALSE)[,-1])
Z <- as.matrix(read.csv(file.path(out, "prop99_Z.csv"), check.names=FALSE)[,-1])
storage.mode(Y) <- "double"; storage.mode(Z) <- "double"
N0 <- nrow(Y)-1L; T0 <- min(which(Z[nrow(Z),]>0))-1L
fit <- MCPanel::mcnnm_cv(Y, 1-Z, num_lam_L=20)
M <- fit$L + outer(fit$u, fit$v, "+")
write.csv(data.frame(
  method=c("OLS_DID","SC","SDID","MC_NNM_CV"),
  ref_tau=c(as.numeric(synthdid::did_estimate(Y,N0,T0)),
            as.numeric(synthdid::sc_estimate(Y,N0,T0)),
            as.numeric(synthdid::synthdid_estimate(Y,N0,T0)),
            sum(Z*(Y-M))/sum(Z)),
  reference=c("synthdid::did_estimate","synthdid::sc_estimate",
              "synthdid::synthdid_estimate","MCPanel::mcnnm_cv")
), file.path(out,"r_references.csv"), row.names=FALSE)
"""

OUT.mkdir(parents=True, exist_ok=True)
Y_df, Z_df, _ = load_dataset("smoking", datasets_path=default_raw_datasets_path())
years = np.arange(1970, 2001)
names = sorted(s for s in Y_df.index if s != "California") + ["California"]
Y_df, Z_df = Y_df.reindex(index=names, columns=years), Z_df.reindex(index=names, columns=years)
O, Z = Y_df.to_numpy(float), Z_df.to_numpy(float)
t0 = int(np.where(Z[-1] > 0)[0][0])

ct = pd.DataFrame([
    {"method": m, "ct_tau": float(get_fit_result_from_method(m, O, Z)[0].tau)}
    for m in METHODS
])

env = os.environ.copy()
env["R_LIBS_USER"] = str(Path.home() / "AppData/Local/R/win-library/4.5")
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    Y_df.rename_axis("unit").reset_index().to_csv(tmp / "prop99_Y.csv", index=False)
    Z_df.rename_axis("unit").reset_index().to_csv(tmp / "prop99_Z.csv", index=False)
    subprocess.run([str(RSCRIPT), "-", str(tmp).replace("\\", "/")], input=R, text=True, check=True, env=env)
    r_refs = pd.read_csv(tmp / "r_references.csv")

long = Y_df.rename_axis("unit").reset_index().melt(id_vars="unit", var_name="year", value_name="y")
long["year"] = long["year"].astype(int)
pre, post = years[:t0].tolist(), years[t0:].tolist()
rsc = RobustSynth()
rsc.fit(
    Dataprep(
        foo=long, predictors=["y"], predictors_op="mean", dependent="y",
        unit_variable="unit", time_variable="year",
        treatment_identifier="California", controls_identifier=list(names[:-1]),
        time_predictors_prior=pre, time_optimize_ssr=pre,
    ),
    lambda_=LAMBDA, sv_count=RANK,
)
refs = pd.concat([
    r_refs,
    pd.DataFrame([{"method": "RSC", "ref_tau": float(rsc.att(time_period=post)["att"]),
                  "reference": f"pysyncon r={RANK} lambda={LAMBDA}"}]),
], ignore_index=True)

cmp = ct.merge(refs, on="method")
cmp["abs_diff"] = (cmp["ct_tau"] - cmp["ref_tau"]).abs()
cmp.to_csv(OUT / "prop99_reference_comparison.csv", index=False)
print(cmp.to_string(index=False))
