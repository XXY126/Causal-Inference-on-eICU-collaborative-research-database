"""
=============================================================
Naive, IPW, AIPW — Stima ATE causale
=============================================================
Esegui dalla root del progetto:
    python3 naive_ipw_aipw.py
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

import torch
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

N_BOOT  = 50
DATA_IN = "../data/processed/dataset_XAY.csv"


# ─────────────────────────────────────────
# 1. CARICA DATASET
# ─────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Carica dataset eICU")
print("=" * 60)

df      = pd.read_csv(DATA_IN)
EXCLUDE = ["patientunitstayid", "Y", "A"]
X_cols  = [c for c in df.columns if c not in EXCLUDE]

A = df["A"].values.astype(float)
Y = df["Y"].values.astype(float)
n = len(A)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[X_cols]).astype(np.float32)

print(f"  Pazienti:  {n}")
print(f"  Feature:   {len(X_cols)}")
print(f"  A=1: {int(A.sum())} ({A.mean()*100:.1f}%)")
print(f"  Y=1: {int(Y.sum())} ({Y.mean()*100:.1f}%)")

# ─────────────────────────────────────────
# 2. PROPENSITY SCORE
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Propensity Score (logistic regression)")
print("=" * 60)

ps_lr  = LogisticRegression(max_iter=1000, C=0.1, random_state=SEED)
ps_lr.fit(X_scaled, A)
e_x    = ps_lr.predict_proba(X_scaled)[:, 1]
e_clip = np.clip(e_x, 0.01, 0.99)
auc_lr = roc_auc_score(A, e_x)

print(f"  AUC PS: {auc_lr:.3f}")

# ─────────────────────────────────────────
# 3. NAIVE
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Naive (differenza grezza)")
print("=" * 60)

ATE_naive = float(Y[A==1].mean() - Y[A==0].mean())
print(f"  Mortalita A=1: {Y[A==1].mean()*100:.1f}%")
print(f"  Mortalita A=0: {Y[A==0].mean()*100:.1f}%")
print(f"  ATE naive:     {ATE_naive*100:+.2f}%  (non causale, confounded)")

# ─────────────────────────────────────────
# 4. IPW
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — IPW (Inverse Probability Weighting)")
print("=" * 60)

p_A1   = A.mean()
w_iptw = np.where(A==1, p_A1/e_clip, (1-p_A1)/(1-e_clip))

ATE_ipw = float(
    np.mean(w_iptw * A * Y)     / np.mean(w_iptw * A) -
    np.mean(w_iptw * (1-A) * Y) / np.mean(w_iptw * (1-A))
)

# Bootstrap CI
boot_ipw = []
for _ in range(N_BOOT):
    idx = np.random.choice(n, n, replace=True)
    Ab, Yb, eb = A[idx], Y[idx], e_clip[idx]
    w_b = np.where(Ab==1, p_A1/eb, (1-p_A1)/(1-eb))
    ate_b = (np.mean(w_b * Ab * Yb)     / np.mean(w_b * Ab) -
             np.mean(w_b * (1-Ab) * Yb) / np.mean(w_b * (1-Ab)))
    boot_ipw.append(float(ate_b))

ci_ipw  = np.percentile(boot_ipw, [2.5, 97.5])
sig_ipw = "NON sign." if ci_ipw[0] <= 0 <= ci_ipw[1] else "SIGN."

print(f"  ATE IPW:  {ATE_ipw*100:+.2f}%")
print(f"  CI 95%:   [{ci_ipw[0]*100:+.2f}%, {ci_ipw[1]*100:+.2f}%]  {sig_ipw}")

# ─────────────────────────────────────────
# 5. AIPW
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — AIPW (Augmented IPW, doubly robust)")
print("=" * 60)

om1 = LogisticRegression(max_iter=1000, C=0.1, random_state=SEED)
om0 = LogisticRegression(max_iter=1000, C=0.1, random_state=SEED)
om1.fit(X_scaled[A==1], Y[A==1])
om0.fit(X_scaled[A==0], Y[A==0])
mu1  = om1.predict_proba(X_scaled)[:, 1]
mu0  = om0.predict_proba(X_scaled)[:, 1]
psi1 = mu1 + A     * (Y - mu1) / e_clip
psi0 = mu0 + (1-A) * (Y - mu0) / (1 - e_clip)
ATE_aipw = float((psi1 - psi0).mean())

# Bootstrap CI
def aipw_boot(idx):
    Ab, Yb, Xb = A[idx], Y[idx], X_scaled[idx]
    if Ab.sum() < 5 or (1-Ab).sum() < 5:
        return np.nan
    ps_b = LogisticRegression(max_iter=300, C=0.1, random_state=SEED)
    ps_b.fit(Xb, Ab)
    eb = np.clip(ps_b.predict_proba(Xb)[:, 1], 0.01, 0.99)
    o1 = LogisticRegression(max_iter=300, C=0.1, random_state=SEED)
    o0 = LogisticRegression(max_iter=300, C=0.1, random_state=SEED)
    o1.fit(Xb[Ab==1], Yb[Ab==1])
    o0.fit(Xb[Ab==0], Yb[Ab==0])
    m1 = o1.predict_proba(Xb)[:, 1]
    m0 = o0.predict_proba(Xb)[:, 1]
    p1 = m1 + Ab     * (Yb - m1) / eb
    p0 = m0 + (1-Ab) * (Yb - m0) / (1 - eb)
    return float((p1 - p0).mean())

boot_aipw = [aipw_boot(np.random.choice(n, n, replace=True)) for _ in range(N_BOOT)]
boot_aipw = [b for b in boot_aipw if not np.isnan(b)]
ci_aipw   = np.percentile(boot_aipw, [2.5, 97.5])
sig_aipw  = "NON sign." if ci_aipw[0] <= 0 <= ci_aipw[1] else "SIGN."

print(f"  ATE AIPW: {ATE_aipw*100:+.2f}%")
print(f"  CI 95%:   [{ci_aipw[0]*100:+.2f}%, {ci_aipw[1]*100:+.2f}%]  {sig_aipw}")

# ─────────────────────────────────────────
# 6. RIEPILOGO ESTIMATORI
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("RIEPILOGO")
print("=" * 60)
print(f"""
  Coorte:  {n} paz.  |  A=1: {int(A.sum())}  |  Y=1: {int(Y.sum())}
  AUC PS (logistic): {auc_lr:.3f}

  Metodo         ATE         CI 95%                        Sig.
  ─────────────────────────────────────────────────────────────
  Naive          {ATE_naive*100:+.2f}%      —                             (non causale)
  IPW            {ATE_ipw*100:+.2f}%      [{ci_ipw[0]*100:+.2f}%, {ci_ipw[1]*100:+.2f}%]   {sig_ipw}
  AIPW           {ATE_aipw*100:+.2f}%      [{ci_aipw[0]*100:+.2f}%, {ci_aipw[1]*100:+.2f}%]   {sig_aipw}
""")
