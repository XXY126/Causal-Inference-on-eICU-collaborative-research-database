"""
=============================================================
Naive, IPW, AIPW, DragonNet — Stima ATE causale
=============================================================
Esegui dalla root del progetto:
    python3 dragonnet_eicu.py
=============================================================
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
import random
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

N_BOOT    = 30
N_BOOT_DN = 30
DATA_IN   = "../data/processed/dataset_XAY.csv"

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
# 6. DRAGONNET
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — DragonNet (TF2/Keras, targeted regularization)")
print("=" * 60)


class EpsilonLayer(layers.Layer):
    def build(self, input_shape):
        self.epsilon = self.add_weight(
            name="epsilon",
            shape=(1, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
    def call(self, inputs):
        return self.epsilon * tf.ones_like(inputs)


def build_dragonnet(input_dim, neurons=100, neurons_out=50):
    """
    Architettura ridotta (100/50 invece di 200/100) + Dropout 0.2
    per gestire l'overfitting su dataset piccolo (n=1980).
    """
    X = layers.Input(shape=(input_dim,), name="X")
    t = layers.Input(shape=(1,),         name="t")

    # Representation network
    z = layers.Dense(neurons, activation="elu")(X)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(neurons, activation="elu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(neurons, activation="elu")(z)
    z = layers.Dropout(0.2)(z)

    # Outcome heads
    y0 = layers.Dense(neurons_out, activation="elu")(z)
    y0 = layers.Dropout(0.1)(y0)
    y0 = layers.Dense(neurons_out, activation="elu")(y0)
    y0 = layers.Dense(1, activation="sigmoid", name="y0")(y0)

    y1 = layers.Dense(neurons_out, activation="elu")(z)
    y1 = layers.Dropout(0.1)(y1)
    y1 = layers.Dense(neurons_out, activation="elu")(y1)
    y1 = layers.Dense(1, activation="sigmoid", name="y1")(y1)

    # Propensity head
    g = layers.Dense(1, activation="sigmoid", name="g")(z)

    # Epsilon
    eps = EpsilonLayer(name="epsilon")(g)

    out = layers.Concatenate(name="output")([y0, y1, g, eps])
    return Model(inputs=[X, t], outputs=out)


def dragonnet_loss(ratio=1.0):
    def loss(y_true, y_pred):
        y   = y_true[:, 0]
        t   = y_true[:, 1]
        y0  = y_pred[:, 0]
        y1  = y_pred[:, 1]
        g   = y_pred[:, 2]
        eps = y_pred[:, 3]

        y_hat   = t * y1 + (1 - t) * y0
        loss_y  = tf.reduce_mean((y - y_hat) ** 2)
        loss_g  = -tf.reduce_mean(
            t * tf.math.log(g + 1e-8) + (1 - t) * tf.math.log(1 - g + 1e-8)
        )
        h       = t / (g + 1e-8) - (1 - t) / (1 - g + 1e-8)
        y_pert  = y_hat + eps * h
        loss_tr = tf.reduce_mean((y - y_pert) ** 2)

        return loss_y + ratio * loss_g + loss_tr
    return loss


def fit_dragonnet(X, t, y,
                  ratio=1.0,
                  adam_epochs=50,  adam_lr=1e-3,
                  sgd_epochs=100,  sgd_lr=1e-5,  sgd_momentum=0.9,
                  batch_size=64,   verbose=0):

    model  = build_dragonnet(X.shape[1])
    t_col  = t.reshape(-1, 1).astype(np.float32)
    y_true = np.stack([y, t], axis=1).astype(np.float32)

    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

    # Fase 1 — Adam
    model.compile(
        optimizer=optimizers.Adam(learning_rate=adam_lr),
        loss=dragonnet_loss(ratio=ratio),
    )
    model.fit([X, t_col], y_true,
              epochs=adam_epochs, batch_size=batch_size,
              verbose=verbose, callbacks=[es])

    # Fase 2 — SGD
    model.compile(
        optimizer=optimizers.SGD(learning_rate=sgd_lr, momentum=sgd_momentum),
        loss=dragonnet_loss(ratio=ratio),
    )
    model.fit([X, t_col], y_true,
              epochs=sgd_epochs, batch_size=batch_size,
              verbose=verbose, callbacks=[es])

    return model


def predict_ate_dragonnet(model, X):
    t_dummy = np.zeros((len(X), 1), dtype=np.float32)
    preds   = model.predict([X, t_dummy], verbose=0)
    mu0 = preds[:, 0]
    mu1 = preds[:, 1]
    return mu0, mu1, float((mu1 - mu0).mean())


# ── Training principale ───────────────────────────────────────
print("  Training DragonNet...")
dn_model = fit_dragonnet(
    X_scaled, A, Y,
    ratio        = 1.0,
    adam_epochs  = 50,
    adam_lr      = 1e-3,
    sgd_epochs   = 100,
    sgd_lr       = 1e-5,
    sgd_momentum = 0.9,
    batch_size   = 64,
    verbose      = 0,
)

mu0_dn, mu1_dn, ATE_dn = predict_ate_dragonnet(dn_model, X_scaled)
print(f"  ATE DragonNet (point): {ATE_dn*100:+.2f}%")

# ── Bootstrap CI ─────────────────────────────────────────────
# Il modello viene riallenato sul campione bootstrap,
# ma l'ATE viene valutato sempre sul dataset ORIGINALE.
# Questo cattura la variabilità dello stimatore, non del campione.
print(f"  Bootstrap CI ({N_BOOT_DN} repliche)...")

boot_dn = []
for i in range(N_BOOT_DN):
    idx = np.random.choice(n, n, replace=True)
    Xb, Ab, Yb = X_scaled[idx], A[idx], Y[idx]

    dn_b = fit_dragonnet(
        Xb, Ab, Yb,
        ratio        = 1.0,
        adam_epochs  = 50,
        adam_lr      = 1e-3,
        sgd_epochs   = 100,
        sgd_lr       = 1e-5,
        sgd_momentum = 0.9,
        batch_size   = 64,
        verbose      = 0,
    )

    # Predici sul dataset ORIGINALE, non sul bootstrap
    _, _, ate_b = predict_ate_dragonnet(dn_b, X_scaled)
    boot_dn.append(ate_b)

    del dn_b
    K.clear_session()

    if (i + 1) % 5 == 0:
        print(f"    {i+1}/{N_BOOT_DN} | ATE replica: {ate_b*100:+.2f}%")

ci_dn  = np.percentile(boot_dn, [2.5, 97.5])
sig_dn = "NON sign." if ci_dn[0] <= 0 <= ci_dn[1] else "SIGN."

ATE_dn_median = float(np.median(boot_dn))
ATE_dn_mean   = float(np.mean(boot_dn))

print(f"  ATE point (training):  {ATE_dn*100:+.2f}%")
print(f"  ATE mediana bootstrap: {ATE_dn_median*100:+.2f}%")
print(f"  ATE media bootstrap:   {ATE_dn_mean*100:+.2f}%")

print(f"  CI 95%:        [{ci_dn[0]*100:+.2f}%, {ci_dn[1]*100:+.2f}%]  {sig_dn}")

# ─────────────────────────────────────────
# 7. RIEPILOGO FINALE
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
  DragonNet      {ATE_dn_mean*100:+.2f}%      [{ci_dn[0]*100:+.2f}%, {ci_dn[1]*100:+.2f}%]   {sig_dn}
""")

# ─────────────────────────────────────────
# 7. DR LEARNER (XGBoost)
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — DR Learner (XGBoost)")
print("=" * 60)

from xgboost import XGBRegressor, XGBClassifier

# ── 1. Propensity model (più flessibile)
ps_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=SEED,
    eval_metric="logloss"
)

ps_model.fit(X_scaled, A)
e_x = ps_model.predict_proba(X_scaled)[:, 1]
e_x = np.clip(e_x, 0.01, 0.99)

# ── 2. Outcome models (μ0, μ1)
mu0_model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=SEED
)

mu1_model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=SEED
)

mu0_model.fit(X_scaled[A == 0], Y[A == 0])
mu1_model.fit(X_scaled[A == 1], Y[A == 1])

mu0 = mu0_model.predict(X_scaled)
mu1 = mu1_model.predict(X_scaled)

# ── 3. Pseudo-outcome DR
tau_dr = (
    mu1 - mu0
    + A * (Y - mu1) / e_x
    - (1 - A) * (Y - mu0) / (1 - e_x)
)

ATE_dr = float(np.mean(tau_dr))

print(f"  ATE DR: {ATE_dr*100:+.2f}%")

boot_dr = []

for _ in range(N_BOOT):
    idx = np.random.choice(n, n, replace=True)
    Xb, Ab, Yb = X_scaled[idx], A[idx], Y[idx]

    # Propensity
    ps_b = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05)
    ps_b.fit(Xb, Ab)
    eb = np.clip(ps_b.predict_proba(Xb)[:, 1], 0.01, 0.99)

    # Outcome
    m0 = XGBRegressor(n_estimators=100, max_depth=3)
    m1 = XGBRegressor(n_estimators=100, max_depth=3)

    m0.fit(Xb[Ab == 0], Yb[Ab == 0])
    m1.fit(Xb[Ab == 1], Yb[Ab == 1])

    mu0b = m0.predict(Xb)
    mu1b = m1.predict(Xb)

    tau_b = (
        mu1b - mu0b
        + Ab * (Yb - mu1b) / eb
        - (1 - Ab) * (Yb - mu0b) / (1 - eb)
    )

    boot_dr.append(np.mean(tau_b))

ci_dr  = np.percentile(boot_dr, [2.5, 97.5])
sig_dr = "NON sign." if ci_dr[0] <= 0 <= ci_dr[1] else "SIGN."

print(f"  CI 95%: [{ci_dr[0]*100:+.2f}%, {ci_dr[1]*100:+.2f}%]  {sig_dr}")