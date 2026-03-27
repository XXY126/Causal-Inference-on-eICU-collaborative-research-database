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


import random
import numpy as np
import tensorflow as tf

SEED = 42
tf.set_random_seed(SEED)



N_BOOT  = 200
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

"""
=============================================================
DragonNet — Stima ATE causale con Neural Networks
=============================================================
Esegui dalla root del progetto:
    python3 dragonnet_analysis.py
=============================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Parametri
N_BOOT = 200
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_LAYERS = [100, 50]
DATA_IN = "../data/processed/dataset_XAY.csv"

print("=" * 60)
print("DRAGONNET — Stima ATE causale")
print("=" * 60)

# ─────────────────────────────────────────
# 1. CARICA DATASET
# ─────────────────────────────────────────
print("\n[1] Caricamento dataset...")

df = pd.read_csv(DATA_IN)
EXCLUDE = ["patientunitstayid", "Y", "A"]
X_cols = [c for c in df.columns if c not in EXCLUDE]

A = df["A"].values.astype(float).reshape(-1, 1)
Y = df["Y"].values.astype(float)
n = len(A)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[X_cols]).astype(np.float32)

print(f"  Pazienti: {n}")
print(f"  Feature: {len(X_cols)}")
print(f"  A=1: {int(A.sum())} ({A.mean()*100:.1f}%)")
print(f"  Y=1: {int(Y.sum())} ({Y.mean()*100:.1f}%)")

# ─────────────────────────────────────────
# 2. PROPENSITY SCORE (per confronto)
# ─────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
ps_model = LogisticRegression(max_iter=1000, C=0.1, random_state=SEED)
ps_model.fit(X_scaled, A.ravel())
e_x = ps_model.predict_proba(X_scaled)[:, 1]
e_clip = np.clip(e_x, 0.01, 0.99)
auc_ps = roc_auc_score(A.ravel(), e_x)
print(f"  AUC PS (logistic): {auc_ps:.3f}")

# ─────────────────────────────────────────
# 3. COSTRUISCI MODELLO DRAGONNET
# ─────────────────────────────────────────
print("\n[2] Costruzione DragonNet...")

def build_dragonnet(input_dim, hidden_layers=[200, 100, 50], learning_rate=0.001):
    """
    Costruisce il modello DragonNet per causal inference.
    """
    # Input
    input_x = Input(shape=(input_dim,), name='features')
    input_t = Input(shape=(1,), name='treatment')
    
    # Shared layers (dragon)
    x = input_x
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu', 
                         kernel_regularizer=regularizers.l2(0.01))(x)
    
    # Treatment-specific heads
    # Head per treatment=0
    head0 = layers.Dense(50, activation='relu')(x)
    head0 = layers.Dense(25, activation='relu')(head0)
    output0 = layers.Dense(1, activation='sigmoid', name='potential_outcome_0')(head0)
    
    # Head per treatment=1
    head1 = layers.Dense(50, activation='relu')(x)
    head1 = layers.Dense(25, activation='relu')(head1)
    output1 = layers.Dense(1, activation='sigmoid', name='potential_outcome_1')(head1)
    
    # Propensity score head
    propensity = layers.Dense(50, activation='relu')(x)
    propensity = layers.Dense(25, activation='relu')(propensity)
    propensity = layers.Dense(1, activation='sigmoid', name='propensity')(propensity)
    
    # Output combinato per training
    # Usa il treatment per selezionare l'outcome potenziale appropriato
    output = layers.Lambda(
        lambda args: args[0] * args[1] + (1 - args[0]) * args[2],
        name='combined_outcome'
    )([input_t, output1, output0])
    
    model = Model(inputs=[input_x, input_t], outputs=[output, propensity])
    
    # Loss: combinazione di outcome loss e propensity loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'combined_outcome': 'binary_crossentropy',
            'propensity': 'binary_crossentropy'
        },
        loss_weights={'combined_outcome': 1.0, 'propensity': 0.5}
    )
    
    return model

# Crea e mostra il modello
model = build_dragonnet(X_scaled.shape[1], HIDDEN_LAYERS, LEARNING_RATE)
print(model.summary())

# ─────────────────────────────────────────
# 4. TRAINING CON VALIDATION SPLIT
# ─────────────────────────────────────────
print("\n[3] Training DragonNet...")

# Split per validation
X_train, X_val, A_train, A_val, Y_train, Y_val = train_test_split(
    X_scaled, A, Y, test_size=0.2, random_state=SEED, stratify=A.ravel()
)

# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,    
    mode='min',  
    verbose=1
)

history = model.fit(
    [X_train, A_train],
    {'combined_outcome': Y_train, 'propensity': A_train.ravel()},
    validation_data=([X_val, A_val], {'combined_outcome': Y_val, 'propensity': A_val.ravel()}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

print(f"\n  Training completato. Miglior val_loss: {min(history.history['val_loss']):.4f}")

# ─────────────────────────────────────────
# 5. PREDIZIONI DEI POTENZIALI OUTCOME
# ─────────────────────────────────────────
print("\n[4] Calcolo potenziali outcome...")

# Predici per tutti i pazienti
# Per potenziali outcome, creiamo due array di treatment: tutti 0 e tutti 1
A_all_0 = np.zeros((n, 1))
A_all_1 = np.ones((n, 1))

# Predici potenziali outcome
preds_0, prop_0 = model.predict([X_scaled, A_all_0], verbose=0)
preds_1, prop_1 = model.predict([X_scaled, A_all_1], verbose=0)

# Predizioni propensity dal modello (dovrebbe essere simile al logistic)
propensity_dragon = prop_0.ravel()  # o prop_1, sono uguali

# ATE con DragonNet (differenza tra potenziali outcome)
ATE_dragon = np.mean(preds_1 - preds_0)
print(f"\n  ATE DragonNet (puntuale): {ATE_dragon*100:+.2f}%")

# ─────────────────────────────────────────
# 6. BOOTSTRAP PER INTERVALLI DI CONFIDENZA
# ─────────────────────────────────────────
print(f"\n[5] Bootstrap ({N_BOOT} iterazioni) per CI 95%...")

def bootstrap_dragonnet(X, A, Y, n_bootstrap=200):
    """
    Bootstrap per DragonNet (più lento ma robusto)
    """
    n = len(X)
    ate_boot = []
    
    for i in range(n_bootstrap):
        if (i+1) % 50 == 0:
            print(f"    Bootstrap iterazione {i+1}/{n_bootstrap}")
        
        # Resample con replacement
        idx = np.random.choice(n, n, replace=True)
        Xb, Ab, Yb = X[idx], A[idx], Y[idx]
        
        # Evita classi troppo piccole
        if np.sum(Ab) < 5 or np.sum(1-Ab) < 5:
            continue
        
        # Split per validation
        X_train_b, X_val_b, A_train_b, A_val_b, Y_train_b, Y_val_b = train_test_split(
            Xb, Ab, Yb, test_size=0.2, random_state=SEED + i
        )
        
        # Ricostruisci modello
        model_b = build_dragonnet(X.shape[1], HIDDEN_LAYERS, LEARNING_RATE)
        
        # Training con early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        model_b.fit(
            [X_train_b, A_train_b],
            {'combined_outcome': Y_train_b, 'propensity': A_train_b.ravel()},
            validation_data=([X_val_b, A_val_b], {'combined_outcome': Y_val_b, 'propensity': A_val_b.ravel()}),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predici potenziali outcome
        A_all_0_b = np.zeros((len(Xb), 1))
        A_all_1_b = np.ones((len(Xb), 1))
        preds_0_b, _ = model_b.predict([Xb, A_all_0_b], verbose=0)
        preds_1_b, _ = model_b.predict([Xb, A_all_1_b], verbose=0)
        
        ate_b = np.mean(preds_1_b - preds_0_b)
        ate_boot.append(float(ate_b))
    
    return np.array(ate_boot)

# Esegui bootstrap (attenzione: può richiedere tempo!)
# Se vuoi velocizzare, riduci N_BOOT o usa metodo alternativo
try:
    boot_ates = bootstrap_dragonnet(X_scaled, A, Y, N_BOOT)
    ci_dragon = np.percentile(boot_ates, [2.5, 97.5])
    sig_dragon = "NON sign." if ci_dragon[0] <= 0 <= ci_dragon[1] else "SIGN."
except Exception as e:
    print(f"  Bootstrap fallito: {e}")
    print("  Uso metodo alternativo per CI...")
    # Metodo alternativo: CI basato su varianza empirica delle predizioni
    ci_dragon = [ATE_dragon - 1.96 * np.std(preds_1 - preds_0) / np.sqrt(n),
                 ATE_dragon + 1.96 * np.std(preds_1 - preds_0) / np.sqrt(n)]
    sig_dragon = "NON sign." if ci_dragon[0] <= 0 <= ci_dragon[1] else "SIGN."

# ─────────────────────────────────────────
# 7. PROPENSITY SCORE ACCURACY
# ─────────────────────────────────────────
print("\n[6] Confronto propensity score...")

auc_dragon = roc_auc_score(A.ravel(), propensity_dragon)
print(f"  AUC PS (DragonNet): {auc_dragon:.3f}")
print(f"  AUC PS (Logistic):  {auc_ps:.3f}")

# ─────────────────────────────────────────
# 8. RIEPILOGO FINALE
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("RIEPILOGO COMPLETO")
print("=" * 60)

print(f"""
  DATASET:
    Pazienti:     {n}
    Feature:      {len(X_cols)}
    Trattamento:  A=1 → {int(A.sum())} ({A.mean()*100:.1f}%)
    Outcome:      Y=1 → {int(Y.sum())} ({Y.mean()*100:.1f}%)

  PROPENSITY SCORE:
    AUC Logistic:  {auc_ps:.3f}
    AUC DragonNet: {auc_dragon:.3f}

  ATE (Average Treatment Effect):
    ─────────────────────────────────────────────────────────────
    Metodo         ATE         CI 95%                        Sig.
    ─────────────────────────────────────────────────────────────
    Naive          {float(Y[A.ravel()==1].mean() - Y[A.ravel()==0].mean())*100:+.2f}%      —                             (non causale)
    IPW            {ATE_ipw*100:+.2f}%      —                             (calcola prima)
    AIPW           {ATE_aipw*100:+.2f}%      —                             (calcola prima)
    DragonNet      {ATE_dragon*100:+.2f}%      [{ci_dragon[0]*100:+.2f}%, {ci_dragon[1]*100:+.2f}%]   {sig_dragon}
    ─────────────────────────────────────────────────────────────
""")

print("\nDragonNet completato!")