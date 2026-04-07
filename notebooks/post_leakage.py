import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

UP = "../data/raw/"

# Carica trattamento e vitali/lab con timestamp
treatment = pd.read_csv(f"{UP}treatment.csv",
                        usecols=["patientunitstayid","treatmentoffset","treatmentstring"])
vital     = pd.read_csv(f"{UP}vitalPeriodic.csv",
                        usecols=["patientunitstayid","observationoffset"])
lab_df    = pd.read_csv(f"{UP}lab.csv",
                        usecols=["patientunitstayid","labresultoffset","labname"])

# Identifica i trattati e il loro offset di inizio antibiotico
def is_therapeutic(s):
    s = str(s).lower()
    if "prophylactic" in s: return False
    return any(k in s for k in ["therapeutic antibacterials",
        "pulmonary|medications|antibacterials",
        "cardiovascular|other therapies|antibacterials"])

abx = treatment[treatment["treatmentstring"].apply(is_therapeutic)].copy()
abx_first = (abx[abx["treatmentoffset"] >= 0]
             .sort_values("treatmentoffset")
             .groupby("patientunitstayid")["treatmentoffset"]
             .first()
             .reset_index()
             .rename(columns={"treatmentoffset": "abx_offset"}))

# ─────────────────────────────────────────
# CHECK VITALI: quanti sono post-trattamento?
# ─────────────────────────────────────────
vital_merged = vital.merge(abx_first, on="patientunitstayid", how="inner")
vital_merged["is_post_treatment"] = (
    vital_merged["observationoffset"] > vital_merged["abx_offset"]
)

# Solo vitali nella finestra [0, 120 min]
vital_win = vital_merged[
    (vital_merged["observationoffset"] >= 0) &
    (vital_merged["observationoffset"] <= 120)
]

n_total_vital = len(vital_win)
n_post_vital  = vital_win["is_post_treatment"].sum()
pct_post_vital = n_post_vital / n_total_vital * 100

print("=" * 55)
print("CHECK LEAKAGE — Vitali basali [0, 120 min]")
print("=" * 55)
print(f"  Osservazioni totali nella finestra: {n_total_vital:,}")
print(f"  Post-trattamento:  {n_post_vital:,} ({pct_post_vital:.1f}%)")
print(f"  Pre-trattamento:   {n_total_vital - n_post_vital:,} ({100-pct_post_vital:.1f}%)")

# ─────────────────────────────────────────
# CHECK LAB: quanti sono post-trattamento?
# ─────────────────────────────────────────
lab_merged = lab_df.merge(abx_first, on="patientunitstayid", how="inner")
lab_merged["is_post_treatment"] = (
    lab_merged["labresultoffset"] > lab_merged["abx_offset"]
)

lab_win = lab_merged[
    (lab_merged["labresultoffset"] >= 0) &
    (lab_merged["labresultoffset"] <= 360)
]

n_total_lab = len(lab_win)
n_post_lab  = lab_win["is_post_treatment"].sum()
pct_post_lab = n_post_lab / n_total_lab * 100

print("\n" + "=" * 55)
print("CHECK LEAKAGE — Lab basali [0, 360 min]")
print("=" * 55)
print(f"  Osservazioni totali nella finestra: {n_total_lab:,}")
print(f"  Post-trattamento:  {n_post_lab:,} ({pct_post_lab:.1f}%)")
print(f"  Pre-trattamento:   {n_total_lab - n_post_lab:,} ({100-pct_post_lab:.1f}%)")

# ─────────────────────────────────────────
# DISTRIBUZIONE OFFSET ABX vs FINESTRE
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("DISTRIBUZIONE OFFSET ANTIBIOTICO (trattati)")
print("=" * 55)
for threshold in [30, 60, 120, 180, 360, 720, 1440]:
    n = (abx_first["abx_offset"] <= threshold).sum()
    pct = n / len(abx_first) * 100
    label = f"{threshold//60}h" if threshold >= 60 else f"{threshold}min"
    print(f"  Abx entro {label:<6}: {n:>4} ({pct:.1f}%)")

# ─────────────────────────────────────────
# FIGURA
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Check Post-Treatment Leakage",
             fontsize=13, fontweight="bold")

# Panel 1: distribuzione offset abx
ax = axes[0]
ax.hist(abx_first["abx_offset"].clip(0, 1440),
        bins=48, color="#e74c3c", alpha=0.8, edgecolor="white")
ax.axvline(120, color="steelblue", ls="--", lw=1.5,
           label="Fine finestra vitali (2h)")
ax.axvline(360, color="orange",    ls="--", lw=1.5,
           label="Fine finestra lab (6h)")
ax.set_xlabel("Offset inizio antibiotico (min)")
ax.set_ylabel("N pazienti")
ax.set_title("Distribuzione timing\nprescrizione antibiotico", fontweight="bold")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel 2: % vitali post-trattamento per paziente
ax2 = axes[1]
per_pat_vital = (vital_win.groupby("patientunitstayid")["is_post_treatment"]
                 .mean() * 100)
ax2.hist(per_pat_vital, bins=20,
         color="steelblue", alpha=0.8, edgecolor="white")
ax2.axvline(50, color="red", ls="--", lw=1.5, label="50%")
ax2.set_xlabel("% osservazioni vitali post-trattamento")
ax2.set_ylabel("N pazienti")
ax2.set_title(f"Vitali [0-120min]\n% post-trattamento per paziente", fontweight="bold")
ax2.legend(fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Panel 3: % lab post-trattamento per paziente
ax3 = axes[2]
per_pat_lab = (lab_win.groupby("patientunitstayid")["is_post_treatment"]
               .mean() * 100)
ax3.hist(per_pat_lab, bins=20,
         color="orange", alpha=0.8, edgecolor="white")
ax3.axvline(50, color="red", ls="--", lw=1.5, label="50%")
ax3.set_xlabel("% risultati lab post-trattamento")
ax3.set_ylabel("N pazienti")
ax3.set_title(f"Lab [0-360min]\n% post-trattamento per paziente", fontweight="bold")
ax3.legend(fontsize=9)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("../output_png/leakage_check.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────
# RIEPILOGO E RACCOMANDAZIONE
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("RIEPILOGO")
print("=" * 55)
print(f"  Vitali [0-120min] post-trattamento: {pct_post_vital:.1f}%")
print(f"  Lab    [0-360min] post-trattamento: {pct_post_lab:.1f}%")

if pct_post_vital < 10 and pct_post_lab < 10:
    print("\n  RISULTATO: leakage trascurabile — finestre OK")
elif pct_post_vital < 25 and pct_post_lab < 25:
    print("\n  RISULTATO: leakage moderato — discutere come limite")
else:
    print("\n  RISULTATO: leakage rilevante — considerare finestre piu' strette")
    print("  SOLUZIONE: usare solo misure con offset < abx_offset")