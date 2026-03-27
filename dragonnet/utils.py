"""
DragonNet utils - PyTorch conversion
Original TensorFlow implementation converted to PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def convert_pd_to_np(*args):
    """Converte pandas DataFrame/Series in numpy array se necessario."""
    output = [obj.to_numpy() if hasattr(obj, "to_numpy") else obj for obj in args]
    return output if len(output) > 1 else output[0]


# ─────────────────────────────────────────────
# EPSILON LAYER
# ─────────────────────────────────────────────

class EpsilonLayer(nn.Module):
    """
    Layer custom che impara epsilon durante il training.
    Equivalente al custom Keras Layer originale.
    
    epsilon è un singolo peso scalare trainable,
    replicato per ogni campione del batch.
    """

    def __init__(self):
        super(EpsilonLayer, self).__init__()
        # Peso scalare trainable, inizializzato casualmente
        self.epsilon = nn.Parameter(torch.randn(1, 1))

    def forward(self, inputs):
        # Restituisce epsilon replicato per ogni campione
        # inputs shape: (batch_size, 1)
        # output shape: (batch_size, 1)
        return self.epsilon * torch.ones_like(inputs)[:, 0:1]


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────

def binary_classification_loss(concat_true, concat_pred):
    """
    Loss di classificazione binaria (binary cross-entropy) sul propensity score.

    Args:
        concat_true: tensor (n_samples, 2) → ogni riga è (y, treatment)
        concat_pred: tensor (n_samples, 4) → ogni riga è (y0, y1, propensity, epsilon)

    Returns:
        float: binary cross-entropy loss sul trattamento
    """
    t_true = concat_true[:, 1]           # trattamento vero (0 o 1)
    t_pred = concat_pred[:, 2]           # propensity score predetto

    # Clip per stabilità numerica (evita log(0))
    # Equivalente al (t_pred + 0.001) / 1.002 originale
    t_pred = (t_pred + 0.001) / 1.002

    # Binary cross-entropy: -[t*log(p) + (1-t)*log(1-p)]
    loss_t = F.binary_cross_entropy(t_pred, t_true, reduction='sum')

    return loss_t


def regression_loss(concat_true, concat_pred):
    """
    Loss di regressione (MSE) sull'outcome Y0 e Y1.

    Nota: calcola l'errore solo sul gruppo corretto —
    - loss0 sui pazienti NON trattati (t=0) → usa y0_pred
    - loss1 sui pazienti trattati     (t=1) → usa y1_pred

    Args:
        concat_true: tensor (n_samples, 2) → ogni riga è (y, treatment)
        concat_pred: tensor (n_samples, 4) → ogni riga è (y0, y1, propensity, epsilon)

    Returns:
        float: somma delle due loss di regressione
    """
    y_true = concat_true[:, 0]           # outcome vero
    t_true = concat_true[:, 1]           # trattamento vero

    y0_pred = concat_pred[:, 0]          # outcome predetto senza trattamento
    y1_pred = concat_pred[:, 1]          # outcome predetto con trattamento

    # Errore quadratico solo sul gruppo corrispondente
    loss0 = torch.sum((1.0 - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true         * torch.square(y_true - y1_pred))

    return loss0 + loss1


def dragonnet_loss_binarycross(concat_true, concat_pred):
    """
    Loss totale DragonNet = regression_loss + binary_classification_loss.

    Combina:
    - errore sull'outcome (regressione)
    - errore sul propensity score (classificazione binaria)

    Args:
        concat_true: tensor (n_samples, 2) → ogni riga è (y, treatment)
        concat_pred: tensor (n_samples, 4) → ogni riga è (y0, y1, propensity, epsilon)

    Returns:
        float: loss totale DragonNet
    """
    return regression_loss(concat_true, concat_pred) + \
           binary_classification_loss(concat_true, concat_pred)


def make_tarreg_loss(ratio=1.0, dragonnet_loss=dragonnet_loss_binarycross):
    """
    Aggiunge la targeted regularization alla loss DragonNet.

    La targeted regularization spinge il modello verso stime causali
    statisticamente ottimali (dalla teoria TMLE).

    Args:
        ratio  (float):    peso della targeted regularization (default 1.0)
        dragonnet_loss:    loss base da usare (default dragonnet_loss_binarycross)

    Returns:
        function: loss con targeted regularization
    """

    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        """
        Loss DragonNet + targeted regularization.

        Il termine aggiuntivo minimizza la influence function,
        garantendo doppia robustezza nella stima dell'ATE.
        """
        # Loss base (regression + classification)
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true  = concat_true[:, 0]
        t_true  = concat_true[:, 1]

        y0_pred  = concat_pred[:, 0]
        y1_pred  = concat_pred[:, 1]
        t_pred   = concat_pred[:, 2]
        epsilons = concat_pred[:, 3]

        # Clip per stabilità numerica
        t_pred = (t_pred + 0.01) / 1.02

        # Outcome predetto in base al trattamento reale ricevuto
        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        # Influence function h(x) — cuore della targeted regularization
        # h = t/e(x) - (1-t)/(1-e(x))
        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        # Outcome perturbato da epsilon × h
        y_pert = y_pred + epsilons * h

        # Targeted regularization: minimizza l'errore sull'outcome perturbato
        targeted_regularization = torch.sum(torch.square(y_true - y_pert))

        # Loss finale
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss


# ─────────────────────────────────────────────
# METRICS (per monitorare il training)
# ─────────────────────────────────────────────

def treatment_accuracy(concat_true, concat_pred):
    """
    Accuratezza binaria sulla predizione del trattamento (propensity score).

    Args:
        concat_true: tensor (n_samples, 2)
        concat_pred: tensor (n_samples, 4)

    Returns:
        float: percentuale di trattamenti predetti correttamente
    """
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]

    # Soglia 0.5: se propensity > 0.5 → predice trattato
    t_pred_binary = (t_pred >= 0.5).float()
    accuracy = (t_pred_binary == t_true).float().mean()

    return accuracy


def track_epsilon(concat_true, concat_pred):
    """
    Monitora il valore assoluto medio di epsilon durante il training.
    Epsilon piccolo = targeted regularization ha poco effetto.
    Epsilon grande  = targeted regularization sta correggendo molto.

    Args:
        concat_true: tensor (n_samples, 2)
        concat_pred: tensor (n_samples, 4)

    Returns:
        float: |mean(epsilon)|
    """
    epsilons = concat_pred[:, 3]
    return torch.abs(torch.mean(epsilons))