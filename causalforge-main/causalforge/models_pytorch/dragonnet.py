"""
DragonNet - Conversione PyTorch
Originale TensorFlow: https://github.com/claudiashi57/dragonnet
Skeleton: https://github.com/uber/causalml/blob/master/causalml/inference/tf/dragonnet.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils import (
    dragonnet_loss_binarycross,
    EpsilonLayer,
    regression_loss,
    binary_classification_loss,
    treatment_accuracy,
    track_epsilon,
    convert_pd_to_np,
    make_tarreg_loss,
)


# ─────────────────────────────────────────────
# ARCHITETTURA
# ─────────────────────────────────────────────

class DragonNetArchitecture(nn.Module):
    """
    Architettura DragonNet in PyTorch.

    Struttura:
        Input
          ↓
        [3 strati condivisi ELU]     ← rappresentazione Z
          ↓
        ┌──────────────┬─────────────────────────┐
        ↓              ↓                         ↓
      t_pred         y0_head                  y1_head
    (propensity)   [2 strati ELU + output]  [2 strati ELU + output]
        ↓
      epsilon (trainable)
        ↓
      Output: [y0, y1, t_pred, epsilon]  shape: (batch, 4)
    """

    def __init__(self, input_dim, neurons_per_layer=200, reg_l2=0.01):
        super(DragonNetArchitecture, self).__init__()

        self.reg_l2 = reg_l2
        half = neurons_per_layer // 2

        # ── Strati condivisi (rappresentazione Z) ──
        self.shared = nn.Sequential(
            nn.Linear(input_dim, neurons_per_layer),
            nn.ELU(),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ELU(),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ELU(),
        )
        # Inizializzazione RandomNormal come nell'originale
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        # ── Testa propensity score (t_pred) ──
        # sigmoid → output tra 0 e 1
        self.t_head = nn.Sequential(
            nn.Linear(neurons_per_layer, 1),
            nn.Sigmoid()
        )

        # ── Testa Y0 (outcome senza trattamento) ──
        self.y0_head = nn.Sequential(
            nn.Linear(neurons_per_layer, half),
            nn.ELU(),
            nn.Linear(half, half),
            nn.ELU(),
            nn.Linear(half, 1),
            # Nessuna activation finale → regressione libera
        )

        # ── Testa Y1 (outcome con trattamento) ──
        self.y1_head = nn.Sequential(
            nn.Linear(neurons_per_layer, half),
            nn.ELU(),
            nn.Linear(half, half),
            nn.ELU(),
            nn.Linear(half, 1),
        )

        # ── Epsilon Layer (per targeted regularization) ──
        self.epsilon_layer = EpsilonLayer()

    def forward(self, x):
        # Strati condivisi
        z = self.shared(x)                          # (batch, neurons)

        # Propensity score
        t_pred = self.t_head(z)                     # (batch, 1)

        # Outcome Y0 e Y1
        y0 = self.y0_head(z)                        # (batch, 1)
        y1 = self.y1_head(z)                        # (batch, 1)

        # Epsilon
        epsilons = self.epsilon_layer(t_pred)        # (batch, 1)

        # Concatena tutto → (batch, 4)
        out = torch.cat([y0, y1, t_pred, epsilons], dim=1)
        return out


# ─────────────────────────────────────────────
# CLASSE DRAGONNET (training + predict)
# ─────────────────────────────────────────────

class DragonNet:
    """
    DragonNet in PyTorch — equivalente alla classe Keras originale.

    Addestramento in due fasi:
        Fase 1 → Adam   (veloce, esplora lo spazio dei pesi)
        Fase 2 → SGD    (lento e preciso, rifinisce la soluzione)
    """

    def build(self, user_params):
        if 'input_dim' not in user_params:
            raise Exception("input_dim must be specified!")

        # Parametri di default (identici all'originale)
        params = {
            'neurons_per_layer': 200,
            'reg_l2': 0.01,
            'targeted_reg': True,
            'verbose': True,
            'val_split': 0.2,
            'ratio': 1.0,
            'batch_size': 64,
            'epochs': 100,
            'learning_rate': 1e-5,
            'momentum': 0.9,
            'use_adam': True,
            'adam_epochs': 30,
            'adam_learning_rate': 1e-3,
        }
        for k in params:
            params[k] = user_params.get(k, params[k])
        params['input_dim'] = user_params['input_dim']

        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Costruisce l'architettura
        self.model = DragonNetArchitecture(
            input_dim=params['input_dim'],
            neurons_per_layer=params['neurons_per_layer'],
            reg_l2=params['reg_l2'],
        ).to(self.device)

    def support_ite(self):
        return True

    # ── Predizioni ──────────────────────────────

    def predict_ite(self, X):
        """
        Predice l'Individual Treatment Effect per ogni paziente.
        ITE = Y1 - Y0  (differenza tra outcome con e senza trattamento)
        """
        X = convert_pd_to_np(X)
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()

        return preds[:, 1] - preds[:, 0]   # y1 - y0

    def predict_ate(self, X, treatment, y):
        """
        Calcola l'Average Treatment Effect = mean(ITE).
        Usa TUTTI i dati (no data splitting — come da paper DragonNet).
        """
        return np.mean(self.predict_ite(X))

    # ── Training ─────────────────────────────────

    def fit(self, X, treatment, y):
        """
        Addestra DragonNet in due fasi:
            1. Adam  (adam_epochs, adam_learning_rate)
            2. SGD con momentum e Nesterov (epochs, learning_rate)
        """
        # Converti in numpy
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        # Combina y e treatment in un unico array → shape (n, 2)
        # Ogni riga: [y, treatment]
        y_combined = np.hstack((y.reshape(-1, 1), treatment.reshape(-1, 1)))

        # Scegli la loss function
        if self.params.get('targeted_reg', True):
            loss_fn = make_tarreg_loss(
                ratio=self.params['ratio'],
                dragonnet_loss=dragonnet_loss_binarycross
            )
        else:
            loss_fn = dragonnet_loss_binarycross

        # Crea DataLoader con validation split
        dataset = self._make_dataset(X, y_combined)
        train_loader, val_loader = self._split_dataset(
            dataset,
            val_split=self.params['val_split'],
            batch_size=self.params['batch_size']
        )

        # ── Fase 1: Adam ──────────────────────────
        if self.params.get('use_adam', True):
            if self.params['verbose']:
                print("\n── Fase 1: Adam ──────────────────────")

            optimizer_adam = optim.Adam(
                self.model.parameters(),
                lr=self.params['adam_learning_rate']
            )
            # ReduceLROnPlateau equivalente
            scheduler_adam = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_adam, mode='min', factor=0.5,
                patience=5, min_lr=0, eps=1e-8
            )

            self._train_loop(
                optimizer=optimizer_adam,
                scheduler=scheduler_adam,
                loss_fn=loss_fn,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.params['adam_epochs'],
                patience=2,               # EarlyStopping patience Adam
                verbose=self.params['verbose']
            )

        # ── Fase 2: SGD con Momentum ──────────────
        if self.params['verbose']:
            print("\n── Fase 2: SGD con Momentum ──────────")

        optimizer_sgd = optim.SGD(
            self.model.parameters(),
            lr=self.params['learning_rate'],
            momentum=self.params['momentum'],
            nesterov=True               # Nesterov momentum come nell'originale
        )
        scheduler_sgd = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_sgd, mode='min', factor=0.5,
            patience=5, min_lr=0
        )

        self._train_loop(
            optimizer=optimizer_sgd,
            scheduler=scheduler_sgd,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.params['epochs'],
            patience=40,                # EarlyStopping patience SGD
            verbose=self.params['verbose']
        )

    # ── Helpers interni ──────────────────────────

    def _make_dataset(self, X, y_combined):
        """Converte numpy array in TensorDataset PyTorch."""
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y_combined)
        return TensorDataset(X_t, y_t)

    def _split_dataset(self, dataset, val_split, batch_size):
        """Divide il dataset in train e validation set."""
        n_val   = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def _train_loop(self, optimizer, scheduler, loss_fn,
                    train_loader, val_loader,
                    epochs, patience, verbose):
        """
        Loop di training con:
        - EarlyStopping (patience)
        - ReduceLROnPlateau (scheduler)
        - TerminateOnNaN
        """
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            # ── Training ──
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.model(X_batch)

                # L2 regularization manuale (equivalente a kernel_regularizer=l2)
                l2_reg = self._l2_regularization()

                loss = loss_fn(y_batch, preds) + l2_reg
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # TerminateOnNaN
                if torch.isnan(loss):
                    print("⚠️  Loss è NaN — training interrotto.")
                    return

            # ── Validation ──
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(self.device)
                    y_val = y_val.to(self.device)
                    preds_val = self.model(X_val)
                    val_loss += loss_fn(y_val, preds_val).item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)

            # ReduceLROnPlateau — monitora train loss
            scheduler.step(avg_train)

            if verbose:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"train_loss: {avg_train:.4f} | "
                      f"val_loss: {avg_val:.4f} | "
                      f"lr: {current_lr:.2e}")

            # EarlyStopping — monitora val loss
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"  Early stopping alla epoch {epoch+1}")
                    return

    def _l2_regularization(self):
        """
        Calcola la L2 regularization sui pesi delle teste Y0 e Y1.
        Equivalente a kernel_regularizer=l2(reg_l2) in Keras.
        Solo sulle teste — gli strati condivisi non hanno regularization.
        """
        l2_loss = 0.0
        reg = self.params['reg_l2']

        for name, param in self.model.named_parameters():
            # Applica solo alle teste y0 e y1, non agli strati condivisi
            if ('y0_head' in name or 'y1_head' in name) and 'weight' in name:
                l2_loss += reg * torch.sum(param ** 2)

        return l2_loss