"""
Test DragonNet su dati sintetici con effetto causale noto
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import random
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# DRAGONNET IMPLEMENTAZIONE (versione corretta)
# ─────────────────────────────────────────

class DragonNet(nn.Module):
    """DragonNet con output sigmoid per probabilità valide"""
    
    def __init__(self, input_dim, neurons_per_layer=64, reg_l2=0.01):
        super(DragonNet, self).__init__()
        
        self.input_dim = input_dim
        self.neurons_per_layer = neurons_per_layer
        self.reg_l2 = reg_l2
        
        # Shared representation
        self.rep_layer1 = nn.Linear(input_dim, neurons_per_layer)
        self.rep_layer2 = nn.Linear(neurons_per_layer, neurons_per_layer)
        self.rep_layer3 = nn.Linear(neurons_per_layer, neurons_per_layer)
        self.activation = nn.ELU()
        
        # Treatment head (sigmoid per probabilità)
        self.treatment_head = nn.Linear(neurons_per_layer, 1)
        
        # Outcome heads (con sigmoid alla fine per probabilità in [0,1])
        hidden_size = neurons_per_layer // 2
        
        self.y0_layer1 = nn.Linear(neurons_per_layer, hidden_size)
        self.y0_layer2 = nn.Linear(hidden_size, hidden_size)
        self.y0_out = nn.Linear(hidden_size, 1)
        
        self.y1_layer1 = nn.Linear(neurons_per_layer, hidden_size)
        self.y1_layer2 = nn.Linear(hidden_size, hidden_size)
        self.y1_out = nn.Linear(hidden_size, 1)
        
        # Epsilon for targeted regularization
        self.epsilon = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        # Shared representation
        x = self.activation(self.rep_layer1(x))
        x = self.activation(self.rep_layer2(x))
        x = self.activation(self.rep_layer3(x))
        
        # Treatment prediction (sigmoid per probabilità)
        t_pred = torch.sigmoid(self.treatment_head(x))
        
        # Y0 prediction (sigmoid per probabilità)
        y0_h = self.activation(self.y0_layer1(x))
        y0_h = self.activation(self.y0_layer2(y0_h))
        y0_pred = torch.sigmoid(self.y0_out(y0_h))
        
        # Y1 prediction (sigmoid per probabilità)
        y1_h = self.activation(self.y1_layer1(x))
        y1_h = self.activation(self.y1_layer2(y1_h))
        y1_pred = torch.sigmoid(self.y1_out(y1_h))
        
        # Epsilon term
        epsilon_term = t_pred * self.epsilon
        
        return y0_pred, y1_pred, t_pred, epsilon_term
    
    def get_l2_loss(self):
        """Compute L2 regularization loss"""
        l2_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name and 'epsilon' not in name:
                l2_loss += torch.norm(param, 2) ** 2
        return self.reg_l2 * l2_loss


class DragonNetModel:
    """Wrapper per addestramento DragonNet"""
    
    def __init__(self, input_dim, neurons_per_layer=64, reg_l2=0.01,
                 epochs=100, learning_rate=1e-3, batch_size=64, 
                 val_split=0.2, verbose=True):
        
        self.input_dim = input_dim
        self.neurons_per_layer = neurons_per_layer
        self.reg_l2 = reg_l2
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.val_split = val_split
        self.verbose = verbose
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
    
    def build(self):
        """Build the model"""
        self.model = DragonNet(
            input_dim=self.input_dim,
            neurons_per_layer=self.neurons_per_layer,
            reg_l2=self.reg_l2
        ).to(self.device)
    
    def _prepare_data(self, X, t, y):
        """Prepare data loaders"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        t_tensor = torch.FloatTensor(t).reshape(-1, 1).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        # Combine for loss
        y_combined = torch.cat([y_tensor, t_tensor], dim=1)
        
        dataset = TensorDataset(X_tensor, y_combined)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _loss_function(self, y_true, y0_pred, y1_pred, t_pred, epsilon):
        """DragonNet loss with binary cross-entropy"""
        y = y_true[:, 0]  # outcome
        t = y_true[:, 1]  # treatment
        
        # Outcome loss (use predicted outcome based on actual treatment)
        y_pred = torch.where(t == 1, y1_pred, y0_pred)
        outcome_loss = nn.functional.binary_cross_entropy(y_pred, y)
        
        # Treatment loss
        treatment_loss = nn.functional.binary_cross_entropy(t_pred, t)
        
        # Targeted regularization
        eps_term = epsilon * (2 * t - 1) * (y - (y1_pred - y0_pred))
        
        total_loss = outcome_loss + treatment_loss + eps_term.mean()
        
        return total_loss
    
    def fit(self, X, t, y):
        """Train the model"""
        train_loader, val_loader = self._prepare_data(X, t, y)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                y0_pred, y1_pred, t_pred, epsilon = self.model(X_batch)
                loss = self._loss_function(y_batch, y0_pred, y1_pred, t_pred, epsilon)
                loss += self.model.get_l2_loss()
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y0_pred, y1_pred, t_pred, epsilon = self.model(X_batch)
                    loss = self._loss_function(y_batch, y0_pred, y1_pred, t_pred, epsilon)
                    loss += self.model.get_l2_loss()
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Early stopping
            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                      f"train_loss: {avg_train_loss:.4f} | "
                      f"val_loss: {avg_val_loss:.4f}")
            
            if patience_counter >= patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
    
    def predict_ite(self, X):
        """Predict Individual Treatment Effect"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            y0_pred, y1_pred, _, _ = self.model(X_tensor)
            ite = (y1_pred - y0_pred).cpu().numpy()
        
        return ite.flatten()
    
    def predict_ate(self, X, t=None, y=None):
        """Predict Average Treatment Effect"""
        ite = self.predict_ite(X)
        return np.mean(ite)
    
    def predict_potential_outcomes(self, X):
        """Predict potential outcomes"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            y0_pred, y1_pred, _, _ = self.model(X_tensor)
        
        return y0_pred.cpu().numpy().flatten(), y1_pred.cpu().numpy().flatten()


# ─────────────────────────────────────────
# FUNZIONE PER FISSARE IL SEED
# ─────────────────────────────────────────

def set_seed(seed):
    """Fissa tutti i seed per riproducibilità"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────
# TEST SU DATI SINTETICI
# ─────────────────────────────────────────

print("=" * 60)
print("TEST DRAGONNET SU DATI SINTETICI")
print("=" * 60)

# Genera dati sintetici
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
ate_true = 0.05  # +5%

# Crea trattamento e outcome con effetto noto
t = np.random.binomial(1, 0.5, n)
y = 0.1 * X[:, 0] + ate_true * t + np.random.randn(n) * 0.1
y = (y > 0).astype(float)  # binarizza

print(f"\nDati sintetici:")
print(f"  n = {n}")
print(f"  Trattati: {t.sum()} ({t.mean()*100:.1f}%)")
print(f"  Outcome positivo: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  ATE vero: {ate_true*100:+.2f}%")

# Testa DragonNet con diversi seed
print("\n" + "=" * 60)
print("TEST CON DIVERSI SEED")
print("=" * 60)

results = []

for seed in [42, 123, 456, 789, 1024]:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed)
    
    model = DragonNetModel(
        input_dim=10,
        neurons_per_layer=64,
        reg_l2=0.01,
        epochs=100,
        learning_rate=1e-3,
        batch_size=64,
        verbose=False
    )
    model.build()
    model.fit(X, t, y)
    
    ate = model.predict_ate(X)
    mu0, mu1 = model.predict_potential_outcomes(X)
    
    results.append({
        'seed': seed,
        'ate': ate,
        'mu0_mean': mu0.mean(),
        'mu1_mean': mu1.mean(),
        'mu0_range': [mu0.min(), mu0.max()],
        'mu1_range': [mu1.min(), mu1.max()]
    })
    
    print(f"  ATE: {ate*100:+.2f}% (vero: +5.00%)")
    print(f"  μ0 mean: {mu0.mean():.4f} (range: [{mu0.min():.4f}, {mu0.max():.4f}])")
    print(f"  μ1 mean: {mu1.mean():.4f} (range: [{mu1.min():.4f}, {mu1.max():.4f}])")
    
    # Check per valori fuori range
    if mu0.max() > 1 or mu0.min() < 0 or mu1.max() > 1 or mu1.min() < 0:
        print(f"  ⚠️  WARNING: Outcome predictions fuori [0,1]!")

# Statistiche finali
print("\n" + "=" * 60)
print("RIEPILOGO SU 5 SEED")
print("=" * 60)

ates = [r['ate'] for r in results]
print(f"ATE medio: {np.mean(ates)*100:+.2f}%")
print(f"ATE std:   {np.std(ates)*100:+.2f}%")
print(f"ATE range: [{np.min(ates)*100:+.2f}%, {np.max(ates)*100:+.2f}%]")

# Confronto con ATE vero
print(f"\nATE vero:   +5.00%")
print(f"Errore medio: {abs(np.mean(ates) - ate_true)*100:+.2f}%")

if np.std(ates) * 100 < 2:
    print("\n✅ Stabilità BUONA: variazione < 2% tra seed")
elif np.std(ates) * 100 < 5:
    print("\n⚠️  Stabilità MODERATA: variazione < 5% tra seed")
else:
    print("\n❌ Stabilità SCARSA: variazione > 5% tra seed")