from collections import deque
import torch
import torch.nn as nn
import numpy as np


# ==========================================================
# LSTM-Modell für sequentielle Gestenerkennung
# ==========================================================
class LSTMNet(nn.Module):
    """
    Einfaches LSTM-basiertes Sequenzmodell.
    Nutzt den letzten Zeitschritt zur Klassifikation.
    """
    def __init__(self, input_dim, num_classes, hidden=128, layers=1, bidir=True, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=(dropout if layers > 1 else 0.0)
        )
        out_dim = hidden * (2 if bidir else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):  # x: (N, T, D)
        y, _ = self.lstm(x)
        feat = y[:, -1, :]
        return self.fc(self.dropout(feat))


# ==========================================================
# Transformer-Modell für sequentielle Gestenerkennung
# ==========================================================
class TransformerNet(nn.Module):
    """
    Transformer-basierter Encoder für Sequenzklassifikation.
    Nutzt den letzten Token als Repräsentation (kein CLS-Token).
    """
    def __init__(self, input_dim, num_classes, model_dim=128, heads=4, layers=2,
                 mlp_dim=256, dropout=0.1, max_len=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pos = nn.Parameter(torch.randn(1, max_len, model_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):  # x: (N, T, D)
        n, t, _ = x.shape
        h = self.proj(x) + self.pos[:, :t, :]
        h = self.encoder(self.dropout(h))
        feat = h[:, -1, :]          
        return self.fc(self.dropout(feat))


# ==========================================================
# Wrapper-Klasse zur Integration in bestehende Pipelines
# ==========================================================
class SequenceModelWrapper:
    def __init__(self, arch: str, arch_kwargs: dict, state_dict: dict,
                 classes, win: int, feat_dim: int, scaler=None, device_preference="cuda"):
        self.arch = arch
        self.arch_kwargs = arch_kwargs
        self._state_dict = {k: v.cpu() for k, v in state_dict.items()}  # Modellparameter auf CPU sichern
        self.classes_ = np.array(list(classes))
        self.win = int(win)
        self.feat_dim = int(feat_dim)
        self.scaler = scaler
        self.device_pref = device_preference
        self._model = None
        self._device = None

    def _ensure_model(self):
        """
        Lädt und initialisiert das Torch-Modell bei Bedarf.
        """
        if self._model is not None:
            return
        num_classes = len(self.classes_)

        # Modellarchitektur anhand der arch-Angabe auswählen
        if self.arch.lower() == "lstm":
            net = LSTMNet(input_dim=self.feat_dim, num_classes=num_classes, **self.arch_kwargs)
        elif self.arch.lower() in ("transformer", "transformer_encoder", "tx"):
            net = TransformerNet(input_dim=self.feat_dim, num_classes=num_classes, **self.arch_kwargs)
        else:
            raise ValueError(f"Unbekannte Architektur: {self.arch}")

        # Gerät wählen (GPU bevorzugt, falls verfügbar)
        device = "cuda" if (self.device_pref.startswith("cuda") and torch.cuda.is_available()) else "cpu"

        # Gewichtsdaten laden und Modell in Evaluationsmodus setzen
        net.load_state_dict(self._state_dict, strict=True)
        net.eval().to(device)
        self._model = net
        self._device = device

    def _prep(self, X_flat: np.ndarray) -> torch.Tensor:
        """
        Wandelt flache Eingaben (N, win*feat_dim) in Tensorform (N, T, D) um
        und wendet optional Standardisierung an.
        """
        X = np.asarray(X_flat, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        N, F = X.shape
        T = self.win
        D = self.feat_dim
        assert F == T * D, f"Featurelänge {F} stimmt nicht mit win*feat_dim {T*D} überein"
        X = X.reshape(N, T, D)

        if self.scaler is not None:
            X = X.reshape(N, -1)
            X = self.scaler.transform(X).reshape(N, T, D)
        return torch.from_numpy(X)

    @torch.inference_mode()
    def predict_proba(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Berechnet Klassenwahrscheinlichkeiten für Eingabefenster.
        Gibt numpy-Array (N, C) zurück.
        """
        self._ensure_model()
        x = self._prep(X_flat).to(self._device)
        logits = self._model(x)                     # (N, C)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X_flat: np.ndarray):
        """
        Gibt das Klassenlabel mit höchster Wahrscheinlichkeit zurück.
        """
        probs = self.predict_proba(X_flat)
        idx = probs.argmax(axis=1)
        return self.classes_[idx]
