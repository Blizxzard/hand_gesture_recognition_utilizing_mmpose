import argparse, os, glob, json_tricks as json, numpy as np, joblib, random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lstm_transformer import LSTMNet, TransformerNet, SequenceModelWrapper


# ==========================================================
# Hilfsfunktionen
# ==========================================================

def load_jsonl(dir_path):
    """
    Lädt alle .jsonl-Dateien aus einem Verzeichnis und gibt
    eine Liste aus JSON-Objekten zurück.
    """
    paths = sorted(glob.glob(os.path.join(dir_path, "*.jsonl")))
    samples = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    samples.append(obj)
                except Exception:
                    pass 
    return samples


def build_arrays(samples, classes, win=None, make_overlap=True, stride=4):
    """
    Baut NumPy-Arrays aus geladenen JSON-Samples.

    - Jede Probe enthält eine Liste von Feature-Vektoren ("features")
    - Erzeugt Sequenzen der Form (N, T, D)
    - Optional mit überlappenden Fenstern zur Datenaugmentation

    Parameter:
        samples: Liste der geladenen JSON-Samples
        classes: bekannte Klassenbezeichnungen
        win: Fensterlänge (T)
        make_overlap: True für überlappende Segmente
        stride: Schrittweite für Überlappung
    """
    X_seq, Y = [], []
    cls_to_id = {c: i for i, c in enumerate(classes)}

    for s in samples:
        label = s["label"]
        if label not in cls_to_id:
            continue
        seq = np.asarray(s["features"], dtype=np.float32)
        L, D = seq.shape
        T = win or L
        if L < T:
            continue

        if make_overlap:
            # Fenster mit Überlappung generieren
            for start in range(0, L - T + 1, stride):
                X_seq.append(seq[start:start+T])
                Y.append(cls_to_id[label])
        else:
            # Nur ein Fenster pro Sequenz
            X_seq.append(seq[:T])
            Y.append(cls_to_id[label])

    X = np.stack(X_seq, axis=0)          # (N, T, D)
    y = np.asarray(Y, dtype=np.int64)
    return X, y


# ==========================================================
# Dataset-Klasse für Torch DataLoader
# ==========================================================
class SeqDataset(Dataset):
    """
    Torch Dataset für Sequenzdaten (X: float32, y: int64).
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]


def set_seed(seed=42):
    """
    Setzt Zufallssamen für Reproduzierbarkeit.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================================
# Trainingsroutine
# ==========================================================
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, device="cuda"):
    """
    Führt Training und Validierung über mehrere Epochen durch.

    Parameter:
        model: Torch-Modell (LSTM oder Transformer)
        train_loader: Trainings-DataLoader
        val_loader: Validierungs-DataLoader
        epochs: Anzahl der Trainingsdurchläufe
        lr: Lernrate
        device: "cuda" oder "cpu"
    """
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best = {"acc": 0.0, "state": None}
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss)

        # Validierung
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=-1)
                correct += int((pred == yb).sum())
                total += yb.numel()
        acc = correct / max(1, total)

        # Bestes Modell speichern
        if acc > best["acc"]:
            best["acc"] = acc
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        sched.step()
        print(f"Epoche {ep:02d} | Trainingsverlust={tr_loss/len(train_loader):.4f} | Val.Acc={acc:.3f}")

    return best


# ==========================================================
# Hauptprogramm
# ==========================================================
def main():
    """
    Trainiert ein dynamisches Gestenmodell (LSTM oder Transformer)
    auf JSONL-Daten und speichert das Modell in einem .joblib-Wrapper.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dyn_data", help="Pfad zu den JSONL-Trainingsdaten")
    ap.add_argument("--arch", choices=["lstm", "transformer"], default="lstm", help="Modellarchitektur")
    ap.add_argument("--win", type=int, default=30, help="Fenstergröße der Sequenzen")
    ap.add_argument("--stride", type=int, default=4, help="Schrittweite für überlappende Fenster")
    ap.add_argument("--batch", type=int, default=64, help="Batchgröße")
    ap.add_argument("--epochs", type=int, default=40, help="Anzahl Trainings-Epochen")
    ap.add_argument("--lr", type=float, default=1e-3, help="Lernrate")
    ap.add_argument("--hidden", type=int, default=128, help="LSTM-Hidden-Dimension")
    ap.add_argument("--layers", type=int, default=1, help="Anzahl Schichten im Modell")
    ap.add_argument("--bidir", action="store_true", help="Bidirektionales LSTM aktivieren")
    ap.add_argument("--dropout", type=float, default=0.1, help="Dropout-Rate")
    ap.add_argument("--model_dim", type=int, default=128, help="Transformer-Modell-Dimension")
    ap.add_argument("--heads", type=int, default=4, help="Anzahl der Attention-Köpfe")
    ap.add_argument("--mlp_dim", type=int, default=256, help="Feedforward-Dimension im Transformer")
    ap.add_argument("--outfile", default="dynamic_gesture_rf.joblib",
                    help="Ziel-Dateiname für exportiertes Modell")
    args = ap.parse_args()

    set_seed(42)

    # 1) Daten laden
    raw = load_jsonl(args.data_dir)
    # classes from collector:
    all_labels = sorted({s["label"] for s in raw})
    print("Klassen:", all_labels)

    # 2) Feature-Arrays aufbauen (N, T, D)
    X, y = build_arrays(raw, classes=all_labels, win=args.win, make_overlap=True, stride=args.stride)
    N, T, D = X.shape
    print(f"Datensatz: N={N}, T={T}, D={D}")

    # 3) scale features
    scaler = StandardScaler()
    X_flat = X.reshape(N, -1)
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(N, T, D)

    # 4) Trainings-/Validierungssplit
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    tr_ds, va_ds = SeqDataset(Xtr, ytr), SeqDataset(Xva, yva)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # 5) Modell erstellen
    num_classes = len(all_labels)
    if args.arch == "lstm":
        arch_kwargs = dict(hidden=args.hidden, layers=args.layers, bidir=args.bidir, dropout=args.dropout)
        model = LSTMNet(input_dim=D, num_classes=num_classes, **arch_kwargs)
    else:
        arch_kwargs = dict(model_dim=args.model_dim, heads=args.heads, layers=args.layers,
                           mlp_dim=args.mlp_dim, dropout=args.dropout, max_len=max(512, T))
        model = TransformerNet(input_dim=D, num_classes=num_classes, **arch_kwargs)

    # 6) Training durchführen
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best = train_model(model, tr_ld, va_ld, epochs=args.epochs, lr=args.lr, device=device)

    # 6) save wrapper
    wrapper = SequenceModelWrapper(
        arch=args.arch,
        arch_kwargs=arch_kwargs,
        state_dict=best["state"],
        classes=all_labels,
        win=T,
        feat_dim=D,
        scaler=scaler,               
        device_preference=device
    )

    payload = {"model": wrapper, "win": T, "classes": np.array(all_labels)}
    joblib.dump(payload, args.outfile)
    print(f"Gespeichert unter {args.outfile} (Val.Acc={best['acc']:.3f})")


if __name__ == "__main__":
    main()
