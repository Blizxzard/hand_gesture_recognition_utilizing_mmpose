import joblib
import json_tricks as json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Daten laden ----
with open("DataLol.json", "r") as f:
    samples = json.load(f)
X = np.array([s["features"] for s in samples], dtype=np.float32)
y = np.array([s["label"] for s in samples])

# Optional: Manuell splitten, falls nicht beim Training getan
from sklearn.model_selection import train_test_split
# (Sonst Testdaten direkt laden)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Modell laden ----
clf = joblib.load("DataLol.pkl")

# ---- Vorhersage ----
y_pred = clf.predict(X_test)

# ---- Metriken ----
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)
print("Classification report:\n", report)
print("Confusion matrix:\n", cm)

# ---- Speichern als CSV ----
classes = np.unique(y)  # oder Liste der Klassen
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.to_csv("confusion_matrix.csv")

# Normale Confusion Matrix (absolut)
cm = confusion_matrix(y_test, y_pred)

# Normalisierte Confusion Matrix (pro Klasse)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # division by row sums
cm_norm = np.nan_to_num(cm_norm)  # für den Fall, dass eine Klasse nicht vorkommt

# Als PNG speichern
plt.figure(figsize=(6,5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes, cmap="viridis")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (normalized)")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png")
plt.close()

# Optional: als CSV speichern
pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv("confusion_matrix_normalized.csv")

# ---- Confusion Matrix Plot ----
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
