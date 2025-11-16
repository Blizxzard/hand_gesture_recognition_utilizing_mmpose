# train_gesture_model.py

import json_tricks as json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Daten laden
with open("DataLol.json","r") as f:
    samples = json.load(f)

X = [s["features"] for s in samples]
y = [s["label"]    for s in samples]

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Test-Accuracy:", clf.score(X_test, y_test))

joblib.dump(clf, "DataLol.pkl")
print("Modell gespeichert als gesture_model.pkl")
