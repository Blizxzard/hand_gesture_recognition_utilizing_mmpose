import argparse, os, glob, json_tricks as json, numpy as np, joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import torch
from collections import defaultdict

# ==========================================================
# Hilfsfunktionen
# ==========================================================

def load_jsonl(dir_path):
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

def build_arrays_concat_stride(samples, classes, win=None, stride=4):
    X_seq, Y = [], []
    cls_to_id = {c: i for i, c in enumerate(classes)}
    
    label_to_seqs = {label: [] for label in classes}
    for s in samples:
        label = s["label"]
        if label not in cls_to_id:
            continue
        seq = np.asarray(s["features"], dtype=np.float32)
        if seq.shape[0] > 0:
            label_to_seqs[label].append(seq)
    
    T = win
    if T is None:
        raise ValueError("win (Fensterlänge) muss angegeben werden")
    
    for label, seqs in label_to_seqs.items():
        if len(seqs) == 0:
            continue
        
        concat_seq = np.vstack(seqs)
        L, D = concat_seq.shape
        
        if L < T:
            continue
        
        for start in range(0, L - T + 1, stride):
            window = concat_seq[start:start+T]
            X_seq.append(window)
            Y.append(cls_to_id[label])
    
    X = np.stack(X_seq, axis=0)
    y = np.asarray(Y, dtype=np.int64)
    
    return X, y

def build_arrays_per_sample_stride(samples, classes, win=None, stride=4):
    X_seq, Y = [], []
    cls_to_id = {c: i for i, c in enumerate(classes)}
    
    T = win
    if T is None:
        raise ValueError("win (Fensterlänge) muss angegeben werden")
    
    for s in samples:
        label = s["label"]
        if label not in cls_to_id:
            continue
        
        seq = np.asarray(s["features"], dtype=np.float32)
        L, D = seq.shape
        
        if L < T:
            continue
        
        # Stride innerhalb eines Samples
        for start in range(0, L - T + 1, stride):
            window = seq[start:start+T]
            X_seq.append(window)
            Y.append(cls_to_id[label])
    
    X = np.stack(X_seq, axis=0)
    y = np.asarray(Y, dtype=np.int64)
    
    return X, y

# ==========================================================
# Evaluierungsfunktion
# ==========================================================

def evaluate_model(model_wrapper, X_test, y_test, class_names):

    N, T, D = X_test.shape
    X_flat = X_test.reshape(N, -1).astype(np.float32)
    
    y_pred = model_wrapper.predict(X_flat)
    y_pred_ids = np.array([np.where(class_names == c)[0][0] for c in y_pred])
    
    accuracy = accuracy_score(y_test, y_pred_ids)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred_ids, labels=np.arange(len(class_names)), zero_division=0
    )
    
    cm = confusion_matrix(y_test, y_pred_ids, labels=np.arange(len(class_names)))
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "true_labels": y_test
    }
    
    return results

# ==========================================================
# Visualisierung und Reporting
# ==========================================================

def print_results(results, class_names):
    accuracy = results["accuracy"]
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]
    support = results["support"]
    cm = results["confusion_matrix"]
    
    print("\n" + "="*70)
    print("EVALUIERUNGSERGEBNISSE")
    print("="*70)
    
    print(f"\nGesamtgenauigkeit (Accuracy): {accuracy:.4f}")
    
    print("\nKlassenweise Metriken:")
    print("-" * 70)
    print(f"{'Klasse':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
    print("-" * 70)
    
    for i, cls_name in enumerate(class_names):
        print(f"{cls_name:<20} {precision[i]:<15.4f} {recall[i]:<15.4f} {f1[i]:<15.4f} {int(support[i]):<10}")
    
    print("-" * 70)
    print(f"{'Mittelwert':<20} {precision.mean():<15.4f} {recall.mean():<15.4f} {f1.mean():<15.4f} {int(support.sum()):<10}")
    
    print("\nConfusion Matrix:")
    print("-" * 70)
    print("Zeilen: True Label | Spalten: Vorhersage")
    print("\n", cm)
    
    print("\nDetaillierter Classification Report:")
    print("-" * 70)
    print(classification_report(results["true_labels"], 
                              np.array([np.where(class_names == c)[0][0] 
                                       for c in results["predictions"]]),
                              target_names=class_names))

# ==========================================================
# Hauptprogramm
# ==========================================================

def main():
    ap = argparse.ArgumentParser(description="Evaluiert trainiertes Gestenmodell")
    ap.add_argument("--model", default="dynamic_gesture_concat_stride.joblib",
                    help="Pfad zum trainierten Modell")
    ap.add_argument("--test_dir", default="dyn_data",
                    help="Pfad zu den Test-JSONL-Dateien")
    ap.add_argument("--win", type=int, default=30,
                    help="Fenstergröße (muss mit Training übereinstimmen)")
    ap.add_argument("--stride", type=int, default=4,
                    help="Schrittweite (muss mit Training übereinstimmen)")
    ap.add_argument("--method", choices=["concat_stride", "per_sample_stride"],
                    default="concat_stride",
                    help="Methode zur Fenstergenerierung")
    ap.add_argument("--output_report", default="evaluation_report.txt",
                    help="Ausgabedatei für detaillierten Report")
    
    args = ap.parse_args()
    

    print(f"Lade Modell von {args.model}...")
    try:
        payload = joblib.load(args.model)
        model_wrapper = payload["model"]
        win = payload["win"]
        classes = payload["classes"]
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return
    
    print(f"Modell geladen. Fenster={win}, Klassen={classes}")
    

    print(f"Lade Testdaten von {args.test_dir}...")
    test_samples = load_jsonl(args.test_dir)
    print(f"Insgesamt {len(test_samples)} Samples geladen")
    
    if len(test_samples) == 0:
        print("Keine Testdaten gefunden!")
        return
    
    if args.method == "concat_stride":
        print("Verwende Konkatenation + Stride Methode...")
        X_test, y_test = build_arrays_concat_stride(test_samples, classes=classes, 
                                                     win=win, stride=args.stride)
    else:
        print("Verwende Per-Sample Stride Methode...")
        X_test, y_test = build_arrays_per_sample_stride(test_samples, classes=classes,
                                                        win=win, stride=args.stride)
    
    print(f"Testset: N={X_test.shape[0]}, T={X_test.shape[1]}, D={X_test.shape[2]}")
    
    N, T, D = X_test.shape
    X_flat = X_test.reshape(N, -1)
    if model_wrapper.scaler is not None:
        X_flat = model_wrapper.scaler.transform(X_flat)
    X_test = X_flat.reshape(N, T, D)
    
    print("\nFühre Evaluierung durch...")
    results = evaluate_model(model_wrapper, X_test, y_test, classes)
    
    print_results(results, classes)
    
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("EVALUIERUNGSBERICHT - DYNAMISCHE GESTENERKENNUNG\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Modell: {args.model}\n")
        f.write(f"Testdaten: {args.test_dir}\n")
        f.write(f"Methode: {args.method}\n")
        f.write(f"Fenster: {win}, Stride: {args.stride}\n")
        f.write(f"Testset Größe: {X_test.shape[0]} Samples\n\n")
        
        f.write(f"Gesamtgenauigkeit: {results['accuracy']:.4f}\n\n")
        
        f.write("Klassenweise Metriken:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Klasse':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}\n")
        f.write("-" * 70 + "\n")
        
        for i, cls_name in enumerate(classes):
            f.write(f"{cls_name:<20} {results['precision'][i]:<15.4f} "
                   f"{results['recall'][i]:<15.4f} {results['f1'][i]:<15.4f} "
                   f"{int(results['support'][i]):<10}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(results['confusion_matrix']) + "\n")
    
    print(f"\nReport gespeichert unter {args.output_report}")

if __name__ == "__main__":
    main()
