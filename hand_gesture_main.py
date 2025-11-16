import warnings

# Unterdrückt spezifische Framework-Warnungen, die im Produktivbetrieb nicht relevant sind
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.autocast.*"
)
warnings.filterwarnings(
    "ignore",
    message=r"RNN module weights are not part of single contiguous chunk of memory.*",
    category=UserWarning,
)

import threading
import queue
import torch
import os, numpy as np, joblib

from dynamic_gestures import SeqFeatureBuffer, TemporalNMS

import joblib
import pyrealsense2 as rs
import cv2

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmpose.utils import adapt_mmdet_pipeline

from collections import deque, Counter  # History-Buffer und Mehrheitsentscheidung


# =========================
# 1) Modelle und Klassifikatoren laden
# =========================

# Hand-Detektor initialisieren (MMDetection)
detector = init_detector(
    'hand2dconfigdet.py',
    'hand2dmodeldet.pth',
    device='cuda:0'
)
# Inferenz-Pipeline des Detektors für MMPose anpassen
detector.cfg = adapt_mmdet_pipeline(detector.cfg)

# 2D-Hand-Pose-Schätzer initialisieren (MMPose)
pose_estimator = init_pose_model(
    'hand2dconfigpose.py',
    'hand2dmodelpose.pth',
    device='cuda:0',
    cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
)

# Statischer Gestenklassifikator (z. B. RandomForest/LogReg) laden
clf = joblib.load("DataLol.pkl")
classes = list(clf.classes_)

# === Optional: dynamischer Gestenklassifikator (Sequenzmodell) laden ===
dyn_model = None
dyn_meta = None
dyn_win = 30  # Sequenzfensterlänge für Feature-Puffer

try:
    if os.path.exists("dynamic_gesture_rf.joblib"):
        dyn_meta = joblib.load("dynamic_gesture_rf.joblib")
        if isinstance(dyn_meta, dict):
            dyn_model = dyn_meta["model"]
            dyn_win = dyn_meta.get("win", 32)
            dyn_classes = dyn_meta.get("classes", None)
        else:
            dyn_model = dyn_meta
            dyn_classes = getattr(dyn_model, "classes_", None)
        print(f"[Dyn] Modell geladen: dynamic_gesture_rf.joblib (WIN={dyn_win})")
    else:
        print("[Dyn] Kein dynamisches Modell gefunden – Sequenzinferenz aus.")
except Exception as e:
    print("[Dyn] Laden fehlgeschlagen:", e)
    dyn_model = None

# Sequenzpuffer und temporales NMS für dynamische Gesten
seqbuf = SeqFeatureBuffer(win_len=dyn_win, stride=2)
nms = TemporalNMS(hold_frames=20)
MOTION_TH = 0.05  # Mindestsignal für Bewegungsaktivität (ggf. feinjustieren)


# =========================
# 2) Kamera-Setup (Intel RealSense)
# =========================
def setup_camera(serial: str):
    """
    Initialisiert eine RealSense-Pipeline für den Farbstream.

    :param serial: Geräteseriennummer
    :return: laufende rs.pipeline
    """
    p = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    p.start(cfg)
    return p


# =========================
# 3) Kamerafusion (Ensemble) per Wahrscheinlichkeit
# =========================
def fuse_gesture(probas, classes, min_cams=2):
    """
    Fusioniert Gestenwahrscheinlichkeiten über mehrere Kameras.

    - Nutzt Durchschnitt nur bei ausreichender Anzahl valider Kameras.
    - Fällt sonst auf stärkstes Einzelresultat zurück.

    :param probas: Liste von Probability-Vektoren (je Kamera)
    :param classes: Klassenlabels (Index entspricht Spaltenordnung in probas)
    :param min_cams: Mindestzahl valider Kameras für echte Fusion
    :return: vorhergesagtes Klassenlabel oder None
    """
    # Nur Vektoren behalten, in denen eine sinnvolle Konfidenz vorhanden ist
    valid = [p for p in probas if np.max(p) > 0]

    if not valid:
        return None

    if len(valid) < min_cams:
        # Zu wenige valide Kameras -> stärkstes Einzelresultat verwenden
        p = valid[0]
        return classes[int(np.argmax(p))]

    # Genügend valide Kameras -> Mittelwertbildung
    avg = np.mean(valid, axis=0)
    return classes[int(np.argmax(avg))]


# =========================
# 4) Worker-Thread pro Kamera
# =========================
def camera_worker(cam_id, serial, frame_queue):
    """
    Liest Frames einer Kamera, führt Detektion, Pose und Klassifikation aus
    und legt das Visualisierungsergebnis plus Wahrscheinlichkeiten in die Queue.

    :param cam_id: fortlaufende Kamera-ID
    :param serial: RealSense-Seriennummer
    :param frame_queue: Single-Item-Queue für (Frame, Proba)
    """
    frame_count = 0

    # Eigene Sequenzpuffer/NMS pro Kamera (entkoppelt von globalen Instanzen)
    seqbuf = SeqFeatureBuffer(win_len=dyn_win, stride=2)
    nms = TemporalNMS(hold_frames=20)
    MOTION_TH = 0.02  # Kameraspezifische Empfindlichkeit (0.01–0.05 sinnvoll)

    pipeline = setup_camera(serial)

    while True:
        frame_count += 1

        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())

        # --- Handdetektion (MMDetection) ---
        det = inference_detector(detector, img)
        inst = det.pred_instances
        scores = inst.scores.detach().cpu().numpy()
        labels = inst.labels.detach().cpu().numpy()
        bboxesAll = inst.bboxes.detach().cpu().numpy()

        # Schwellenwertanpassung je Kamera (empirisch bestimmt)
        if (serial == 'f1150579'):
            mask = (labels == 0) & (scores > 0.25)
        elif (serial == 'f1230805'):
            mask = (labels == 0) & (scores > 0.33)
        else:
            mask = (labels == 0) & (scores > 0.27)

        if not np.any(mask):
            # Keine Hand erkannt -> Nullvektor für Proba, Originalbild anzeigen
            proba = np.zeros(len(classes), dtype=float)
            vis = img.copy()
        else:
            # Beste Instanz wählen und Pose schätzen (Top-Down)
            sel_idx = np.argmax(scores * mask)
            bboxes = bboxesAll[sel_idx:sel_idx+1]

            pose = inference_topdown(pose_estimator, img, bboxes)
            kpts = pose[0].pred_instances.keypoints
            if kpts.ndim == 3:
                kpts = kpts[0]
            pts2d = kpts[:, :2]

            # --- Dynamische Gesten (Sequenzklassifikation) optional ---
            if dyn_model is not None:
                win = seqbuf.push_points(pts2d)
                if seqbuf.motion_energy > MOTION_TH and win is not None:
                    x = win.reshape(1, -1)
                    if hasattr(dyn_model, "predict_proba"):
                        proba_dyn = dyn_model.predict_proba(x)[0]
                        pred_id = int(np.argmax(proba_dyn))
                        conf_dyn = float(proba_dyn[pred_id])
                        if hasattr(dyn_model, "classes_"):
                            label_dyn = dyn_model.classes_[pred_id]
                        else:
                            label_dyn = dyn_classes[pred_id] if dyn_classes else str(pred_id)
                    else:
                        # Fallback ohne Probabilitäten
                        label_dyn = dyn_model.predict(x)[0]
                        conf_dyn = 1.0

                    fired = nms.step(label_dyn, conf_dyn, conf_th=0.60)
                    if fired is not None:
                        print(f"[Dyn][Cam {cam_id}] Geste erkannt: {fired} (conf≈{conf_dyn:.2f})")
                        # Optionales Overlay der dynamischen Geste
                        cv2.putText(img, f"DYN: {fired}", (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # --- Statische Gestenklassifikation ---
            center = pts2d.mean(axis=0)
            feat = (pts2d - center).flatten()[None, :]  # Translationsinvariante Features
            proba = clf.predict_proba(feat)[0]

            # --- Visualisierung ---
            vis = img.copy()
            cv2.rectangle(
                vis,
                tuple(bboxes[0, :2].astype(int)),
                tuple(bboxes[0, 2:].astype(int)),
                (0, 255, 0), 2
            )
            label = classes[int(np.argmax(proba))]
            cv2.putText(
                vis, f"{label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2
            )

        # Nur das neueste Ergebnis in der Queue halten (Backpressure vermeiden)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put((vis, proba))


# =========================
# 5) Anzeige-Loop und Ensemble-Entscheidung
# =========================
def display_loop(frame_queues):
    """
    Visualisiert pro Kamera die Resultate und zeigt ein Ensemble-Panel mit
    History (letzte N Labels) und Mehrheitsvotum.

    :param frame_queues: Liste von Queues (je Kamera)
    """
    # Fenster vorbereiten
    cv2.namedWindow("Ensemble", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ensemble", 600, 200)
    for i in range(len(frame_queues)):
        cv2.namedWindow(f"Camera {i}", cv2.WINDOW_NORMAL)

    # Historie letzter N Labels für Mehrheitsentscheid
    history = deque(maxlen=5)

    # Mindestanzahl valider Kameras: einfache Mehrheit
    min_cams = max(2, len(frame_queues) // 2 + 1)

    while True:
        probas = []
        for cam_id, q in enumerate(frame_queues):
            if not q.empty():
                frame, proba = q.get()
                cv2.imshow(f"Camera {cam_id}", frame)
                probas.append(proba)

        if probas:
            # Aktuelles kombiniertes Label (Fusionsausgabe)
            current_label = fuse_gesture(probas, classes, min_cams=min_cams)
            if current_label is not None:
                history.append(current_label)

            # Mehrheitsentscheid über die History
            vote_label = Counter(history).most_common(1)[0][0] if history else None

            # Ensemble-Canvas rendern
            ensemble_img = np.zeros((200, 1000, 3), dtype=np.uint8)

            # Historie (links)
            for idx, lbl in enumerate(history):
                y = 30 + idx * 30
                cv2.putText(
                    ensemble_img,
                    f"{idx+1}: {lbl}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

            # Mehrheitslabel (rechts unten, prominent)
            cv2.putText(
                ensemble_img,
                f"Voting: {vote_label}",
                (350, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3
            )

            cv2.imshow("Ensemble", ensemble_img)

        # ESC beendet Anzeige
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# =========================
# 6) Haupteinstieg: Geräte finden, Threads starten, Anzeige laufen lassen
# =========================
if __name__ == "__main__":
    # Mindestens eine RealSense-Kamera erforderlich
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 1:
        raise RuntimeError("Bitte mindestens zwei RealSense-Kameras anschließen.")
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]

    # CPU-Threading begrenzen, um Ressourcen zu stabilisieren
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True

    # Eine Queue pro Kamera (nur jeweils aktuelles Frame)
    frame_queues = [queue.Queue(maxsize=1) for _ in serials]

    # Kamera-Threads starten
    threads = []
    for idx, serial in enumerate(serials):
        t = threading.Thread(
            target=camera_worker,
            args=(idx, serial, frame_queues[idx]),
            daemon=True
        )
        t.start()
        threads.append(t)

    # Anzeige-/Fusionsloop
    display_loop(frame_queues)