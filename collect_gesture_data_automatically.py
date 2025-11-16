

import json_tricks as json
import numpy as np
from datetime import datetime

import pyrealsense2 as rs
import cv2

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.utils import adapt_mmdet_pipeline


# 1) Modelle laden
detector = init_detector(
    'hand2dconfigdet.py',
    'hand2dmodeldet.pth',
    device='cuda'
)
detector.cfg = adapt_mmdet_pipeline(detector.cfg)

pose_estimator = init_pose_estimator(
    'hand2dconfigpose.py',
    'hand2dmodelpose.pth',
    device='cuda',
    cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
)

# 2) Label-Mapping
labels = {
    '1': "Daumen hoch",
    '2': "Peace",
    '3': "Faust",
    '4': "Offene Hand"
}

# 2.1) Eine Pose auswählen
print("Wähle eine Pose zum automatischen Labeln:")
for key, name in labels.items():
    print(f"  {key} → {name}")
choice = None
while choice not in labels:
    choice = input("Deine Wahl (1–4): ").strip()
selected_label = labels[choice]
print(f"Automatisches Labeln läuft für: {selected_label}")
print("Drücke 's' zum Speichern + Beenden, 'q' zum Abbrechen ohne Speichern.")

# 3) Alle angeschlossenen RealSense-Kameras finden
ctx = rs.context()
devices = ctx.query_devices()
serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
if not serials:
    raise RuntimeError("Keine RealSense-Kameras gefunden.")
print(f"Gefundene Kameras: {serials}")

# 4) Pipelines pro Kamera starten
pipelines = []
for serial in serials:
    p = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    p.start(cfg)
    pipelines.append(p)

# 5) Fenster anlegen
for idx in range(len(pipelines)):
    cv2.namedWindow(f"Cam {idx}", cv2.WINDOW_NORMAL)

data = []
should_save = True

try:
    while True:
        # Für jede Kamera: Frame, Detection, Pose, Feature, ggf. speichern
        for idx, p in enumerate(pipelines):
            frames = p.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            img = np.asanyarray(color.get_data())

            # a) Detection
            det = inference_detector(detector, img)
            inst = det.pred_instances.cpu().numpy()
            mask = (inst.labels == 0) & (inst.scores > 0.3)
            bboxes = inst.bboxes[mask]

            if len(bboxes) > 0:
                # b) Pose-Estimation Top-Down
                pose = inference_topdown(pose_estimator, img, bboxes)
                kpts = pose[0].pred_instances.keypoints
                if kpts.ndim == 3 and kpts.shape[0] == 1:
                    kpts = kpts[0]

                # c) Feature-Vektor berechnen
                pts2d = kpts[:, :2]
                center = pts2d.mean(axis=0)
                feat = (pts2d - center).flatten().tolist()

                # d) Sample speichern (pro Kamera einzeln)
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "camera_id": idx,
                    "features": feat,
                    "label": selected_label
                }
                data.append(entry)
                print(f"[Kamera {idx}] #{len(data)} gelabelt als: {selected_label}")

                # e) Bounding-Box & Keypoints visualisieren
                x1, y1, x2, y2 = bboxes[0].astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # man muss über entpacktes 
                for x, y in pts2d:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

            # f) Frame anzeigen
            cv2.imshow(f"Cam {idx}", img)

        # g) Tastenabfrage
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Speichern und Beenden
            break
        elif key == ord('q'):
            # Abbrechen ohne Speichern
            should_save = False
            break
        elif key == ord('p'):
            choice = None
            while choice not in labels:
                choice = input("Deine Wahl (1–4): ").strip()
            selected_label = labels[choice]


finally:
    # 6) Pipelines stoppen, Fenster schließen
    for p in pipelines:
        p.stop()
    cv2.destroyAllWindows()

    # 7) Speichern je nach Wunsch
    if should_save and data:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"gesture_data_multi_individual_{choice}_{ts}.json"
        with open(fname, "w") as f:
            json.dump(data, f)
        print(f"{len(data)} Samples gespeichert in {fname}")
    else:
        print("Abgebrochen ohne Speichern.")
