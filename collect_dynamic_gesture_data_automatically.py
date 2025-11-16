import json_tricks as json
import warnings

# Ignoriere Fehlernachricht 
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.autocast.*"
)

import numpy as np
from datetime import datetime
import pyrealsense2 as rs
import cv2
import os

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
from dynamic_gestures import flatten_hand_points

# Modelle laden
detector = init_detector('hand2dconfigdet.py', 'hand2dmodeldet.pth', device='cuda')
detector.cfg = adapt_mmdet_pipeline(detector.cfg)

pose_estimator = init_pose_estimator(
    'hand2dconfigpose.py',
    'hand2dmodelpose.pth',
    device='cuda',
    cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
)

# Label-Mapping
labels = {
    '1': "Winken",
    '2': "Squeezer",
    '3': "Tippen",
    '4': "Swipe",
    '5': "Nothing",
    '6': "Wild"
}

print("Wähle eine dynamische Geste:")
for key, name in labels.items():
    print(f"  {key} → {name}")
choice = None
while choice not in labels:
    choice = input("Deine Wahl (1–5): ").strip()
selected_label = labels[choice]
print(f"Automatisches Aufnehmen läuft für: {selected_label}")
print("Drücke 's' zum Speichern + Beenden, 'q' zum Abbrechen ohne Speichern, 'p' zum Label wechseln.")

# RealSense-Kameras finden
ctx = rs.context()
devices = ctx.query_devices()
serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
if not serials:
    raise RuntimeError("Keine RealSense-Kameras gefunden.")
print(f"Gefundene Kameras: {serials}")

# Pipelines starten
pipelines = []
for serial in serials:
    p = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    p.start(cfg)
    pipelines.append(p)

# Fenster anlegen
for idx in range(len(pipelines)):
    cv2.namedWindow(f"Cam {idx}", cv2.WINDOW_NORMAL)

# Sequenzpuffer pro Kamera, recording false für manuellen start
recording = False
buffers = [[] for _ in pipelines]
prev_pos = [None for _ in pipelines]
samples = []
SEQ_LEN = 30
should_save = True

try:
    while True:
        for idx, p in enumerate(pipelines):
            frames = p.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())


            # Detection
            det = inference_detector(detector, img)
            inst = det.pred_instances.cpu().numpy()
            if(idx==0):
                mask = (inst.labels == 0) & (inst.scores > 0.27)
            elif(idx==2):
                mask = (inst.labels == 0) & (inst.scores > 0.24)
            else:
                mask = (inst.labels == 0) & (inst.scores > 0.27)
            bboxes = inst.bboxes[mask]

            if len(bboxes) > 0:
                pose = inference_topdown(pose_estimator, img, bboxes)
                kpts = pose[0].pred_instances.keypoints
                if kpts.ndim == 3 and kpts.shape[0] == 1:
                    kpts = kpts[0]
                pts2d = kpts[:, :2]

                # Feature berechnen
                pos = flatten_hand_points(pts2d)
                if pos is not None and recording:
                    vel = np.zeros_like(pos) if prev_pos[idx] is None else pos - prev_pos[idx]
                    prev_pos[idx] = pos
                    feat_t = np.concatenate([pos, vel]).astype(float).tolist()
                    buffers[idx].append(feat_t)

                    # Wenn Sequenz voll -> speichern
                    if len(buffers[idx]) >= SEQ_LEN:
                        entry = {
                            "timestamp": datetime.now().isoformat(),
                            "camera_id": idx,
                            "label": selected_label,
                            "features": buffers[idx]
                        }
                        samples.append(entry)
                        print(f"[Kamera {idx}] Sample #{len(samples)} gespeichert")
                        buffers[idx] = []

                # Visualisierung
                x1, y1, x2, y2 = bboxes[0].astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for x, y in pts2d:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

            cv2.putText(img, f"Samples: {len(samples)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow(f"Cam {idx}", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            recording = True
            buffers = [[] for _ in pipelines]
            prev_pos = [None for _ in pipelines]
            print("[Dyn] Aufnahme gestartet …")
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("dyn_data", exist_ok=True)
            fname = os.path.join("dyn_data", f"{selected_label}_{ts}.jsonl")
            with open(fname, "a", encoding="utf-8") as f:
                for entry in samples:
                    json.dump(entry, f)
                    f.write("\n")
            print(f"{len(samples)} Sequenzen gespeichert in {fname}")
            break
        elif key == ord('q'):
            should_save = False
            print("Abbruch ohne Speichern.")
            break
        elif key == ord('p'):
            if recording:
                recording = False
            else:
                recording = True
        elif key == ord('c'):
            choice = None
            while choice not in labels:
                choice = input("Neue Wahl (1–6): ").strip()
            selected_label = labels[choice]
            print(f"Neues Label: {selected_label}")


finally:
    for p in pipelines:
        p.stop()
    cv2.destroyAllWindows()
