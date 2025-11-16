from collections import deque
import numpy as np

def flatten_hand_points(pts2d: np.ndarray) -> np.ndarray:
    """
    pts2d: (N,2) ndarray in Pixeln. Gibt zentrierte, skalen-normierte, flache (2N,) zurück.
    Zentrierung: Abzug des Handmittelpunkts. Skala: Normierung mit mittlerer Distanz zum Mittelpunkt.
    """
    if pts2d is None or len(pts2d) == 0:
        return None
    center = pts2d.mean(axis=0)
    rel = pts2d - center
    scale = np.mean(np.linalg.norm(rel, axis=1)) + 1e-6
    norm = (rel / scale).reshape(-1)
    return norm.astype(np.float32)

class SeqFeatureBuffer:
    """
    Hält pro Frame Positions- & Geschwindigkeitsfeatures und bildet Sliding-Window.
    - features pro Frame: [pos, vel] wobei pos = flatten_hand_points, vel = pos-pos_prev
    - motion_energy: Mittelwert |vel|
    """
    def __init__(self, win_len: int = 32, stride: int = 2):
        self.win_len = int(win_len)
        self.stride = int(stride)
        self._buf = deque(maxlen=self.win_len)
        self._cnt = 0
        self._prev_pos = None
        self.motion_energy = 0.0

    def push_points(self, pts2d: np.ndarray):
        pos = flatten_hand_points(pts2d)
        if pos is None:
            return None
        if self._prev_pos is None:
            vel = np.zeros_like(pos)
        else:
            vel = pos - self._prev_pos
        self._prev_pos = pos
        feat_t = np.concatenate([pos, vel]).astype(np.float32)
        self._buf.append(feat_t)
        # Motion-Energie als gleitender Mittelwert |vel|
        v = np.abs(vel).mean()
        self.motion_energy = 0.9 * self.motion_energy + 0.1 * float(v)
        self._cnt += 1
        if len(self._buf) == self.win_len and (self._cnt % self.stride == 0):
            win = np.concatenate(list(self._buf), axis=0)
            return win  # shape (win_len * feat_dim,)
        return None

    def reset(self):
        self._buf.clear()
        self._cnt = 0
        self._prev_pos = None
        self.motion_energy = 0.0

class TemporalNMS:
    """Einfaches zeitliches Locking, damit eine Geste nicht mehrfach feuert."""
    def __init__(self, hold_frames: int = 15):
        self.hold = hold_frames
        self.cooldown = 0
        self.last_label = None

    def step(self, label: str, conf: float, conf_th: float = 0.6):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        if label is None or conf < conf_th:
            return None
        self.last_label = label
        self.cooldown = self.hold
        return label