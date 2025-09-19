import os
import numpy as np
import torch
import torch.nn as nn

# ---------- MediaPipe import (lazy) ----------
def _import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except Exception as e:
        raise RuntimeError("mediapipe is required. Install with `pip install mediapipe`.") from e

POSE_LM = 33
HAND_LM = 21

def _landmarks_to_xy(landmarks, count_expected):
    if landmarks is None:
        return np.zeros(count_expected*2, dtype=np.float32)
    pts = []
    for lm in landmarks.landmark:
        pts.extend([lm.x, lm.y])
    arr = np.array(pts, dtype=np.float32)
    if arr.size != count_expected*2:
        out = np.zeros(count_expected*2, dtype=np.float32)
        out[:arr.size] = arr
        return out
    return arr

def features_from_frame(results):
    pose_xy  = _landmarks_to_xy(getattr(results, "pose_landmarks", None), POSE_LM)
    lhand_xy = _landmarks_to_xy(getattr(results, "left_hand_landmarks", None), HAND_LM)
    rhand_xy = _landmarks_to_xy(getattr(results, "right_hand_landmarks", None), HAND_LM)
    feat = np.concatenate([pose_xy, lhand_xy, rhand_xy], axis=0)  # 150 dims
    mu, sigma = feat.mean(), feat.std()
    if sigma < 1e-6: sigma = 1.0
    feat = (feat - mu) / sigma
    return feat.astype(np.float32)

# ---------- Model ----------
class GRUClassifier(nn.Module):
    def __init__(self, in_dim=150, hid=256, num_layers=2, num_classes=10, dropout=0.3, bidir=True):
        super().__init__()
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers>1 else 0.0,
                          bidirectional=bidir)
        out_dim = hid * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x):
        out, _ = self.rnn(x)      # (B, T, H*dir)
        feat = out.mean(dim=1)    # mean over time
        logits = self.head(feat)  # (B, C)
        return logits