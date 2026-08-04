"""Dance Dance Revolution — match your limbs to the falling notes."""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import hailo
import numpy as np
import random
import time

from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.common.parser import get_pipeline_parser
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)

logger = get_logger(__name__)

# --- Keypoint indices ---
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# --- Limb definitions: (name, short, primary_kp, colour BGR) ---
LIMBS = [
    ("Left Arm", "LA", LEFT_WRIST, (60, 60, 220)),
    ("Right Arm", "RA", RIGHT_WRIST, (230, 100, 60)),
    ("Left Leg", "LL", LEFT_KNEE, (70, 200, 50)),
    ("Right Leg", "RL", RIGHT_KNEE, (0, 200, 230)),
]

_KP_PAIRS = {
    0: (LEFT_WRIST, LEFT_ELBOW),
    1: (RIGHT_WRIST, RIGHT_ELBOW),
    2: (LEFT_KNEE, LEFT_ANKLE),
    3: (RIGHT_KNEE, RIGHT_ANKLE),
}

LIMB_COLORS = [l[3] for l in LIMBS]
LIMB_SHORTS = [l[1] for l in LIMBS]
NUM_COLUMNS = 4

# --- PiP constants ---
PIP_W = 240
PIP_H = 135
PIP_MARGIN = 12

# --- Game constants ---
GAME_DURATION = 90          # seconds
NODE_BASE_SPEED = 4.0
NODE_SPEED_INCR = 0.3
SPAWN_INTERVAL = 80
SPAWN_MIN = 28
LEVEL_SCORE_STEP = 400
PERFECT_WINDOW = 28
GOOD_WINDOW = 60
POINTS_PERFECT = 100
POINTS_GOOD = 50
COMBO_BONUS = 10
HIT_LINE_FRAC = 0.80

# Node drawing
NODE_W = 90
NODE_H = 44

# Colours (BGR)
DARK_BG = (30, 10, 12)
COL_BG = [(42, 16, 18), (50, 20, 22), (42, 16, 18), (50, 20, 22)]
HIT_COL = (160, 255, 255)
WHITE = (255, 255, 255)
GREY = (110, 90, 90)


class Node:
    def __init__(self, col, limb_idx, speed):
        self.col = col
        self.limb_idx = limb_idx
        self.color = LIMB_COLORS[limb_idx]
        self.speed = speed
        self.y = float(-NODE_H)
        self.state = 'falling'
        self.label = LIMB_SHORTS[limb_idx]
        self.alpha = 255

    def update(self):
        if self.state == 'falling':
            self.y += self.speed
        else:
            self.alpha = max(0, self.alpha - 14)

    @property
    def center_y(self):
        return self.y + NODE_H / 2

    def is_dead(self):
        return self.state != 'falling' and self.alpha == 0


class Popup:
    def __init__(self, text, x, y, colour):
        self.text = text
        self.x = x
        self.y = y
        self.colour = colour
        self.spawn = time.time()

    def alive(self):
        return (time.time() - self.spawn) < 1.0

    def alpha(self):
        age = time.time() - self.spawn
        return max(0.0, 1.0 - age / 1.0)


class DDRCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True
        self.limb_x = [None] * len(LIMBS)
        self.pip_frame = None  # raw camera frame for PiP

    def set_frame(self, frame):
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break
        try:
            self.frame_queue.put_nowait(frame)
        except Exception:
            pass


def app_callback(element, buffer, user_data):
    pad = element.get_static_pad("src")
    fmt, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and fmt and width and height:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    if frame is None:
        return Gst.FlowReturn.OK

    # Save raw frame for PiP before any rendering
    user_data.pip_frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 1)

    # Extract limb positions
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    limb_x = [None] * len(LIMBS)
    for detection in detections:
        if detection.get_label() != "person":
            continue
        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        pts = landmarks[0].get_points()

        for li in range(len(LIMBS)):
            xs = []
            for kp in _KP_PAIRS[li]:
                if kp < len(pts):
                    x = pts[kp].x() * bbox.width() + bbox.xmin()
                    xs.append(x)
            if xs:
                limb_x[li] = sum(xs) / len(xs)
        break

    user_data.limb_x = limb_x

    # Get/init game state
    now = time.time()
    if not hasattr(user_data, 'game_start'):
        user_data.game_start = now
        user_data.score = 0
        user_data.combo = 0
        user_data.nodes = []
        user_data.popups = []
        user_data.spawn_tmr = 0
        user_data.level = 1
        user_data.last_spawn = now

    elapsed = now - user_data.game_start
    remaining = max(0.0, GAME_DURATION - elapsed)

    if remaining <= 0:
        # Game over — draw final screen
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[:] = DARK_BG
        cv2.putText(output, "GAME OVER", (width // 2 - 140, height // 3), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (70, 70, 255), 4)
        cv2.putText(output, f"SCORE: {user_data.score}", (width // 2 - 120, height // 3 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        cv2.putText(output, "Restarting in 5s...", (width // 2 - 140, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 180), 2)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        user_data.set_frame(output)
        return Gst.FlowReturn.OK

    # Level and spawn
    user_data.level = 1 + user_data.score // LEVEL_SCORE_STEP
    interval = max(SPAWN_MIN, SPAWN_INTERVAL - (user_data.level - 1) * 5)

    if now - user_data.last_spawn >= interval / 60.0:  # approximate seconds
        user_data.last_spawn = now
        col = random.randint(0, NUM_COLUMNS - 1)
        limb = random.randint(0, len(LIMBS) - 1)
        speed = NODE_BASE_SPEED + NODE_SPEED_INCR * (user_data.level - 1)
        user_data.nodes.append(Node(col, limb, speed))

    # Update nodes
    for nd in user_data.nodes:
        nd.update()

    # Hit detection
    col_w = width // NUM_COLUMNS
    for nd in user_data.nodes:
        if nd.state != 'falling':
            continue
        dist = abs(nd.center_y - HIT_LINE_FRAC * height)
        in_col = (limb_x[nd.limb_idx] is not None and
                  int(limb_x[nd.limb_idx] * NUM_COLUMNS) == nd.col)

        if dist <= PERFECT_WINDOW and in_col:
            nd.state = 'hit'
            pts = POINTS_PERFECT + user_data.combo * COMBO_BONUS
            user_data.score += pts
            user_data.combo += 1
            cx = nd.col * col_w + col_w // 2
            user_data.popups.append(Popup("PERFECT!", cx, HIT_LINE_FRAC * height - 70, (80, 255, 255)))

        elif dist <= GOOD_WINDOW and in_col:
            nd.state = 'hit'
            pts = POINTS_GOOD + user_data.combo * COMBO_BONUS
            user_data.score += pts
            user_data.combo += 1
            cx = nd.col * col_w + col_w // 2
            user_data.popups.append(Popup("GOOD", cx, HIT_LINE_FRAC * height - 70, (100, 230, 100)))

        elif nd.center_y > HIT_LINE_FRAC * height + GOOD_WINDOW:
            nd.state = 'missed'
            user_data.combo = 0
            cx = nd.col * col_w + col_w // 2
            user_data.popups.append(Popup("MISS", cx, HIT_LINE_FRAC * height - 70, (70, 70, 230)))

    # Cleanup
    user_data.nodes = [nd for nd in user_data.nodes if not nd.is_dead() and nd.y < height + 80]
    user_data.popups = [p for p in user_data.popups if p.alive()]

    # --- Render ---
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[:] = DARK_BG

    # Column backgrounds
    for ci in range(NUM_COLUMNS):
        x0 = ci * col_w
        color = COL_BG[ci % len(COL_BG)]
        cv2.rectangle(output, (x0, 0), (x0 + col_w, height), color, -1)

    # Column dividers
    for ci in range(1, NUM_COLUMNS):
        cv2.line(output, (ci * col_w, 0), (ci * col_w, height), (72, 42, 45), 2)

    # Limb position guides
    for li, xn in enumerate(limb_x):
        if xn is not None:
            lx = int(xn * width)
            col = LIMB_COLORS[li]
            cv2.line(output, (lx, 0), (lx, height), col, 1)
            # Arrow above hit line
            tip = (lx, int(HIT_LINE_FRAC * height) - 6)
            left = (lx - 7, int(HIT_LINE_FRAC * height) - 20)
            right = (lx + 7, int(HIT_LINE_FRAC * height) - 20)
            cv2.fillPoly(output, [np.array([tip, left, right], np.int32)], col)

    # Nodes
    for nd in user_data.nodes:
        x0 = nd.col * col_w + (col_w - NODE_W) // 2
        y0 = int(nd.y)
        alpha = nd.alpha / 255.0
        color = tuple(int(c * alpha) for c in nd.color)
        cv2.rectangle(output, (x0, y0), (x0 + NODE_W, y0 + NODE_H), color, -1)
        cv2.rectangle(output, (x0, y0), (x0 + NODE_W, y0 + NODE_H), WHITE, 2)
        # Label
        text_size = cv2.getTextSize(nd.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = x0 + (NODE_W - text_size[0]) // 2
        ty = y0 + (NODE_H + text_size[1]) // 2
        cv2.putText(output, nd.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    # Hit line
    hit_y = int(HIT_LINE_FRAC * height)
    cv2.line(output, (0, hit_y), (width, hit_y), HIT_COL, 3)

    # Target rings
    for ci in range(NUM_COLUMNS):
        ring_cx = ci * col_w + col_w // 2
        cv2.circle(output, (ring_cx, hit_y), 34, (85, 50, 55), -1)
        cv2.circle(output, (ring_cx, hit_y), 34, (130, 85, 90), 2)

    # Detection zone — show limb positions
    zone_cy = hit_y + (height - hit_y) // 2
    for ci in range(NUM_COLUMNS):
        in_here = [li for li in range(len(LIMBS)) if limb_x[li] is not None and int(limb_x[li] * NUM_COLUMNS) == ci]
        if in_here:
            spacing = col_w // (len(in_here) + 1)
            for slot, li in enumerate(in_here):
                ix = ci * col_w + spacing * (slot + 1)
                cv2.circle(output, (ix, zone_cy), 18, LIMB_COLORS[li], -1)
                cv2.circle(output, (ix, zone_cy), 18, WHITE, 2)
                cv2.putText(output, LIMB_SHORTS[li], (ix - 8, zone_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        else:
            ring_cx = ci * col_w + col_w // 2
            cv2.circle(output, (ring_cx, zone_cy), 18, (65, 38, 40), -1)
            cv2.circle(output, (ring_cx, zone_cy), 18, (90, 58, 60), 2)

    # Popups
    for p in user_data.popups:
        alpha = p.alpha()
        color = tuple(int(c * alpha) for c in p.colour)
        cv2.putText(output, p.text, (int(p.x) - 30, int(p.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # HUD
    cv2.putText(output, f"Score: {user_data.score}", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(output, f"Combo x{user_data.combo}", (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 215, 255), 2)
    cv2.putText(output, f"Lv {user_data.level}", (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)

    # Timer
    secs = int(remaining)
    timer_col = (80, 80, 255) if remaining < 10 else WHITE
    cv2.putText(output, f"{secs:02d}", (width - 60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_col, 2)

    # PiP camera preview
    if user_data.pip_frame is not None:
        ph, pw = user_data.pip_frame.shape[:2]
        if pw > 0 and ph > 0:
            scale = min(PIP_W / pw, PIP_H / ph)
            new_w, new_h = int(pw * scale), int(ph * scale)
            resized = cv2.resize(user_data.pip_frame, (new_w, new_h))
            x0 = width - new_w - PIP_MARGIN
            y0 = height - new_h - PIP_MARGIN
            cv2.rectangle(output, (x0 - 2, y0 - 2), (x0 + new_w + 2, y0 + new_h + 2), (200, 200, 200), 2)
            cv2.putText(output, "CAM", (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            output[y0:y0 + new_h, x0:x0 + new_w] = resized

    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    user_data.set_frame(output)

    return Gst.FlowReturn.OK


class DDRApp(GStreamerPoseEstimationApp):
    def __init__(self, app_callback, user_data, parser=None):
        super().__init__(app_callback, user_data, parser)
        self.options_menu.use_frame = True
        user_data.use_frame = True


def main():
    parser = get_pipeline_parser()
    args, _ = parser.parse_known_args()

    user_data = DDRCallback()
    app = DDRApp(app_callback, user_data, parser)
    app.run()


if __name__ == "__main__":
    main()
