"""Fruit Ninja — slice fruits with your hands, avoid the bombs."""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import hailo
import math
import numpy as np
import random
import time
from collections import deque

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

# COCO 17 skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 6), (5, 11), (6, 12), (11, 12),    # torso
    (11, 13), (13, 15), (12, 14), (14, 16) # legs
]

# --- Game constants ---
GAME_DURATION = 60
LIVES = 3
FRUIT_SPAWN_CHANCE = 0.025
BOMB_SPAWN_CHANCE = 0.005
GRAVITY = 0.35
LAUNCH_VY_MIN = -20.0
LAUNCH_VY_MAX = -14.0
LAUNCH_VX_RANGE = 4.0
TRAIL_LEN = 20
SWIPE_MIN_DIST = 5
SLICE_RADIUS_MULT = 2.0
BOMB_TIMEOUT_FRAMES = 180
COMBO_WINDOW = 35
COMBO_BONUS = 5
PARTICLE_COUNT = 14

# Fruit definitions: (name, body_colour BGR, highlight, radius, points)
FRUIT_TYPES = [
    ("apple",      (40, 40, 215), (140, 140, 255), 34, 10),
    ("watermelon", (35, 168, 35), (130, 230, 130), 46, 15),
    ("orange",     (0, 138, 255), (110, 205, 255), 34, 10),
    ("lemon",      (0, 218, 230), (160, 255, 255), 28, 10),
    ("peach",      (95, 150, 255), (175, 210, 255), 32, 20),
    ("plum",       (148, 35, 128), (225, 120, 200), 28, 20),
]

# Blade colours per hand (BGR)
BLADE_COLORS = [(255, 210, 80), (55, 155, 255)]

# PiP constants
PIP_W = 240
PIP_H = 135
PIP_MARGIN = 12

# Colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_BG = (20, 5, 8)


def segment_hits_circle(p1, p2, cx, cy, r):
    """True if line segment p1→p2 intersects circle."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    fx, fy = p1[0] - cx, p1[1] - cy
    a = dx * dx + dy * dy
    if a < 1e-9:
        return math.hypot(fx, fy) <= r
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


def make_fruit(w, h):
    ftype = random.choice(FRUIT_TYPES)
    return {
        'x': float(random.randint(80, w - 80)),
        'y': float(h + ftype[3] + 10),
        'vx': random.uniform(-LAUNCH_VX_RANGE, LAUNCH_VX_RANGE),
        'vy': random.uniform(LAUNCH_VY_MIN, LAUNCH_VY_MAX),
        'type': ftype,
        'angle': random.uniform(0, math.pi * 2),
        'spin': random.uniform(-0.07, 0.07),
        'sliced': False,
    }


def make_bomb(w, h):
    return {
        'x': float(random.randint(100, w - 100)),
        'y': float(h + 35),
        'vx': random.uniform(-2.0, 2.0),
        'vy': random.uniform(LAUNCH_VY_MIN + 2, LAUNCH_VY_MAX + 2),
        'angle': 0.0,
        'spin': random.uniform(-0.04, 0.04),
        'fuse_t': 0,
        'sliced': False,
    }


def make_halves(fruit):
    _, color, highlight, radius, _ = fruit['type']
    cut = random.uniform(0, math.pi)
    halves = []
    for side in (0, 1):
        sign = -1 if side == 0 else 1
        halves.append({
            'x': float(fruit['x']),
            'y': float(fruit['y']),
            'vx': sign * random.uniform(2, 6),
            'vy': random.uniform(-5, -1),
            'color': color,
            'highlight': highlight,
            'radius': radius,
            'cut': cut,
            'angle': fruit['angle'],
            'spin': sign * random.uniform(0.06, 0.15),
            't': 55,
            'max_t': 55,
            'side': side,
        })
    return halves


def make_particles(x, y, color):
    parts = []
    for _ in range(PARTICLE_COUNT):
        a = random.uniform(0, math.pi * 2)
        s = random.uniform(3, 13)
        parts.append({
            'x': float(x), 'y': float(y),
            'vx': math.cos(a) * s, 'vy': math.sin(a) * s - 2,
            'color': color,
            't': random.randint(18, 42), 'max_t': 42,
            'r': random.randint(3, 9),
        })
    return parts


def make_popup(x, y, text, colour):
    return {'x': float(x), 'y': float(y), 'text': text,
            'colour': colour, 't': 52, 'max_t': 52}


# ─── Drawing helpers ────────────────────────────────────────────────────────
def draw_fruit(output, fruit):
    name, color, highlight, radius, _ = fruit['type']
    x, y = int(fruit['x']), int(fruit['y'])
    cv2.circle(output, (x, y), radius, color, -1)
    # Shine
    cv2.circle(output, (x - radius // 3, y - radius // 3), max(4, radius // 3), highlight, -1)
    # Outline
    cv2.circle(output, (x, y), radius, (50, 50, 50), max(1, radius // 8))
    # Type markings
    if name == "watermelon":
        for i in range(4):
            ang = fruit['angle'] + i * math.pi / 2
            p1 = (x + int(math.cos(ang) * radius * 0.25), y + int(math.sin(ang) * radius * 0.25))
            p2 = (x + int(math.cos(ang) * radius * 0.88), y + int(math.sin(ang) * radius * 0.88))
            cv2.line(output, p1, p2, (20, 110, 20), 2)
    elif name == "orange":
        for i in range(6):
            ang = fruit['angle'] + i * math.pi / 3
            p2 = (x + int(math.cos(ang) * radius * 0.88), y + int(math.sin(ang) * radius * 0.88))
            cv2.line(output, (x, y), p2, (0, 95, 190), 1)


def draw_half(output, half):
    ratio = half['t'] / half['max_t']
    color = tuple(int(c * ratio) for c in half['color'])
    radius = half['radius']
    x, y = int(half['x']), int(half['y'])
    sign = -1 if half['side'] == 0 else 1
    cut = half['cut']
    ox = int(math.cos(cut + math.pi / 2) * sign * radius * 0.2)
    oy = int(math.sin(cut + math.pi / 2) * sign * radius * 0.2)
    cv2.circle(output, (x + ox, y + oy), radius, color, -1)
    # Cut line
    p1 = (x + int(math.cos(cut) * radius), y + int(math.sin(cut) * radius))
    p2 = (x - int(math.cos(cut) * radius), y - int(math.sin(cut) * radius))
    cv2.line(output, p1, p2, (200, 200, 255), 3)


def draw_bomb(output, bomb):
    x, y = int(bomb['x']), int(bomb['y'])
    cv2.circle(output, (x, y), 28, (30, 30, 30), -1)
    cv2.circle(output, (x, y), 28, (70, 70, 70), 2)
    cv2.circle(output, (x - 8, y - 8), 7, (80, 80, 80), -1)
    # Fuse
    ft = bomb['fuse_t']
    flen = 12 + int(math.sin(ft * 0.3) * 4)
    fa = bomb['angle'] + 1.2
    fx = x + int(math.cos(fa) * 28)
    fy = y + int(math.sin(fa) * 28)
    cv2.line(output, (fx, fy), (fx, fy - flen), (40, 120, 160), 3)
    if (ft // 4) % 2 == 0:
        cv2.circle(output, (fx, fy - flen), 4, (50, 200, 255), -1)


def draw_blade(output, trail, color):
    n = len(trail)
    if n < 2:
        return
    for i in range(n - 1):
        ratio = (i + 1) / n
        r = min(255, int(color[0] * ratio + 255 * (1 - ratio) * 0.25))
        g = min(255, int(color[1] * ratio + 200 * (1 - ratio) * 0.25))
        b = min(255, int(color[2] * ratio + 255 * (1 - ratio) * 0.25))
        w = max(2, int(8 * ratio))
        p1 = (int(trail[i][0]), int(trail[i][1]))
        p2 = (int(trail[i + 1][0]), int(trail[i + 1][1]))
        cv2.line(output, p1, p2, (b, g, r), w)
    tip = trail[-1]
    cv2.circle(output, (int(tip[0]), int(tip[1])), 9, color, -1)


def draw_particles(output, particles):
    for p in particles:
        ratio = p['t'] / p['max_t']
        color = tuple(int(c * ratio) for c in p['color'])
        radius = max(1, int(p['r'] * ratio))
        cv2.circle(output, (int(p['x']), int(p['y'])), radius, color, -1)


def draw_popups(output, popups):
    for p in popups:
        ratio = p['t'] / p['max_t']
        color = tuple(min(255, int(c * ratio)) for c in p['colour'])
        cv2.putText(output, p['text'], (int(p['x']) - 30, int(p['y'])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def draw_hud(output, score, lives_left, combo_count, t_left, w):
    cv2.putText(output, f"SCORE {score}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    if combo_count >= 2:
        label = f"COMBO x{combo_count}!"
        col = (50, 220, 255) if combo_count < 4 else (50, 100, 255)
        cv2.putText(output, label, (w // 2 - 60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    # Lives
    for i in range(LIVES):
        color = (40, 40, 215) if i < lives_left else (75, 55, 55)
        cx = w - 38 - i * 44
        cv2.circle(output, (cx, 34), 15, color, -1)
        cv2.circle(output, (cx, 34), 15, WHITE, 1)

    # Timer
    timer_col = (50, 50, 255) if t_left <= 10 else WHITE
    cv2.putText(output, f"TIME {t_left:02d}", (w - 160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_col, 2)


def draw_skeleton(output, wrist_trails, w, h):
    """Draw blade trails for each hand."""
    for wi, trail in enumerate(wrist_trails):
        if len(trail) >= 2:
            draw_blade(output, trail, BLADE_COLORS[wi % len(BLADE_COLORS)])


# ─── Callback class ─────────────────────────────────────────────────────────
class FruitNinjaCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True
        self.wrist_queues = [deque(maxlen=TRAIL_LEN) for _ in range(2)]
        self.pip_frame = None
        self.skeleton_kps = []  # normalized keypoints for PiP skeleton

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

    # Extract wrist positions
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    wrists = []
    for detection in detections:
        if detection.get_label() != "person":
            continue
        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        pts = landmarks[0].get_points()
        if len(pts) > RIGHT_WRIST:
            for pt_idx in (LEFT_WRIST, RIGHT_WRIST):
                pt = pts[pt_idx]
                xn = pt.x() * bbox.width() + bbox.xmin()
                yn = pt.y() * bbox.height() + bbox.ymin()
                xpx = int(np.clip(xn * width, 0, width - 1))
                ypx = int(np.clip(yn * height, 0, height - 1))
                wrists.append((xpx, ypx))
        break

    # Extract all 17 keypoints for skeleton overlay on PiP
    user_data.skeleton_kps = []
    for detection in detections:
        if detection.get_label() != "person":
            continue
        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        pts = landmarks[0].get_points()
        user_data.skeleton_kps = [
            (pt.x() * bbox.width() + bbox.xmin(),
             pt.y() * bbox.height() + bbox.ymin())
            for pt in pts
        ]
        break

    # Update wrist trails
    for wi in range(min(2, len(wrists))):
        user_data.wrist_queues[wi].append(wrists[wi])

    # Get/init game state
    now = time.time()
    if not hasattr(user_data, 'game_start'):
        user_data.game_start = now
        user_data.score = 0
        user_data.lives = LIVES
        user_data.fruits = []
        user_data.bombs = []
        user_data.halves = []
        user_data.particles = []
        user_data.popups = []
        user_data.combo_count = 0
        user_data.combo_timer = 0
        user_data.bomb_timeout = 0

    remaining = max(0, int(GAME_DURATION - (now - user_data.game_start)))

    if remaining <= 0:
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[:] = DARK_BG
        cv2.putText(output, "TIME'S UP!", (width // 2 - 140, height // 3), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (50, 210, 255), 4)
        cv2.putText(output, f"SCORE: {user_data.score}", (width // 2 - 120, height // 3 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        cv2.putText(output, "Restarting in 5s...", (width // 2 - 140, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 210, 80), 2)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        user_data.set_frame(output)
        return Gst.FlowReturn.OK

    # Spawn
    if random.random() < FRUIT_SPAWN_CHANCE:
        user_data.fruits.append(make_fruit(width, height))
    if random.random() < BOMB_SPAWN_CHANCE:
        user_data.bombs.append(make_bomb(width, height))

    # Physics
    for fruit in user_data.fruits[:]:
        fruit['x'] += fruit['vx']
        fruit['y'] += fruit['vy']
        fruit['vy'] += GRAVITY
        fruit['angle'] += fruit['spin']
        if fruit['y'] > height + fruit['type'][3] + 20:
            user_data.fruits.remove(fruit)
            if not fruit['sliced']:
                user_data.lives -= 1

    for bomb in user_data.bombs[:]:
        bomb['x'] += bomb['vx']
        bomb['y'] += bomb['vy']
        bomb['vy'] += GRAVITY
        bomb['angle'] += bomb['spin']
        bomb['fuse_t'] += 1
        if bomb['y'] > height + 40:
            user_data.bombs.remove(bomb)

    for h in user_data.halves[:]:
        h['x'] += h['vx']
        h['y'] += h['vy']
        h['vy'] += GRAVITY * 0.4
        h['angle'] += h['spin']
        h['t'] -= 1
        if h['t'] <= 0 or h['y'] > height + 60:
            user_data.halves.remove(h)

    for p in user_data.particles[:]:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['vy'] += GRAVITY * 0.25
        p['t'] -= 1
        if p['t'] <= 0:
            user_data.particles.remove(p)

    for pop in user_data.popups[:]:
        pop['y'] -= 1.6
        pop['t'] -= 1
        if pop['t'] <= 0:
            user_data.popups.remove(pop)

    if user_data.combo_timer > 0:
        user_data.combo_timer -= 1
    else:
        user_data.combo_count = 0

    if user_data.bomb_timeout > 0:
        user_data.bomb_timeout -= 1

    # Slice detection
    for wi in range(2):
        trail = list(user_data.wrist_queues[wi])
        if len(trail) < 2:
            continue
        p1, p2 = trail[-2], trail[-1]
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < SWIPE_MIN_DIST:
            continue
        if user_data.bomb_timeout > 0:
            continue

        blade_col = BLADE_COLORS[wi % len(BLADE_COLORS)]

        for fruit in user_data.fruits[:]:
            if fruit['sliced']:
                continue
            r = fruit['type'][3] * SLICE_RADIUS_MULT
            if segment_hits_circle(p1, p2, fruit['x'], fruit['y'], r):
                fruit['sliced'] = True
                user_data.combo_timer = COMBO_WINDOW
                user_data.combo_count += 1
                pts = fruit['type'][4] + COMBO_BONUS * max(0, user_data.combo_count - 1)
                user_data.score += pts
                user_data.halves += make_halves(fruit)
                user_data.particles += make_particles(fruit['x'], fruit['y'], fruit['type'][1])
                label = f"+{pts}" if user_data.combo_count < 2 else f"COMBO x{user_data.combo_count}! +{pts}"
                user_data.popups.append(make_popup(fruit['x'], fruit['y'] - 22, label, blade_col))

        for bomb in user_data.bombs[:]:
            if bomb['sliced']:
                continue
            if segment_hits_circle(p1, p2, bomb['x'], bomb['y'], 28 * SLICE_RADIUS_MULT):
                bomb['sliced'] = True
                user_data.bombs.remove(bomb)
                user_data.particles += make_particles(bomb['x'], bomb['y'], (30, 100, 200))
                user_data.bomb_timeout = BOMB_TIMEOUT_FRAMES
                user_data.popups.append(make_popup(bomb['x'], bomb['y'] - 22, "TIMEOUT!", (30, 80, 255)))
                break

    # --- Render ---
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[:] = DARK_BG

    for fruit in user_data.fruits:
        if not fruit['sliced']:
            draw_fruit(output, fruit)

    for bomb in user_data.bombs:
        if not bomb['sliced']:
            draw_bomb(output, bomb)

    for h in user_data.halves:
        draw_half(output, h)

    draw_particles(output, user_data.particles)

    # Blade trails
    for wi in range(2):
        trail = list(user_data.wrist_queues[wi])
        if len(trail) >= 2:
            draw_blade(output, trail, BLADE_COLORS[wi % len(BLADE_COLORS)])

    draw_popups(output, user_data.popups)
    draw_hud(output, user_data.score, user_data.lives, user_data.combo_count, remaining, width)

    # Bomb timeout indicator
    if user_data.bomb_timeout > 0:
        secs_left = math.ceil(user_data.bomb_timeout / 60)
        cv2.putText(output, f"FROZEN {secs_left}s", (width // 2 - 60, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 80, 255), 3)

    # PiP camera preview with skeleton
    if user_data.pip_frame is not None:
        ph, pw = user_data.pip_frame.shape[:2]
        if pw > 0 and ph > 0:
            scale = min(PIP_W / pw, PIP_H / ph)
            new_w, new_h = int(pw * scale), int(ph * scale)
            resized = cv2.resize(user_data.pip_frame, (new_w, new_h))
            # Draw skeleton on PiP
            if user_data.skeleton_kps:
                for a, b in COCO_SKELETON:
                    if a < len(user_data.skeleton_kps) and b < len(user_data.skeleton_kps):
                        ax = int(user_data.skeleton_kps[a][0] * new_w)
                        ay = int(user_data.skeleton_kps[a][1] * new_h)
                        bx = int(user_data.skeleton_kps[b][0] * new_w)
                        by = int(user_data.skeleton_kps[b][1] * new_h)
                        cv2.line(resized, (ax, ay), (bx, by), (0, 255, 0), 1)
                for i, (xn, yn) in enumerate(user_data.skeleton_kps):
                    kx, ky = int(xn * new_w), int(yn * new_h)
                    r = 3 if i in (9, 10) else 2
                    col = (0, 255, 255) if i in (9, 10) else (0, 200, 0)
                    cv2.circle(resized, (kx, ky), r, col, -1)
            x0 = width - new_w - PIP_MARGIN
            y0 = height - new_h - PIP_MARGIN
            cv2.rectangle(output, (x0 - 2, y0 - 2), (x0 + new_w + 2, y0 + new_h + 2), (200, 200, 200), 2)
            cv2.putText(output, "CAM", (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            output[y0:y0 + new_h, x0:x0 + new_w] = resized

    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    user_data.set_frame(output)

    return Gst.FlowReturn.OK


class FruitNinjaApp(GStreamerPoseEstimationApp):
    def __init__(self, app_callback, user_data, parser=None):
        super().__init__(app_callback, user_data, parser)
        self.options_menu.use_frame = True
        user_data.use_frame = True


def main():
    parser = get_pipeline_parser()
    args, _ = parser.parse_known_args()

    user_data = FruitNinjaCallback()
    app = FruitNinjaApp(app_callback, user_data, parser)
    app.run()


if __name__ == "__main__":
    main()
