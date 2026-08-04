"""Soccer — penalty kick game. Kick the ball with your foot into the goal."""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import hailo
import math
import numpy as np
import time
import queue
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

# --- Keypoint indices (COCO 17) ---
NOSE = 0
LEFT_HIP = 11; RIGHT_HIP = 12
LEFT_KNEE = 13; RIGHT_KNEE = 14
LEFT_ANKLE = 15; RIGHT_ANKLE = 16
ANKLE_INDICES = [LEFT_ANKLE, RIGHT_ANKLE]
LEG_KEYPOINTS = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]

# COCO 17 skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 6), (5, 11), (6, 12), (11, 12),    # torso
    (11, 13), (13, 15), (12, 14), (14, 16) # legs
]

# --- Game constants ---
GAME_DURATION = 90          # seconds
KICK_SCREEN_R = 85          # px proximity to ball
KICK_VEL_THR = 4.0          # ankle speed threshold (px/frame)
KICK_COOLDOWN_FRAMES = 65   # frames between kicks
STATE_HOLD = 140            # frames to show result

# 3D projection constants
CAM_H = 4.0
FOCAL = 600.0
NEAR_Z = 0.5
GOAL_Z = 40.0
GOAL_HW = 3.5
GOAL_H = 3.5
PENALTY_Z = 8.0
BALL_R_W = 0.5
GRAVITY_W = 0.004
BOUNCE_DAMP = 0.45
FRICTION_W = 0.995
KICK_VZ_BASE = 1.0
KICK_VZ_BONUS = 0.03
KICK_LAT = 0.01
KICK_LIFT = 0.012
FIELD_HALF_W = 18.0
FIELD_Z_FAR = 55.0

GK_HW = 0.45
GK_H = 3.0
GK_SPEED = 0.10
GK_REACT_Z = 25.0

# States
S_WAITING = 0
S_FLIGHT = 1
S_GOAL = 2
S_BLOCKED = 3
S_MISS = 4

# Colours (BGR for OpenCV)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
F_GREEN = (50, 125, 46)
F_STRIPE = (60, 142, 56)
GOAL_COL = (235, 235, 235)
NET_COL = (180, 180, 180)
GK_COL = (0, 140, 255)
YELLOW = (0, 230, 255)
CYAN = (220, 210, 0)
RED = (30, 40, 220)


def project(wx, wy, wz, screen_w, screen_h):
    """Project world (wx, wy, wz) to screen (sx, sy, scale)."""
    if wz < NEAR_Z:
        return None
    s = FOCAL / wz
    screen_cx = screen_w // 2
    screen_cy = int(0.85 * screen_h - CAM_H * FOCAL / PENALTY_Z)
    sx = int(screen_cx + wx * s)
    sy = int(screen_cy + (CAM_H - wy) * s)
    return sx, sy, s


def proj_ground(wx, wz, screen_w, screen_h):
    return project(wx, 0.0, wz, screen_w, screen_h)


# ─── Game objects ────────────────────────────────────────────────────────────
class Ball:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = float(PENALTY_Z)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.rot = 0.0

    def reset(self):
        self.__init__()


class Goalkeeper:
    def __init__(self):
        self.x = 0.0

    def update(self, ball, state):
        if state == S_FLIGHT and ball.z > GOAL_Z - GK_REACT_Z:
            target = np.clip(ball.x, -(GOAL_HW - GK_HW), GOAL_HW - GK_HW)
            dx = target - self.x
            self.x += float(np.clip(dx, -GK_SPEED, GK_SPEED))
        elif state != S_FLIGHT:
            self.x += (0.0 - self.x) * 0.04


class Popup:
    def __init__(self, text, x, y, colour, duration=1.5):
        self.text = text
        self.x = x
        self.y = y
        self.colour = colour
        self.spawn = time.time()
        self.duration = duration

    def alive(self):
        return (time.time() - self.spawn) < self.duration

    def alpha(self):
        age = time.time() - self.spawn
        return max(0.0, 1.0 - age / self.duration)


# ─── Drawing helpers ────────────────────────────────────────────────────────
def draw_field(output, w, h):
    """Draw sky gradient and field stripes."""
    # Sky gradient
    for row in range(h // 3):
        t = row / max(1, h // 3)
        color = (
            int(65 + (130 - 65) * t),
            int(125 + (190 - 125) * t),
            int(200 + (235 - 200) * t),
        )
        cv2.line(output, (0, row), (w, row), color)

    # Field stripes
    stripe_wz = 4.0
    z = NEAR_Z + 0.1
    i = 0
    while z < FIELD_Z_FAR:
        z2 = min(z + stripe_wz, FIELD_Z_FAR)
        col = F_GREEN if i % 2 == 0 else F_STRIPE
        pts = []
        for wz in (z, z2):
            for wx in (-FIELD_HALF_W, FIELD_HALF_W):
                p = proj_ground(wx, wz, w, h)
                if p:
                    pts.append(p[:2])
        if len(pts) == 4:
            poly = np.array([pts[0], pts[1], pts[3], pts[2]], np.int32)
            cv2.fillPoly(output, [poly], col)
        z += stripe_wz
        i += 1

    # Field lines
    lw = 2
    def gline(x1, z1, x2, z2):
        a = proj_ground(x1, z1, w, h)
        b = proj_ground(x2, z2, w, h)
        if a and b:
            cv2.line(output, a[:2], b[:2], WHITE, lw)

    gline(-FIELD_HALF_W, NEAR_Z + 0.2, -FIELD_HALF_W, FIELD_Z_FAR)
    gline(FIELD_HALF_W, NEAR_Z + 0.2, FIELD_HALF_W, FIELD_Z_FAR)
    gline(-FIELD_HALF_W, GOAL_Z, FIELD_HALF_W, GOAL_Z)

    # Penalty area
    pa_hw = GOAL_HW + 5.0
    gline(-pa_hw, GOAL_Z, -pa_hw, GOAL_Z - 8.0)
    gline(pa_hw, GOAL_Z, pa_hw, GOAL_Z - 8.0)
    gline(-pa_hw, GOAL_Z - 8.0, pa_hw, GOAL_Z - 8.0)

    # Penalty spot
    ps = proj_ground(0, PENALTY_Z, w, h)
    if ps:
        cv2.circle(output, ps[:2], 4, WHITE, -1)


def draw_goal(output, w, h):
    """Draw goal posts and net."""
    def post_quad(wx_l, wy_b, wx_r, wy_t, wz):
        corners = [
            project(wx_l, wy_b, wz, w, h),
            project(wx_r, wy_b, wz, w, h),
            project(wx_r, wy_t, wz, w, h),
            project(wx_l, wy_t, wz, w, h),
        ]
        pts = [c[:2] for c in corners if c]
        if len(pts) == 4:
            pts_arr = np.array(pts, np.int32)
            cv2.fillPoly(output, [pts_arr], GOAL_COL)
            cv2.polylines(output, [pts_arr], True, WHITE, 1)

    pw = 0.15
    post_quad(-GOAL_HW - pw, 0, -GOAL_HW + pw, GOAL_H, GOAL_Z)
    post_quad(GOAL_HW - pw, 0, GOAL_HW + pw, GOAL_H, GOAL_Z)
    post_quad(-GOAL_HW - pw, GOAL_H - pw, GOAL_HW + pw, GOAL_H + pw, GOAL_Z)

    # Net lines
    for i in range(8):
        wx = -GOAL_HW + i * (GOAL_HW * 2 / 7)
        a = project(wx, 0.0, GOAL_Z, w, h)
        b = project(wx, GOAL_H, GOAL_Z, w, h)
        if a and b:
            cv2.line(output, a[:2], b[:2], NET_COL, 1)
    for j in range(5):
        wy = j * (GOAL_H / 4)
        a = project(-GOAL_HW, wy, GOAL_Z, w, h)
        b = project(GOAL_HW, wy, GOAL_Z, w, h)
        if a and b:
            cv2.line(output, a[:2], b[:2], NET_COL, 1)


def draw_goalkeeper(output, gk, w, h):
    """Draw goalkeeper as a projected quad."""
    gk_wz = GOAL_Z - 0.3
    bl = project(gk.x - GK_HW, 0.0, gk_wz, w, h)
    br = project(gk.x + GK_HW, 0.0, gk_wz, w, h)
    tl = project(gk.x - GK_HW, GK_H, gk_wz, w, h)
    tr = project(gk.x + GK_HW, GK_H, gk_wz, w, h)

    if not all([bl, br, tl, tr]):
        return

    pts = np.array([bl[:2], br[:2], tr[:2], tl[:2]], np.int32)
    cv2.fillPoly(output, [pts], GK_COL)
    cv2.polylines(output, [pts], True, (0, 80, 180), 2)

    # Head
    head_c = project(gk.x, GK_H + 0.5, gk_wz, w, h)
    if head_c:
        hr = max(3, int(0.5 * FOCAL / gk_wz))
        cv2.circle(output, head_c[:2], hr, (120, 165, 200), -1)


def draw_ball(output, ball, w, h):
    """Draw the soccer ball with shadow and rotation."""
    p = project(ball.x, ball.y, ball.z, w, h)
    if p is None:
        return
    bsx, bsy, scale = p
    br = max(3, int(BALL_R_W * scale))

    # Shadow
    shadow_p = proj_ground(ball.x, ball.z, w, h)
    if shadow_p:
        h_ratio = max(0.0, 1.0 - ball.y / (GOAL_H * 1.5))
        sh_r = max(2, int(br * 1.4 * h_ratio))
        if sh_r > 0:
            cv2.ellipse(output, shadow_p[:2], (sh_r, sh_r // 3), 0, 0, 360, (30, 90, 30), -1)

    # Ball body
    cv2.circle(output, (bsx, bsy), br, WHITE, -1)

    # Pentagon dots
    for k in range(5):
        angle = math.radians(ball.rot + k * 72)
        px = int(bsx + math.cos(angle) * br * 0.5)
        py = int(bsy + math.sin(angle) * br * 0.5)
        dot_r = max(1, br // 4)
        cv2.circle(output, (px, py), dot_r, (30, 30, 30), -1)

    # Outline
    cv2.circle(output, (bsx, bsy), br, (60, 60, 60), max(1, br // 8))


def draw_leg_skeleton(output, leg_pos, w, h, kick_cd, ball_sx, ball_sy):
    """Draw detected leg keypoints and kick ring."""
    if not leg_pos:
        return

    skeleton = [
        (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    ]

    for a_idx, b_idx in skeleton:
        if a_idx in leg_pos and b_idx in leg_pos:
            ax = int(leg_pos[a_idx][0] * w)
            ay = int(leg_pos[a_idx][1] * h)
            bx = int(leg_pos[b_idx][0] * w)
            by = int(leg_pos[b_idx][1] * h)
            cv2.line(output, (ax, ay), (bx, by), (180, 180, 180), 4)

    for kp_idx in LEG_KEYPOINTS:
        if kp_idx not in leg_pos:
            continue
        kx = int(leg_pos[kp_idx][0] * w)
        ky = int(leg_pos[kp_idx][1] * h)
        if kp_idx in ANKLE_INDICES:
            col = (0, 180, 255)
        elif kp_idx in (LEFT_KNEE, RIGHT_KNEE):
            col = (100, 220, 100)
        else:
            col = (255, 200, 0)
        cv2.circle(output, (kx, ky), 7, col, -1)
        cv2.circle(output, (kx, ky), 7, WHITE, 2)

    # Kick ring around ankle
    for ankle_idx in ANKLE_INDICES:
        if ankle_idx not in leg_pos:
            continue
        fx = int(leg_pos[ankle_idx][0] * w)
        fy = int(leg_pos[ankle_idx][1] * h)
        dist = math.hypot(fx - ball_sx, fy - ball_sy)
        if dist < KICK_SCREEN_R:
            col = (0, 220, 255) if kick_cd <= 0 else (80, 200, 255)
            cv2.circle(output, (fx, fy), KICK_SCREEN_R, col, 2)

        fc = (0, 60, 255) if kick_cd > 40 else (0, 240, 255) if kick_cd <= 0 else (0, 165, 255)
        cv2.circle(output, (fx, fy), 11, fc, -1)
        cv2.circle(output, (fx, fy), 11, WHITE, 2)


# ─── PiP camera preview ──────────────────────────────────────────────────────
PIP_W = 240
PIP_H = 135
PIP_MARGIN = 12

def draw_pip(output, pip_frame, skeleton_kps, w, h):
    """Overlay a small camera preview with skeleton in the bottom-right corner."""
    if pip_frame is None:
        return
    ph, pw = pip_frame.shape[:2]
    if pw == 0 or ph == 0:
        return
    # Scale to PIP size
    scale = min(PIP_W / pw, PIP_H / ph)
    new_w = int(pw * scale)
    new_h = int(ph * scale)
    resized = cv2.resize(pip_frame, (new_w, new_h))

    # Draw skeleton overlay on PiP
    if skeleton_kps:
        for a, b in COCO_SKELETON:
            if a < len(skeleton_kps) and b < len(skeleton_kps):
                ax = int(skeleton_kps[a][0] * new_w)
                ay = int(skeleton_kps[a][1] * new_h)
                bx = int(skeleton_kps[b][0] * new_w)
                by = int(skeleton_kps[b][1] * new_h)
                cv2.line(resized, (ax, ay), (bx, by), (0, 255, 0), 1)
        for i, (xn, yn) in enumerate(skeleton_kps):
            kx = int(xn * new_w)
            ky = int(yn * new_h)
            r = 3 if i in (9, 10) else 2  # wrists bigger
            col = (0, 255, 255) if i in (9, 10) else (0, 200, 0)
            cv2.circle(resized, (kx, ky), r, col, -1)

    # Position bottom-right
    x0 = w - new_w - PIP_MARGIN
    y0 = h - new_h - PIP_MARGIN

    # Border
    cv2.rectangle(output, (x0 - 2, y0 - 2), (x0 + new_w + 2, y0 + new_h + 2), (200, 200, 200), 2)
    # Label
    cv2.putText(output, "CAM", (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # Overlay the camera frame
    output[y0:y0 + new_h, x0:x0 + new_w] = resized


def draw_hud(output, score, attempts, state, w, h):
    """Draw score and state banners."""
    # Score panel
    overlay = output[:60, :220, :].copy()
    cv2.rectangle(overlay, (0, 0), (220, 60), (0, 0, 0), -1)
    output[:60, :220, :] = cv2.addWeighted(overlay, 0.7, output[:60, :220, :], 0.3, 0)
    cv2.putText(output, f"GOALS: {score}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
    if attempts > 0:
        pct = int(score / attempts * 100)
        cv2.putText(output, f"Shots: {attempts}  {pct}%", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # State banners
    if state == S_WAITING:
        now = time.time()
        pulse = 0.5 + 0.5 * math.sin(now * 4)
        col = (0, int(230 * pulse), int(255 * pulse))
        cv2.putText(output, "KICK!", (w // 2 - 60, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 3)

    elif state == S_GOAL:
        cv2.putText(output, "GOAL!", (w // 2 - 100, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 220, 255), 4)

    elif state == S_BLOCKED:
        cv2.putText(output, "SAVED!", (w // 2 - 100, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (30, 80, 255), 4)

    elif state == S_MISS:
        cv2.putText(output, "MISS!", (w // 2 - 80, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (90, 90, 200), 4)

    # Timer bar
    elapsed = time.time() - output._game_start if hasattr(output, '_game_start') else 0
    remaining = max(0, GAME_DURATION - elapsed)


# ─── Callback class ─────────────────────────────────────────────────────────
class SoccerCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True
        self.leg_queue = queue.Queue(maxsize=4)
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


# ─── Callback function ──────────────────────────────────────────────────────
def app_callback(element, buffer, user_data):
    import queue
    pad = element.get_static_pad("src")
    fmt, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and fmt and width and height:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    if frame is None:
        return Gst.FlowReturn.OK

    # Save raw frame for PiP before any rendering
    user_data.pip_frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 1)

    # Extract leg keypoints
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    leg_pos = {}
    for detection in detections:
        if detection.get_label() != "person":
            continue
        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        pts = landmarks[0].get_points()
        for idx in LEG_KEYPOINTS:
            if idx < len(pts):
                pt = pts[idx]
                leg_pos[idx] = (
                    float(pt.x() * bbox.width() + bbox.xmin()),
                    float(pt.y() * bbox.height() + bbox.ymin()),
                )
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

    # Get game state from user_data (set in main loop)
    if not hasattr(user_data, 'ball'):
        user_data.ball = Ball()
        user_data.gk = Goalkeeper()
        user_data.score = 0
        user_data.attempts = 0
        user_data.state = S_WAITING
        user_data.state_timer = 0
        user_data.kick_cd = 0
        user_data.game_start = time.time()
        user_data.leg_hist = {idx: deque(maxlen=6) for idx in LEG_KEYPOINTS}
        user_data.particles = []
        user_data.popups = []

    ball = user_data.ball
    gk = user_data.gk

    # Update leg history
    for kp_idx, (xn, yn) in leg_pos.items():
        user_data.leg_hist[kp_idx].append((xn * width, yn * height))

    # Ball physics
    if user_data.state == S_WAITING:
        ball.y = 0.1 * math.sin(time.time() * 2)
        ball.rot += 0.8

    elif user_data.state == S_FLIGHT:
        ball.x += ball.vx
        ball.y += ball.vy
        ball.z += ball.vz
        ball.vy -= GRAVITY_W
        ball.vx *= FRICTION_W
        ball.rot += ball.vz * 6

        if ball.y < 0.0:
            ball.y = 0.0
            if abs(ball.vy) > 0.02:
                ball.vy *= -BOUNCE_DAMP
            else:
                ball.vy = 0.0

        if ball.z >= GOAL_Z:
            in_goal = (-GOAL_HW < ball.x < GOAL_HW) and (0.0 < ball.y < GOAL_H)
            hits_gk = (abs(ball.x - gk.x) < GK_HW * 1.1) and (0.0 < ball.y < GK_H * 1.05)

            if in_goal and not hits_gk:
                user_data.state = S_GOAL
                user_data.score += 1
                user_data.attempts += 1
                user_data.state_timer = STATE_HOLD
                logger.info("GOAL! Score: %d", user_data.score)
            elif hits_gk:
                user_data.state = S_BLOCKED
                user_data.attempts += 1
                user_data.state_timer = STATE_HOLD
                logger.info("SAVED!")
            else:
                user_data.state = S_MISS
                user_data.attempts += 1
                user_data.state_timer = STATE_HOLD

        elif ball.z < 0 or abs(ball.x) > FIELD_HALF_W * 1.5 or ball.y > GOAL_H * 3:
            user_data.state = S_MISS
            user_data.attempts += 1
            user_data.state_timer = STATE_HOLD // 2

    # State timer
    if user_data.state in (S_GOAL, S_BLOCKED, S_MISS):
        user_data.state_timer -= 1
        if user_data.state_timer <= 0:
            ball.reset()
            user_data.state = S_WAITING

    # Goalkeeper
    gk.update(ball, user_data.state)

    # Kick detection
    user_data.kick_cd = max(0, user_data.kick_cd - 1)
    if user_data.state == S_WAITING and user_data.kick_cd == 0:
        bp = project(ball.x, ball.y, ball.z, width, height)
        if bp:
            bsx, bsy = bp[0], bp[1]
            for ankle_idx in ANKLE_INDICES:
                if ankle_idx not in leg_pos:
                    continue
                fx = int(leg_pos[ankle_idx][0] * width)
                fy = int(leg_pos[ankle_idx][1] * height)
                if math.hypot(fx - bsx, fy - bsy) >= KICK_SCREEN_R:
                    continue

                hist = user_data.leg_hist[ankle_idx]
                vel = 0.0
                if len(hist) >= 2:
                    vel = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])

                if vel < KICK_VEL_THR:
                    continue

                dx_s = bsx - fx
                dy_s = bsy - fy
                kick_vz = float(np.clip(KICK_VZ_BASE + vel * KICK_VZ_BONUS, KICK_VZ_BASE, 2.0))
                kick_vx = dx_s * KICK_LAT
                kick_vy = -dy_s * KICK_LIFT

                ball.vx = kick_vx
                ball.vy = max(0.0, kick_vy)
                ball.vz = kick_vz
                user_data.state = S_FLIGHT
                user_data.kick_cd = KICK_COOLDOWN_FRAMES
                logger.info("KICK! vz=%.2f", kick_vz)
                break

    # Render
    output = np.zeros((height, width, 3), dtype=np.uint8)
    draw_field(output, width, height)
    draw_goal(output, width, height)
    draw_goalkeeper(output, gk, width, height)
    draw_ball(output, ball, width, height)

    bp = project(ball.x, ball.y, ball.z, width, height)
    bsx, bsy = (bp[0], bp[1]) if bp else (width // 2, int(height * 0.85))
    draw_leg_skeleton(output, leg_pos, width, height, user_data.kick_cd, bsx, bsy)
    draw_hud(output, user_data.score, user_data.attempts, user_data.state, width, height)
    draw_pip(output, user_data.pip_frame, user_data.skeleton_kps, width, height)

    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    user_data.set_frame(output)

    return Gst.FlowReturn.OK


# ─── App class ───────────────────────────────────────────────────────────────
class SoccerApp(GStreamerPoseEstimationApp):
    def __init__(self, app_callback, user_data, parser=None):
        super().__init__(app_callback, user_data, parser)
        self.options_menu.use_frame = True
        user_data.use_frame = True


def main():
    parser = get_pipeline_parser()
    args, _ = parser.parse_known_args()

    user_data = SoccerCallback()
    app = SoccerApp(app_callback, user_data, parser)
    app.run()


if __name__ == "__main__":
    main()
