"""Space Invaders — steer with your head, raise hands to fire, defend Earth from aliens."""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import hailo
import math
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

# --- Constants ---
GAME_DURATION = 90          # seconds
PLAYER_SPEED = 12           # pixels per frame (for smooth tracking, this is interpolation)
BULLET_SPEED = 18           # pixels per frame upward
ALIEN_SPEED_BASE = 1.5     # pixels per frame
ALIEN_DESCENT = 30          # pixels to drop when hitting edge
ALIEN_ROWS = 4
ALIEN_COLS = 6
ALIEN_SPACING_X = 70
ALIEN_SPACING_Y = 55
FIRE_COOLDOWN = 0.25        # seconds between shots
POPUP_DURATION = 0.6        # seconds for score popups
RESTART_DELAY = 5           # seconds before auto-restart

# PiP constants
PIP_W = 240
PIP_H = 135
PIP_MARGIN = 12

# Keypoint indices
NOSE = 0
LEFT_WRIST = 9
RIGHT_WRIST = 10

# COCO 17 skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 6), (5, 11), (6, 12), (11, 12),    # torso
    (11, 13), (13, 15), (12, 14), (14, 16) # legs
]

# Alien type point values (top rows worth more)
ALIEN_POINTS = [40, 30, 20, 10]  # row 0 (top) → row 3 (bottom)

# Alien colours (RGB) per row
ALIEN_COLORS = [
    (255, 60, 60),      # red (top row)
    (255, 160, 40),     # orange
    (60, 220, 60),      # green
    (60, 180, 255),     # blue (bottom row)
]

INITIAL_LIVES = 3


# ─── Game objects ────────────────────────────────────────────────────────────
class Alien:
    def __init__(self, row, col, x, y):
        self.row = row
        self.col = col
        self.x = x
        self.y = y
        self.alive = True
        self.points = ALIEN_POINTS[min(row, len(ALIEN_POINTS) - 1)]
        self.color = ALIEN_COLORS[min(row, len(ALIEN_COLORS) - 1)]
        self.frame = 0  # animation frame


class Bullet:
    def __init__(self, x, y, is_alien=False):
        self.x = x
        self.y = y
        self.is_alien = is_alien
        self.alive = True


class AlienBullet(Bullet):
    def __init__(self, x, y):
        super().__init__(x, y, is_alien=True)


class Popup:
    def __init__(self, text, x, y, colour):
        self.text = text
        self.x = x
        self.y = y
        self.colour = colour
        self.spawn = time.time()

    def alive(self):
        return (time.time() - self.spawn) < POPUP_DURATION

    def alpha(self):
        age = time.time() - self.spawn
        return max(0.0, 1.0 - age / POPUP_DURATION)

    def current_y(self):
        age = time.time() - self.spawn
        return int(self.y - age * 50)


# ─── Drawing helpers ────────────────────────────────────────────────────────
def draw_alien(img, alien, frame_width, frame_height):
    """Draw a Space Invaders alien at its position."""
    x, y = int(alien.x), int(alien.y)
    size = 18
    color = alien.color

    # Body — rectangular with notches (classic space invader shape)
    # Main body
    cv2.rectangle(img, (x - size, y - size // 2), (x + size, y + size // 2), color, -1, cv2.LINE_AA)
    # Eyes
    eye_y = y - 3
    cv2.circle(img, (x - 7, eye_y), 4, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.circle(img, (x + 7, eye_y), 4, (30, 30, 30), -1, cv2.LINE_AA)
    # Eye highlights
    cv2.circle(img, (x - 6, eye_y - 1), 1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (x + 8, eye_y - 1), 1, (255, 255, 255), -1, cv2.LINE_AA)

    # Antennae
    anim_offset = 4 if alien.frame % 2 == 0 else -4
    cv2.line(img, (x - 8, y - size // 2), (x - 14, y - size // 2 - 8 + anim_offset), color, 2, cv2.LINE_AA)
    cv2.line(img, (x + 8, y - size // 2), (x + 14, y - size // 2 - 8 + anim_offset), color, 2, cv2.LINE_AA)

    # Legs (alternating animation)
    leg_y = y + size // 2
    if alien.frame % 2 == 0:
        cv2.line(img, (x - 10, leg_y), (x - 14, leg_y + 8), color, 2, cv2.LINE_AA)
        cv2.line(img, (x + 10, leg_y), (x + 14, leg_y + 8), color, 2, cv2.LINE_AA)
    else:
        cv2.line(img, (x - 10, leg_y), (x - 6, leg_y + 8), color, 2, cv2.LINE_AA)
        cv2.line(img, (x + 10, leg_y), (x + 6, leg_y + 8), color, 2, cv2.LINE_AA)

    # Outline
    cv2.rectangle(img, (x - size, y - size // 2), (x + size, y + size // 2), (255, 255, 255), 1, cv2.LINE_AA)


def draw_player_ship(img, x, y, frame_width):
    """Draw the player's spaceship."""
    # Ship body — triangle pointing up
    pts = np.array([
        [int(x), int(y) - 22],          # tip
        [int(x) - 18, int(y) + 12],     # bottom left
        [int(x) + 18, int(y) + 12],     # bottom right
    ], np.int32)

    # Glow
    cv2.fillPoly(img, [pts + 2], (40, 120, 40), cv2.LINE_AA)
    # Main body
    cv2.fillPoly(img, [pts], (0, 200, 0), cv2.LINE_AA)
    # Outline
    cv2.polylines(img, [pts], True, (100, 255, 100), 2, cv2.LINE_AA)
    # Cockpit
    cv2.circle(img, (int(x), int(y) - 4), 5, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y) - 4), 5, (200, 255, 200), 1, cv2.LINE_AA)


def draw_bullet(img, bullet, is_alien=False):
    """Draw a bullet."""
    x, y = int(bullet.x), int(bullet.y)
    if is_alien:
        # Alien bullet — red diamond
        pts = np.array([[x, y - 6], [x + 3, y], [x, y + 6], [x - 3, y]], np.int32)
        cv2.fillPoly(img, [pts], (255, 60, 60), cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (255, 150, 150), 1, cv2.LINE_AA)
    else:
        # Player bullet — green line
        cv2.line(img, (x, y - 6), (x, y + 6), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(img, (x, y - 6), (x, y + 6), (200, 255, 200), 1, cv2.LINE_AA)


# ─── Callback class ─────────────────────────────────────────────────────────
class SpaceInvadersCallback(app_callback_class):
    """Holds all game state across frames."""

    def __init__(self):
        super().__init__()
        self.use_frame = True

        # Game state
        self.score = 0
        self.lives = INITIAL_LIVES
        self.game_start = None
        self.game_over = False
        self.game_over_time = None

        # Player ship
        self.player_x = 0.0
        self.player_y = 0.0  # set based on frame height

        # Aliens
        self.aliens = []
        self.alien_direction = 1  # 1 = right, -1 = left
        self.alien_drop_pending = False
        self.alien_anim_timer = 0.0
        self.alien_anim_interval = 0.4  # seconds between animation frames

        # Bullets
        self.bullets = []
        self.alien_bullets = []
        self.last_fire_time = 0.0

        # Popups
        self.popups = []

        # Stars background
        self.stars = []

        # Frame dimensions
        self.fw = 0
        self.fh = 0

        # PiP camera preview
        self.pip_frame = None
        self.skeleton_kps = []
        self.pip_cache = None  # cached flipped+converted frame
        self.pip_cache_shape = None  # (h, w) of cached frame
        self.pip_skeleton_cache = None  # cached PiP with skeleton drawn
        self.pip_last_update = 0.0  # last time PiP was updated
        self.pip_update_interval = 0.1  # update PiP every 100ms (10 FPS for camera)

        # Pre-allocated render buffer (reused every frame to avoid np.zeros)
        self._render_buffer = None
        self._render_buffer_shape = None

    def set_frame(self, frame):
        """Override to drain stale frames so display always shows the latest."""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break
        try:
            self.frame_queue.put_nowait(frame)
        except Exception:
            pass

    def init_aliens(self):
        """Create the initial alien formation."""
        self.aliens = []
        # Center the formation horizontally
        total_width = ALIEN_COLS * ALIEN_SPACING_X
        start_x = (self.fw - total_width) // 2 + ALIEN_SPACING_X // 2
        start_y = 80

        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                x = start_x + col * ALIEN_SPACING_X
                y = start_y + row * ALIEN_SPACING_Y
                self.aliens.append(Alien(row, col, x, y))

    def init_stars(self):
        """Create random background stars."""
        self.stars = []
        for _ in range(80):
            x = random.randint(0, self.fw)
            y = random.randint(0, self.fh)
            brightness = random.randint(80, 220)
            size = random.choice([1, 1, 1, 2])
            self.stars.append((x, y, brightness, size))

    def restart(self):
        """Reset game state for a new round."""
        self.score = 0
        self.lives = INITIAL_LIVES
        self.game_start = None
        self.game_over = False
        self.game_over_time = None
        self.aliens = []
        self.bullets = []
        self.alien_bullets = []
        self.popups = []
        self.alien_direction = 1
        self.alien_drop_pending = False
        self.last_fire_time = 0.0


# ─── Callback function ──────────────────────────────────────────────────────
def app_callback(element, buffer, user_data):
    pad = element.get_static_pad("src")
    fmt, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and fmt and width and height:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    if frame is None:
        return Gst.FlowReturn.OK

    now = time.time()

    # Save raw frame for PiP - only update periodically to reduce overhead
    now_for_pip = time.time()
    if now_for_pip - user_data.pip_last_update >= user_data.pip_update_interval:
        user_data.pip_last_update = now_for_pip
        # Only convert+flip if frame shape changed
        if user_data.pip_cache_shape != frame.shape[:2]:
            user_data.pip_cache_shape = frame.shape[:2]
            user_data.pip_cache = np.empty(frame.shape[:2] + (3,), dtype=np.uint8)
        # Convert RGB→BGR and flip in one pass: flip then convert
        flipped = cv2.flip(frame, 1)  # flip RGB
        cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR, dst=user_data.pip_cache)
        user_data.pip_frame = user_data.pip_cache

    # Lazy init game clock
    if user_data.game_start is None:
        user_data.game_start = now
        user_data.fw = width
        user_data.fh = height
        user_data.player_x = width / 2
        user_data.player_y = height - 60
        user_data.init_aliens()
        user_data.init_stars()

    elapsed = now - user_data.game_start
    remaining = max(0.0, GAME_DURATION - elapsed)

    # --- Auto-restart after game over ---
    if user_data.game_over:
        if user_data.game_over_time and (now - user_data.game_over_time) >= RESTART_DELAY:
            user_data.restart()
            return Gst.FlowReturn.OK
        output = _render_frame(user_data, width, height, now)
        _draw_game_over(output, user_data, width, height)
        cv2.cvtColor(output, cv2.COLOR_RGB2BGR, dst=output)
        user_data.set_frame(output)
        return Gst.FlowReturn.OK

    # --- Check if time's up ---
    if remaining <= 0:
        user_data.game_over = True
        user_data.game_over_time = now
        return Gst.FlowReturn.OK

    # --- Check if all aliens destroyed ---
    if not any(a.alive for a in user_data.aliens):
        user_data.init_aliens()
        user_data.alien_direction = 1

    # --- Get player hand position ---
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    player_head_x = None
    both_hands_up = False

    for detection in detections:
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        points = landmarks[0].get_points()
        if len(points) <= RIGHT_WRIST:
            continue

        # Nose for head tracking
        nose = points[NOSE]
        nose_y = (nose.y() * bbox.height() + bbox.ymin()) * height
        player_head_x = (nose.x() * bbox.width() + bbox.xmin()) * width

        # Check if both wrists are above the head (y increases downward)
        lw = points[LEFT_WRIST]
        rw = points[RIGHT_WRIST]
        lw_y = (lw.y() * bbox.height() + bbox.ymin()) * height
        rw_y = (rw.y() * bbox.height() + bbox.ymin()) * height

        if lw_y < nose_y and rw_y < nose_y:
            both_hands_up = True

        break  # use first person detected

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

    # --- Update player ship position ---
    if player_head_x is not None:
        # Smoothly interpolate ship x toward head position
        target_x = np.clip(player_head_x, 40, width - 40)
        user_data.player_x += (target_x - user_data.player_x) * 0.25

    # --- Fire on hands-up gesture (bypasses auto-fire cooldown) ---
    if both_hands_up and (now - user_data.last_fire_time) >= FIRE_COOLDOWN * 0.5:
        user_data.bullets.append(Bullet(user_data.player_x, user_data.player_y - 22))
        user_data.last_fire_time = now
        logger.debug("Hands up! Fired bullet.")

    # --- Auto-fire (slower passive shooting) ---
    if now - user_data.last_fire_time >= FIRE_COOLDOWN:
        user_data.bullets.append(Bullet(user_data.player_x, user_data.player_y - 22))
        user_data.last_fire_time = now

    # --- Update aliens ---
    alien_alive = [a for a in user_data.aliens if a.alive]
    if alien_alive:
        # Animation timer
        if now - user_data.alien_anim_timer >= user_data.alien_anim_interval:
            for a in alien_alive:
                a.frame += 1
            user_data.alien_anim_timer = now

        # Determine speed based on how many aliens remain
        speed_mult = 1.0 + (1.0 - len(alien_alive) / max(1, ALIEN_ROWS * ALIEN_COLS)) * 2.0
        alien_speed = ALIEN_SPEED_BASE * speed_mult

        # Check bounds
        hit_edge = False
        for a in alien_alive:
            next_x = a.x + alien_speed * user_data.alien_direction
            if next_x < 30 or next_x > width - 30:
                hit_edge = True
                break

        if hit_edge:
            user_data.alien_direction *= -1
            for a in alien_alive:
                a.y += ALIEN_DESCENT
                # Check if aliens reached the player — game over
                if a.y >= user_data.player_y - 30:
                    user_data.game_over = True
                    user_data.game_over_time = now
                    return Gst.FlowReturn.OK
        else:
            for a in alien_alive:
                a.x += alien_speed * user_data.alien_direction

    # --- Alien shooting ---
    if alien_alive and random.random() < 0.02:  # ~2% chance per frame
        shooter = random.choice(alien_alive)
        user_data.alien_bullets.append(AlienBullet(shooter.x, shooter.y + 18))

    # --- Update bullets ---
    for b in user_data.bullets:
        b.y -= BULLET_SPEED
        if b.y < 0:
            b.alive = False

    for b in user_data.alien_bullets:
        b.y += BULLET_SPEED * 0.6
        if b.y > height:
            b.alive = False

    # --- Collision: player bullets vs aliens ---
    for b in user_data.bullets:
        if not b.alive:
            continue
        for a in user_data.aliens:
            if not a.alive:
                continue
            dist = math.hypot(b.x - a.x, b.y - a.y)
            if dist < 22:
                b.alive = False
                a.alive = False
                user_data.score += a.points
                user_data.popups.append(
                    Popup(f"+{a.points}", a.x, a.y, a.color)
                )
                logger.info("Hit alien at (%.0f, %.0f) +%d pts", a.x, a.y, a.points)
                break

    # --- Collision: alien bullets vs player ---
    for b in user_data.alien_bullets:
        if not b.alive:
            continue
        dist = math.hypot(b.x - user_data.player_x, b.y - user_data.player_y)
        if dist < 20:
            b.alive = False
            user_data.lives -= 1
            user_data.popups.append(
                Popup("-1 LIFE", user_data.player_x, user_data.player_y - 40, (255, 80, 80))
            )
            logger.info("Player hit! Lives remaining: %d", user_data.lives)
            if user_data.lives <= 0:
                user_data.game_over = True
                user_data.game_over_time = now
                return Gst.FlowReturn.OK

    # Cleanup dead objects
    user_data.bullets = [b for b in user_data.bullets if b.alive]
    user_data.alien_bullets = [b for b in user_data.alien_bullets if b.alive]

    # --- Render ---
    output = _render_frame(user_data, width, height, now)

    # Convert RGB → BGR for set_frame (in-place to avoid allocation)
    cv2.cvtColor(output, cv2.COLOR_RGB2BGR, dst=output)
    user_data.set_frame(output)

    return Gst.FlowReturn.OK


def _render_frame(user_data, width, height, now):
    """Render the full game frame (stars, aliens, bullets, player, HUD)."""
    # Reuse pre-allocated buffer to avoid np.zeros every frame
    if (user_data._render_buffer is None or
            user_data._render_buffer_shape != (height, width)):
        user_data._render_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        user_data._render_buffer_shape = (height, width)
    output = user_data._render_buffer
    output[:] = (8, 8, 24)  # dark blue-black (reset)

    # Stars (twinkling)
    for sx, sy, brightness, size in user_data.stars:
        twinkle = brightness + int(30 * math.sin(now * 2 + sx * 0.1))
        twinkle = max(50, min(255, twinkle))
        color = (twinkle, twinkle, twinkle)
        cv2.circle(output, (sx, sy), size, color, -1, cv2.LINE_AA)

    # Aliens
    for alien in user_data.aliens:
        if alien.alive:
            draw_alien(output, alien, width, height)

    # Player bullets
    for b in user_data.bullets:
        draw_bullet(output, b, is_alien=False)

    # Alien bullets
    for b in user_data.alien_bullets:
        draw_bullet(output, b, is_alien=True)

    # Player ship
    draw_player_ship(output, user_data.player_x, user_data.player_y, width)

    # Hand tracking indicator (subtle)
    if user_data.player_x > 0:
        cv2.circle(output, (int(user_data.player_x), int(user_data.player_y)),
                   30, (0, 80, 0), 1, cv2.LINE_AA)

    # Popups
    user_data.popups = [p for p in user_data.popups if p.alive()]
    for p in user_data.popups:
        alpha = p.alpha()
        cy = p.current_y()
        color = tuple(int(c * alpha) for c in p.colour)
        cv2.putText(output, p.text, (int(p.x) - 25, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # HUD
    remaining = max(0.0, GAME_DURATION - (now - user_data.game_start))
    _draw_hud(output, user_data.score, user_data.lives, remaining, width)

    # PiP camera preview with skeleton
    if user_data.pip_frame is not None:
        ph, pw = user_data.pip_frame.shape[:2]
        if pw > 0 and ph > 0:
            scale = min(PIP_W / pw, PIP_H / ph)
            new_w, new_h = int(pw * scale), int(ph * scale)
            # Only resize if size changed
            if user_data.pip_skeleton_cache is None or user_data.pip_skeleton_cache.shape[:2] != (new_h, new_w):
                user_data.pip_skeleton_cache = cv2.resize(user_data.pip_frame, (new_w, new_h))
            # Draw skeleton on a fresh copy to avoid corrupting cache
            pip_display = user_data.pip_skeleton_cache.copy()
            if user_data.skeleton_kps:
                for a, b in COCO_SKELETON:
                    if a < len(user_data.skeleton_kps) and b < len(user_data.skeleton_kps):
                        ax = int(user_data.skeleton_kps[a][0] * new_w)
                        ay = int(user_data.skeleton_kps[a][1] * new_h)
                        bx = int(user_data.skeleton_kps[b][0] * new_w)
                        by = int(user_data.skeleton_kps[b][1] * new_h)
                        cv2.line(pip_display, (ax, ay), (bx, by), (0, 255, 0), 1)
                for i, (xn, yn) in enumerate(user_data.skeleton_kps):
                    kx, ky = int(xn * new_w), int(yn * new_h)
                    r = 3 if i in (9, 10) else 2
                    col = (0, 255, 255) if i in (9, 10) else (0, 200, 0)
                    cv2.circle(pip_display, (kx, ky), r, col, -1)
            x0 = width - new_w - PIP_MARGIN
            y0 = height - new_h - PIP_MARGIN
            cv2.rectangle(output, (x0 - 2, y0 - 2), (x0 + new_w + 2, y0 + new_h + 2), (200, 200, 200), 2)
            cv2.putText(output, "CAM", (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            output[y0:y0 + new_h, x0:x0 + new_w] = pip_display

    return output


def _draw_hud(img, score, lives, remaining, width):
    """Draw score, lives, and timer at the top."""
    bar_h = 50
    # Semi-transparent dark bar
    overlay = img[:bar_h, :, :].copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), (10, 10, 30), -1)
    img[:bar_h, :, :] = cv2.addWeighted(overlay, 0.8, img[:bar_h, :, :], 0.2, 0)

    # Score (left)
    cv2.putText(img, f"SCORE: {score}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Lives (center-left)
    for i in range(lives):
        x = 200 + i * 30
        # Mini ship icon
        pts = np.array([[x, 18], [x - 8, 32], [x + 8, 32]], np.int32)
        cv2.fillPoly(img, [pts], (0, 200, 0), cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (100, 255, 100), 1, cv2.LINE_AA)

    # Timer (center)
    mins = int(remaining) // 60
    secs = int(remaining) % 60
    timer_txt = f"{mins}:{secs:02d}"
    timer_color = (80, 220, 80) if remaining > 20 else (255, 80, 80)
    cv2.putText(img, timer_txt, (width // 2 - 30, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2, cv2.LINE_AA)

    # Title
    cv2.putText(img, "SPACE INVADERS", (width // 2 - 130, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1, cv2.LINE_AA)


def _draw_game_over(img, user_data, width, height):
    """Draw game-over screen."""
    # Dark overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    # Title
    cv2.putText(img, "GAME OVER", (width // 2 - 140, height // 3),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 80, 80), 4, cv2.LINE_AA)

    # Final score
    cv2.putText(img, f"FINAL SCORE: {user_data.score}", (width // 2 - 130, height // 3 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Status message
    if user_data.lives <= 0:
        cv2.putText(img, "Your ship was destroyed!", (width // 2 - 150, height // 3 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "Time's up!", (width // 2 - 80, height // 3 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    # Restart countdown
    if user_data.game_over_time:
        left = max(0, RESTART_DELAY - (time.time() - user_data.game_over_time))
        cv2.putText(img, f"Restarting in {int(left) + 1}s ...",
                    (width // 2 - 140, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2, cv2.LINE_AA)


# ─── App class ───────────────────────────────────────────────────────────────
class SpaceInvadersApp(GStreamerPoseEstimationApp):
    """Pose-estimation pipeline with Space Invaders overlay."""

    def __init__(self, app_callback, user_data, parser=None):
        super().__init__(app_callback, user_data, parser)
        # CRITICAL: force use_frame after parent constructor
        self.options_menu.use_frame = True
        user_data.use_frame = True


def main():
    parser = get_pipeline_parser()
    args, _ = parser.parse_known_args()

    user_data = SpaceInvadersCallback()
    app = SpaceInvadersApp(app_callback, user_data, parser)
    app.run()


if __name__ == "__main__":
    main()
