"""Space Invaders — steer with your head, raise hands to fire, defend Earth from aliens.

Architecture:
  - Camera thread (GStreamer daemon thread): extracts pose data, stores in SharedState.
  - Main thread: pygame main loop at fixed 60 FPS, reads pose, renders, displays.
  - SharedState uses threading.Lock to protect the image frame and pose data
    for safe cross-thread access.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import hailo
import math
import os
import random
import sys
import threading
import time

import pygame

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
LEFT_WRIST = 9
RIGHT_WRIST = 10

# --- Game constants ---
GAME_DURATION = 90
ALIEN_ROWS = 4
ALIEN_COLS = 6
ALIEN_SPACING_X = 70
ALIEN_SPACING_Y = 55
ALIEN_DESCENT = 30
ALIEN_SPEED_BASE = 1.5
BULLET_SPEED = 10
FIRE_COOLDOWN = 0.25
FIRE_COOLDOWN_HANDS_UP = 0.12
INITIAL_LIVES = 3
ALIEN_SHOOT_CHANCE = 0.02
ALIEN_POINTS = [40, 30, 20, 10]
NUM_STARS = 80
RESTART_DELAY = 5
FPS = 60

# --- Tunables ---
MIRROR_X = True

# --- Colours (RGB) ---
BLACK = (0, 0, 0)
DARK_BG = (8, 8, 24)
WHITE = (255, 255, 255)
GREEN = (0, 220, 0)
RED = (220, 30, 30)
YELLOW = (255, 230, 0)
BLUE = (60, 180, 255)
ORANGE = (255, 160, 40)
HUD_BG = (30, 30, 30)
GREY = (120, 120, 140)

ALIEN_COLORS = [
    (255, 60, 60),    # red (top)
    (255, 160, 40),   # orange
    (60, 220, 60),    # green
    (60, 180, 255),   # blue (bottom)
]

# PiP camera preview
PIP_W = 240
PIP_H = 135
PIP_MARGIN = 16
PIP_FADE_ALPHA = 64   # 25% of 255 — applied when ship overlaps PiP
PIP_UPDATE_INTERVAL = 0.1  # Update PiP 10×/sec (faster than camera frame rate)

# COCO 17 skeleton connections for PiP overlay
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 6), (5, 11), (6, 12), (11, 12),    # torso
    (11, 13), (13, 15), (12, 14), (14, 16) # legs
]


# ═══════════════════════════════════════════════════════════════════════════
# Game objects
# ═══════════════════════════════════════════════════════════════════════════
class Alien:
    def __init__(self, row, col, x, y):
        self.row = row
        self.col = col
        self.x = x
        self.y = y
        self.alive = True
        self.points = ALIEN_POINTS[min(row, len(ALIEN_POINTS) - 1)]
        self.color = ALIEN_COLORS[min(row, len(ALIEN_COLORS) - 1)]
        self.frame = 0


class Bullet:
    def __init__(self, x, y, is_alien=False):
        self.x = x
        self.y = y
        self.is_alien = is_alien
        self.alive = True


class Popup:
    def __init__(self, text, x, y, colour, duration=0.6):
        self.text = text
        self.x = x
        self.y = y
        self.colour = colour
        self.spawn = time.time()
        self.duration = duration

    def alive(self):
        return (time.time() - self.spawn) < self.duration

    def alpha(self):
        return max(0.0, 1.0 - (time.time() - self.spawn) / self.duration)

    def current_y(self):
        return int(self.y - (time.time() - self.spawn) * 50)


# ═══════════════════════════════════════════════════════════════════════════
# Drawing helpers (pygame)
# ═══════════════════════════════════════════════════════════════════════════
def draw_alien(surf, alien):
    x, y = int(alien.x), int(alien.y)
    size = 18
    color = alien.color

    # Body
    body = pygame.Rect(x - size, y - size // 2, size * 2, size)
    pygame.draw.rect(surf, color, body)
    pygame.draw.rect(surf, WHITE, body, 1)

    # Eyes
    pygame.draw.circle(surf, RED, (x - 7, y - 3), 4)
    pygame.draw.circle(surf, RED, (x + 7, y - 3), 4)

    # Antennae (animated)
    anim = 4 if alien.frame % 2 == 0 else -4
    pygame.draw.line(surf, color, (x - 8, y - size // 2), (x - 14, y - size // 2 - 8 + anim), 2)
    pygame.draw.line(surf, color, (x + 8, y - size // 2), (x + 14, y - size // 2 - 8 + anim), 2)

    # Legs (alternating)
    if alien.frame % 2 == 0:
        pygame.draw.line(surf, color, (x - 10, y + size // 2), (x - 14, y + size // 2 + 8), 2)
        pygame.draw.line(surf, color, (x + 10, y + size // 2), (x + 14, y + size // 2 + 8), 2)
    else:
        pygame.draw.line(surf, color, (x - 10, y + size // 2), (x - 6, y + size // 2 + 8), 2)
        pygame.draw.line(surf, color, (x + 10, y + size // 2), (x + 6, y + size // 2 + 8), 2)


def draw_player_ship(surf, x, y):
    x, y = int(x), int(y)
    # Triangle ship
    points = [(x, y - 22), (x - 18, y + 12), (x + 18, y + 12)]
    pygame.draw.polygon(surf, GREEN, points)
    pygame.draw.polygon(surf, (100, 255, 100), points, 2)
    # Cockpit
    pygame.draw.circle(surf, (0, 255, 0), (x, y - 4), 5)
    pygame.draw.circle(surf, (200, 255, 200), (x, y - 4), 5, 1)


def draw_bullet(surf, bullet):
    x, y = int(bullet.x), int(bullet.y)
    if bullet.is_alien:
        # Red diamond
        pts = [(x, y - 6), (x + 4, y), (x, y + 6), (x - 4, y)]
        pygame.draw.polygon(surf, RED, pts)
        pygame.draw.polygon(surf, (150, 150, 255), pts, 1)
    else:
        # Green line
        pygame.draw.line(surf, GREEN, (x, y - 7), (x, y + 7), 3)
        pygame.draw.line(surf, (200, 255, 200), (x, y - 7), (x, y + 7), 1)


def draw_hud(surf, score, lives, remaining, width):
    bar_h = 50
    # Background bar
    pygame.draw.rect(surf, HUD_BG, (0, 0, width, bar_h))

    # Progress bar
    frac = max(0.0, min(1.0, remaining / GAME_DURATION))
    bar_w = int((width - 260) * frac)
    bar_color = (80, 220, 80) if remaining > 20 else (80, 80, 255)
    pygame.draw.rect(surf, bar_color, (130, 10, bar_w, 24))
    pygame.draw.rect(surf, GREY, (130, 10, width - 260, 24), 1)

    # Time text
    mins = int(remaining) // 60
    secs = int(remaining) % 60
    time_surf = pygame.font.Font(None, 36).render(f"{mins}:{secs:02d}", True, WHITE)
    surf.blit(time_surf, (20, 10))

    # Score
    score_surf = pygame.font.Font(None, 28).render(f"SCORE: {score}", True, GREEN)
    surf.blit(score_surf, (width - 180, 12))

    # Lives
    for i in range(lives):
        lx = width - 230 + i * 25
        pts = [(lx, 20), (lx - 8, 32), (lx + 8, 32)]
        pygame.draw.polygon(surf, GREEN, pts)


def draw_pip(surf, pip_frame, keypoints, width, height, player_x, player_y):
    """Draw a Picture-in-Picture camera preview in the bottom-right corner
    with the pose skeleton overlay. If the player ship is inside the PiP
    area, fade the camera to 25% opacity.

    Returns the (x, y, w, h) rect of the PiP area, or None if no frame.
    """
    if pip_frame is None:
        return None

    pip_h, pip_w = pip_frame.shape[:2]
    if pip_w == 0 or pip_h == 0:
        return None

    # Position bottom-right
    x0 = width - PIP_W - PIP_MARGIN
    y0 = height - PIP_H - PIP_MARGIN

    # --- Check if ship is inside the PiP area ---
    # If the ship is under the PiP, hide it entirely (no camera, no border).
    ship_in_pip = (x0 <= player_x <= x0 + PIP_W and
                   y0 <= player_y <= y0 + PIP_H)
    if ship_in_pip:
        return None

    # Convert numpy frame (H, W, 3) RGB → pygame Surface
    # pygame.surfarray.make_surface expects (W, H, 3) layout
    pip_surf = pygame.surfarray.make_surface(pip_frame.swapaxes(0, 1))
    pip_surf = pygame.transform.scale(pip_surf, (PIP_W, PIP_H))
    # Flip horizontally so the camera view matches a mirror (natural feel)
    pip_surf = pygame.transform.flip(pip_surf, True, False)

    # --- Draw skeleton ON TOP of the camera frame (on pip_surf, not surf) ---
    # This ensures the skeleton is visible above the camera image,
    # not hidden underneath it.
    if keypoints:
        # Mirror X so the skeleton aligns with the flipped camera frame
        kp_pixels = [(int((1.0 - x) * PIP_W), int(y * PIP_H)) for x, y in keypoints]

        # Skeleton lines (green)
        for a, b in COCO_SKELETON:
            if a < len(kp_pixels) and b < len(kp_pixels):
                pygame.draw.line(pip_surf, (0, 255, 0),
                                 kp_pixels[a], kp_pixels[b], 1)

        # Keypoint dots (wrists highlighted in yellow)
        for i, (px, py) in enumerate(kp_pixels):
            color = (0, 255, 255) if i in (LEFT_WRIST, RIGHT_WRIST) else (0, 220, 0)
            radius = 4 if i in (LEFT_WRIST, RIGHT_WRIST) else 2
            pygame.draw.circle(pip_surf, color, (px, py), radius)

    # Blit the PiP (with skeleton on top) to the main surface
    surf.blit(pip_surf, (x0, y0))

    # Border
    pygame.draw.rect(surf, (200, 200, 200), (x0 - 2, y0 - 2, PIP_W + 4, PIP_H + 4), 2)

    # Label
    font_small = pygame.font.Font(None, 18)
    label = font_small.render("CAM", True, (200, 200, 200))
    surf.blit(label, (x0 + 4, y0 + 2))

    return (x0, y0, PIP_W, PIP_H)


def draw_game_over(surf, score, width, height):
    # Dark overlay
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((10, 10, 30, 200))
    surf.blit(overlay, (0, 0))

    big = pygame.font.Font(None, 80)
    med = pygame.font.Font(None, 48)
    small = pygame.font.Font(None, 32)

    go = big.render("GAME OVER", True, (60, 220, 255))
    surf.blit(go, go.get_rect(center=(width // 2, height // 4)))

    sc = med.render(f"FINAL SCORE: {score}", True, WHITE)
    surf.blit(sc, sc.get_rect(center=(width // 2, height // 4 + 70)))

    rt = small.render("Restarting...", True, (180, 180, 255))
    surf.blit(rt, rt.get_rect(center=(width // 2, height - 60)))


# ═══════════════════════════════════════════════════════════════════════════
# Thread-safe shared state between camera and game threads
# ═══════════════════════════════════════════════════════════════════════════
class SharedState:
    """Thread-safe container for data shared between camera and game threads.

    Holds:
      - Latest raw camera frame (for PiP display)
      - Head position + hands-up gesture (for game logic)
      - All 17 COCO keypoints (for PiP skeleton overlay)
    """

    def __init__(self):
        # Frame lock — protects the raw camera frame
        self._frame_lock = threading.Lock()
        self._frame = None       # RGB numpy array (H, W, 3) or None
        self._frame_w = 0
        self._frame_h = 0

        # Pose lock — protects pose data + keypoints
        self._pose_lock = threading.Lock()
        self._head_x_norm = None
        self._both_hands_up = False
        self._keypoints = []     # list of (x_norm, y_norm) tuples for all 17 keypoints

        self.running = True

    def set_frame(self, frame, width, height):
        """Camera thread: store latest raw frame."""
        with self._frame_lock:
            self._frame = frame
            self._frame_w = width
            self._frame_h = height

    def get_frame(self):
        """Game thread: get a copy of the latest frame, or None."""
        with self._frame_lock:
            if self._frame is None:
                return None, 0, 0
            return self._frame.copy(), self._frame_w, self._frame_h

    def set_pose(self, head_x_norm, both_hands_up, keypoints):
        """Camera thread: store latest pose data + all keypoints."""
        with self._pose_lock:
            self._head_x_norm = head_x_norm
            self._both_hands_up = both_hands_up
            self._keypoints = keypoints

    def get_pose(self):
        """Game thread: get latest pose data + keypoints."""
        with self._pose_lock:
            return self._head_x_norm, self._both_hands_up, list(self._keypoints)

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════════
# Game state
# ═══════════════════════════════════════════════════════════════════════════
class GameState:
    def __init__(self, width, height):
        self.score = 0
        self.lives = INITIAL_LIVES
        self.game_start = None
        self.game_over = False
        self.game_over_time = None
        self.player_x = width / 2
        self.player_y = height - 60
        self.aliens = []
        self.alien_direction = 1
        self.alien_anim_timer = 0.0
        self.alien_anim_interval = 0.4
        self.bullets = []
        self.alien_bullets = []
        self.last_fire_time = 0.0
        self.popups = []
        self.stars = []
        self.fw = width
        self.fh = height
        self._init_done = False

    def init_world(self):
        self.aliens = []
        total = ALIEN_COLS * ALIEN_SPACING_X
        start_x = (self.fw - total) // 2 + ALIEN_SPACING_X // 2
        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                self.aliens.append(Alien(row, col,
                                          start_x + col * ALIEN_SPACING_X,
                                          80 + row * ALIEN_SPACING_Y))
        self.stars = []
        for _ in range(NUM_STARS):
            self.stars.append((
                random.randint(0, self.fw),
                random.randint(0, self.fh),
                random.randint(80, 220),
                random.choice([1, 1, 1, 2]),
            ))
        self._init_done = True

    def restart(self):
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
        self.last_fire_time = 0.0
        self._init_done = False


# ═══════════════════════════════════════════════════════════════════════════
# Camera callback — runs in GStreamer thread
# Fast: only extracts pose data. No rendering.
# ═══════════════════════════════════════════════════════════════════════════
def camera_callback(element, buffer, user_data):
    """Camera thread: extract pose data and raw frame, store in SharedState."""
    shared = user_data.shared
    pad = element.get_static_pad("src")
    fmt, width, height = get_caps_from_pad(pad)

    if not fmt or width is None or height is None:
        return Gst.FlowReturn.OK

    # --- Extract raw frame (for PiP display) ---
    try:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)
        if frame is not None:
            shared.set_frame(frame, width, height)
    except Exception:
        pass  # Non-fatal — PiP just won't show this frame

    # --- Extract pose data ---
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        head_x_norm = None
        both_hands_up = False
        all_keypoints = []  # All 17 keypoints, normalised to bbox

        for det in detections:
            if det.get_label() != "person":
                continue
            bbox = det.get_bbox()
            landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not landmarks:
                continue
            points = landmarks[0].get_points()
            if len(points) <= RIGHT_WRIST:
                continue

            # All 17 keypoints (normalised to bbox 0-1)
            all_keypoints = [
                (pt.x() * bbox.width() + bbox.xmin(),
                 pt.y() * bbox.height() + bbox.ymin())
                for pt in points
            ]

            # Head position (after mirroring for game logic)
            nose = points[NOSE]
            head_x_norm = nose.x() * bbox.width() + bbox.xmin()
            if MIRROR_X:
                head_x_norm = 1.0 - head_x_norm
            nose_y = (nose.y() * bbox.height() + bbox.ymin()) * height

            # Hands-up gesture
            lw_y = (points[LEFT_WRIST].y() * bbox.height() + bbox.ymin()) * height
            rw_y = (points[RIGHT_WRIST].y() * bbox.height() + bbox.ymin()) * height
            if lw_y < nose_y and rw_y < nose_y:
                both_hands_up = True
            break

        shared.set_pose(head_x_norm, both_hands_up, all_keypoints)
    except Exception as e:
        logger.debug("Pose extraction error: %s", e)

    return Gst.FlowReturn.OK


# ═══════════════════════════════════════════════════════════════════════════
# Callback class — wraps SharedState
# ═══════════════════════════════════════════════════════════════════════════
class SpaceInvadersCallback(app_callback_class):
    """Bridges the camera callback thread and the main game thread."""

    def __init__(self):
        super().__init__()
        # use_frame is False — we don't need the GStreamer display process;
        # pygame handles the display in the main thread.
        self.use_frame = False
        self.shared = SharedState()

    def stop(self):
        self.shared.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Main — pygame game loop
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = get_pipeline_parser()
    args, _ = parser.parse_known_args()

    # Init pygame
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    pygame.init()
    pygame.display.set_caption("Space Invaders — Hailo Edition")

    # Fullscreen display
    info = pygame.display.Info()
    win_w, win_h = info.current_w, info.current_h
    if win_w < 640 or win_h < 480:
        win_w, win_h = 1280, 720
    screen = pygame.display.set_mode((win_w, win_h), pygame.FULLSCREEN)
    clock = pygame.time.Clock()

    # Shared state + game state
    user_data = SpaceInvadersCallback()
    game = GameState(win_w, win_h)

    # Start GStreamer pipeline in a daemon thread
    gst_app = GStreamerPoseEstimationApp(camera_callback, user_data, parser)
    gst_thread = threading.Thread(
        target=lambda: (gst_app.run(), user_data.shared.stop()),
        daemon=True, name='CameraThread'
    )
    gst_thread.start()
    # Give the pipeline a moment to start
    time.sleep(1.0)

    # Pre-rendered star surface for twinkling effect
    def draw_stars(surf, game, now):
        for sx, sy, brightness, size in game.stars:
            twinkle = max(50, min(255, brightness + int(30 * math.sin(now * 2 + sx * 0.1))))
            col = (twinkle, twinkle, twinkle)
            if size == 1:
                surf.set_at((sx, sy), col)
            else:
                pygame.draw.circle(surf, col, (sx, sy), size)

    running = True
    while running and user_data.shared.running:
        # --- Events ---
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    game.restart()

        now = time.time()

        # --- Read pose from camera thread ---
        head_x_norm, both_hands_up, keypoints = user_data.shared.get_pose()

        # --- Lazy init ---
        if not game._init_done:
            game.game_start = now
            game.init_world()

        # --- Game over / restart ---
        if game.game_over:
            if game.game_over_time and (now - game.game_over_time) >= RESTART_DELAY:
                game.restart()
                clock.tick(FPS)
                continue
            # Render game over screen
            screen.fill(DARK_BG)
            draw_stars(screen, game, now)
            for alien in game.aliens:
                if alien.alive:
                    draw_alien(screen, alien)
            for b in game.bullets:
                draw_bullet(screen, b)
            for b in game.alien_bullets:
                draw_bullet(screen, b)
            draw_player_ship(screen, game.player_x, game.player_y)
            remaining = max(0.0, GAME_DURATION - (now - game.game_start)) if game.game_start else 0
            draw_hud(screen, game.score, game.lives, remaining, win_w)
            # PiP stays visible during game over
            if hasattr(game, '_cached_pip_frame') and game._cached_pip_frame is not None:
                draw_pip(screen, game._cached_pip_frame, game._cached_keypoints,
                         win_w, win_h, game.player_x, game.player_y)
            draw_game_over(screen, game.score, win_w, win_h)
            pygame.display.flip()
            clock.tick(FPS)
            continue

        # --- Time's up ---
        elapsed = now - game.game_start
        remaining = max(0.0, GAME_DURATION - elapsed)
        if remaining <= 0:
            game.game_over = True
            game.game_over_time = now
            clock.tick(FPS)
            continue

        # --- Respawn aliens ---
        if not any(a.alive for a in game.aliens):
            game.init_world()
            game.alien_direction = 1

        # --- Update ship from head ---
        if head_x_norm is not None:
            target_x = head_x_norm * win_w
            target_x = max(40, min(win_w - 40, target_x))
            game.player_x += (target_x - game.player_x) * 0.25

        # --- Fire ---
        cooldown = FIRE_COOLDOWN_HANDS_UP if both_hands_up else FIRE_COOLDOWN
        if now - game.last_fire_time >= cooldown:
            game.bullets.append(Bullet(game.player_x, game.player_y - 22))
            game.last_fire_time = now

        # --- Update aliens ---
        alive = [a for a in game.aliens if a.alive]
        if alive:
            if now - game.alien_anim_timer >= game.alien_anim_interval:
                for a in alive:
                    a.frame += 1
                game.alien_anim_timer = now

            speed_mult = 1.0 + (1.0 - len(alive) / (ALIEN_ROWS * ALIEN_COLS)) * 2.0
            speed = ALIEN_SPEED_BASE * speed_mult

            hit_edge = any(
                a.x + speed * game.alien_direction < 30 or
                a.x + speed * game.alien_direction > win_w - 30
                for a in alive
            )

            if hit_edge:
                game.alien_direction *= -1
                for a in alive:
                    a.y += ALIEN_DESCENT
                    if a.y >= game.player_y - 30:
                        game.game_over = True
                        game.game_over_time = now
                        break
            else:
                for a in alive:
                    a.x += speed * game.alien_direction

            if not game.game_over and random.random() < ALIEN_SHOOT_CHANCE:
                shooter = random.choice(alive)
                game.alien_bullets.append(Bullet(shooter.x, shooter.y + 18, is_alien=True))

        # --- Update bullets ---
        for b in game.bullets:
            b.y -= BULLET_SPEED
            if b.y < 0:
                b.alive = False
        for b in game.alien_bullets:
            b.y += BULLET_SPEED * 0.6
            if b.y > win_h:
                b.alive = False

        # --- Collisions: player bullets vs aliens ---
        for b in game.bullets:
            if not b.alive:
                continue
            for a in game.aliens:
                if not a.alive:
                    continue
                if math.hypot(b.x - a.x, b.y - a.y) < 22:
                    b.alive = False
                    a.alive = False
                    game.score += a.points
                    game.popups.append(Popup(f"+{a.points}", a.x, a.y, a.color))
                    break

        # --- Collisions: alien bullets vs player ---
        for b in game.alien_bullets:
            if not b.alive:
                continue
            if math.hypot(b.x - game.player_x, b.y - game.player_y) < 20:
                b.alive = False
                game.lives -= 1
                game.popups.append(Popup("-1 LIFE", game.player_x, game.player_y - 40, RED))
                if game.lives <= 0:
                    game.game_over = True
                    game.game_over_time = now
                    break

        # --- Cleanup ---
        game.bullets = [b for b in game.bullets if b.alive]
        game.alien_bullets = [b for b in game.alien_bullets if b.alive]

        # --- Render ---
        screen.fill(DARK_BG)
        draw_stars(screen, game, now)

        for alien in game.aliens:
            if alien.alive:
                draw_alien(screen, alien)

        for b in game.bullets:
            draw_bullet(screen, b)
        for b in game.alien_bullets:
            draw_bullet(screen, b)

        draw_player_ship(screen, game.player_x, game.player_y)

        # Ship tracking ring
        if game.player_x > 0:
            pygame.draw.circle(screen, (0, 80, 0),
                               (int(game.player_x), int(game.player_y)), 30, 1)

        # Popups
        game.popups = [p for p in game.popups if p.alive()]
        for p in game.popups:
            alpha = p.alpha()
            color = tuple(int(c * alpha) for c in p.colour)
            text = pygame.font.Font(None, 32).render(p.text, True, color)
            screen.blit(text, (int(p.x) - 25, p.current_y()))

        draw_hud(screen, game.score, game.lives, remaining, win_w)

        # --- PiP camera preview (drawn last, on top) ---
        # Throttle: only get a new frame every PIP_UPDATE_INTERVAL seconds
        if not hasattr(game, '_last_pip_time') or (now - game._last_pip_time) >= PIP_UPDATE_INTERVAL:
            game._last_pip_time = now
            pip_frame, _, _ = user_data.shared.get_frame()
            game._cached_pip_frame = pip_frame
            game._cached_keypoints = keypoints
        # Draw using cached data
        if hasattr(game, '_cached_pip_frame') and game._cached_pip_frame is not None:
            draw_pip(screen, game._cached_pip_frame, game._cached_keypoints,
                     win_w, win_h, game.player_x, game.player_y)

        pygame.display.flip()
        clock.tick(FPS)

    # --- Cleanup ---
    user_data.shared.stop()
    try:
        gst_app.pipeline.set_state(Gst.State.NULL)
    except Exception:
        pass
    gst_thread.join(timeout=2.0)
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
