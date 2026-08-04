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

# --- Keypoint indices (COCO 17) ---
NOSE = 0
LEFT_WRIST = 9
RIGHT_WRIST = 10

# --- Game constants ---
GAME_DURATION = 90          # seconds
ALIEN_ROWS = 4
ALIEN_COLS = 6
ALIEN_SPACING_X = 70
ALIEN_SPACING_Y = 55
ALIEN_DESCENT = 30
ALIEN_SPEED_BASE = 1.5

BULLET_SPEED = 18
FIRE_COOLDOWN = 0.25
FIRE_COOLDOWN_HANDS_UP = 0.12

INITIAL_LIVES = 3
ALIEN_SHOOT_CHANCE = 0.02

# Alien rows: top row = most points
ALIEN_POINTS = [40, 30, 20, 10]
ALIEN_COLORS = [
    (255, 60, 60),      # red (top)
    (255, 160, 40),     # orange
    (60, 220, 60),      # green
    (60, 180, 255),     # blue (bottom)
]

# --- Tunables ---
MIRROR_X = True            # Mirror camera input so head left = ship left
RENDER_EVERY = 1           # Render every Nth frame (1 = always)
NUM_STARS = 80             # Background stars
RESTART_DELAY = 5          # Seconds before game over auto-restarts

# --- Colours (RGB for OpenCV drawing) ---
BGR_BLACK = (0, 0, 0)
BGR_DARK_BG = (8, 8, 24)
BGR_WHITE = (255, 255, 255)
BGR_GREEN = (0, 220, 0)
BGR_RED = (60, 60, 255)
BGR_YELLOW = (0, 230, 255)
BGR_HUD_BG = (30, 30, 30)


# ═══════════════════════════════════════════════════════════════════════════
# Game objects
# ═══════════════════════════════════════════════════════════════════════════
class Alien:
    """One alien in the formation."""
    def __init__(self, row, col, x, y):
        self.row = row
        self.col = col
        self.x = x
        self.y = y
        self.alive = True
        self.points = ALIEN_POINTS[min(row, len(ALIEN_POINTS) - 1)]
        self.color = ALIEN_COLORS[min(row, len(ALIEN_COLORS) - 1)]
        self.frame = 0  # animation frame for legs/antennae


class Bullet:
    """Player or alien bullet."""
    def __init__(self, x, y, is_alien=False):
        self.x = x
        self.y = y
        self.is_alien = is_alien
        self.alive = True


class Popup:
    """Floating score text."""
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
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════
def draw_alien(output, alien):
    """Draw a classic space invader alien with animated legs/antennae."""
    x, y = int(alien.x), int(alien.y)
    size = 18
    color = alien.color

    # Body — hexagonal
    body_pts = np.array([
        [x - size, y - size // 2],
        [x + size, y - size // 2],
        [x + size, y + size // 2],
        [x - size, y + size // 2],
    ], np.int32)
    cv2.fillPoly(output, [body_pts], color, lineType=cv2.LINE_AA)

    # Eyes
    eye_y = y - 3
    cv2.circle(output, (x - 7, eye_y), 4, BGR_BLACK, -1, cv2.LINE_AA)
    cv2.circle(output, (x + 7, eye_y), 4, BGR_BLACK, -1, cv2.LINE_AA)
    cv2.circle(output, (x - 6, eye_y - 1), 1, BGR_WHITE, -1, cv2.LINE_AA)
    cv2.circle(output, (x + 8, eye_y - 1), 1, BGR_WHITE, -1, cv2.LINE_AA)

    # Antennae (animated)
    anim_offset = 4 if alien.frame % 2 == 0 else -4
    cv2.line(output, (x - 8, y - size // 2), (x - 14, y - size // 2 - 8 + anim_offset),
             color, 2, cv2.LINE_AA)
    cv2.line(output, (x + 8, y - size // 2), (x + 14, y - size // 2 - 8 + anim_offset),
             color, 2, cv2.LINE_AA)

    # Legs (alternating animation)
    leg_y = y + size // 2
    if alien.frame % 2 == 0:
        cv2.line(output, (x - 10, leg_y), (x - 14, leg_y + 8), color, 2, cv2.LINE_AA)
        cv2.line(output, (x + 10, leg_y), (x + 14, leg_y + 8), color, 2, cv2.LINE_AA)
    else:
        cv2.line(output, (x - 10, leg_y), (x - 6, leg_y + 8), color, 2, cv2.LINE_AA)
        cv2.line(output, (x + 10, leg_y), (x + 6, leg_y + 8), color, 2, cv2.LINE_AA)

    # Outline
    cv2.polylines(output, [body_pts], True, BGR_WHITE, 1, cv2.LINE_AA)


def draw_player_ship(output, x, y):
    """Draw the player's green triangle ship with engine glow."""
    x, y = int(x), int(y)
    pts = np.array([
        [x, y - 22],          # nose
        [x - 18, y + 12],     # bottom left
        [x + 18, y + 12],     # bottom right
    ], np.int32)

    # Glow
    cv2.fillPoly(output, [pts + 2], (40, 120, 40), cv2.LINE_AA)
    # Body
    cv2.fillPoly(output, [pts], BGR_GREEN, cv2.LINE_AA)
    # Outline
    cv2.polylines(output, [pts], True, (100, 255, 100), 2, cv2.LINE_AA)
    # Cockpit
    cv2.circle(output, (x, y - 4), 5, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(output, (x, y - 4), 5, (200, 255, 200), 1, cv2.LINE_AA)


def draw_bullet(output, bullet):
    """Draw a bullet (player = green line, alien = red diamond)."""
    x, y = int(bullet.x), int(bullet.y)
    if bullet.is_alien:
        pts = np.array([[x, y - 6], [x + 3, y], [x, y + 6], [x - 3, y]], np.int32)
        cv2.fillPoly(output, [pts], BGR_RED, cv2.LINE_AA)
        cv2.polylines(output, [pts], True, (150, 150, 255), 1, cv2.LINE_AA)
    else:
        cv2.line(output, (x, y - 6), (x, y + 6), BGR_GREEN, 3, cv2.LINE_AA)
        cv2.line(output, (x, y - 6), (x, y + 6), (200, 255, 200), 1, cv2.LINE_AA)


def draw_hud(output, score, lives, remaining, width):
    """Draw score, lives, and countdown timer bar."""
    bar_h = 50
    # Dark background bar
    overlay = output[:bar_h, :, :].copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), BGR_HUD_BG, -1)
    output[:bar_h, :, :] = cv2.addWeighted(overlay, 0.8, output[:bar_h, :, :], 0.2, 0)

    # Progress bar
    frac = max(0.0, min(1.0, remaining / GAME_DURATION))
    bar_w = int((width - 260) * frac)
    bar_color = (80, 220, 80) if remaining > 20 else (80, 80, 255)
    cv2.rectangle(output, (130, 10), (130 + bar_w, 34), bar_color, -1, cv2.LINE_AA)
    cv2.rectangle(output, (130, 10), (width - 130, 34), (180, 180, 180), 1, cv2.LINE_AA)

    # Time text
    mins = int(remaining) // 60
    secs = int(remaining) % 60
    cv2.putText(output, f"{mins}:{secs:02d}", (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, BGR_WHITE, 2, cv2.LINE_AA)

    # Score
    cv2.putText(output, f"SCORE: {score}", (width - 200, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BGR_GREEN, 2, cv2.LINE_AA)

    # Lives (mini ship icons)
    for i in range(lives):
        lx = width - 250 + i * 30
        pts = np.array([[lx, 18], [lx - 8, 32], [lx + 8, 32]], np.int32)
        cv2.fillPoly(output, [pts], BGR_GREEN, cv2.LINE_AA)


def draw_game_over(output, score, width, height):
    """Draw the game over screen with final score."""
    # Dark overlay
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)

    cv2.putText(output, "GAME OVER", (width // 2 - 160, height // 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (60, 220, 255), 4, cv2.LINE_AA)
    cv2.putText(output, f"FINAL SCORE: {score}", (width // 2 - 140, height // 4 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, BGR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(output, "Restarting...", (width // 2 - 100, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
# Callback class
# ═══════════════════════════════════════════════════════════════════════════
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
        self.player_y = 0.0

        # Aliens
        self.aliens = []
        self.alien_direction = 1
        self.alien_anim_timer = 0.0
        self.alien_anim_interval = 0.4

        # Bullets
        self.bullets = []
        self.alien_bullets = []
        self.last_fire_time = 0.0

        # Popups
        self.popups = []

        # Stars
        self.stars = []

        # Frame dimensions
        self.fw = 0
        self.fh = 0

        # Pre-allocated render buffer
        self._render_buffer = None
        self._render_buffer_shape = None

        # Frame counter for render-skipping
        self._frame_counter = 0

    def set_frame(self, frame):
        """Drain stale frames so display always shows the latest."""
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
        total_width = ALIEN_COLS * ALIEN_SPACING_X
        start_x = (self.fw - total_width) // 2 + ALIEN_SPACING_X // 2
        start_y = 80
        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                self.aliens.append(Alien(row, col,
                                          start_x + col * ALIEN_SPACING_X,
                                          start_y + row * ALIEN_SPACING_Y))

    def init_stars(self):
        """Create random background stars."""
        self.stars = []
        for _ in range(NUM_STARS):
            self.stars.append((
                random.randint(0, self.fw),
                random.randint(0, self.fh),
                random.randint(80, 220),
                random.choice([1, 1, 1, 2]),
            ))

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
        self.last_fire_time = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Callback function
# ═══════════════════════════════════════════════════════════════════════════
def app_callback(element, buffer, user_data):
    pad = element.get_static_pad("src")
    fmt, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and fmt and width and height:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    if frame is None:
        return Gst.FlowReturn.OK

    now = time.time()
    user_data._frame_counter += 1

    # --- Lazy init ---
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

    # --- Game over screen (always render — static) ---
    if user_data.game_over:
        if user_data.game_over_time and (now - user_data.game_over_time) >= RESTART_DELAY:
            user_data.restart()
            return Gst.FlowReturn.OK
        _render(user_data, width, height, now, game_over=True)
        return Gst.FlowReturn.OK

    # --- Time's up ---
    if remaining <= 0:
        user_data.game_over = True
        user_data.game_over_time = now
        return Gst.FlowReturn.OK

    # --- Respawn aliens if all destroyed ---
    if not any(a.alive for a in user_data.aliens):
        user_data.init_aliens()
        user_data.alien_direction = 1

    # --- Extract pose: head X + hands-up gesture ---
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    player_head_x = None
    both_hands_up = False

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

        # Head tracking (nose)
        nose = points[NOSE]
        nose_x_norm = nose.x() * bbox.width() + bbox.xmin()
        if MIRROR_X:
            nose_x_norm = 1.0 - nose_x_norm
        player_head_x = nose_x_norm * width
        nose_y = (nose.y() * bbox.height() + bbox.ymin()) * height

        # Hands-up: both wrists above nose
        lw_y = (points[LEFT_WRIST].y() * bbox.height() + bbox.ymin()) * height
        rw_y = (points[RIGHT_WRIST].y() * bbox.height() + bbox.ymin()) * height
        if lw_y < nose_y and rw_y < nose_y:
            both_hands_up = True
        break

    # --- Update ship position (smoothed) ---
    if player_head_x is not None:
        target_x = np.clip(player_head_x, 40, width - 40)
        user_data.player_x += (target_x - user_data.player_x) * 0.25

    # --- Fire bullets ---
    cooldown = FIRE_COOLDOWN_HANDS_UP if both_hands_up else FIRE_COOLDOWN
    if now - user_data.last_fire_time >= cooldown:
        user_data.bullets.append(
            Bullet(user_data.player_x, user_data.player_y - 22)
        )
        user_data.last_fire_time = now

    # --- Update aliens ---
    alien_alive = [a for a in user_data.aliens if a.alive]
    if alien_alive:
        # Animation tick
        if now - user_data.alien_anim_timer >= user_data.alien_anim_interval:
            for a in alien_alive:
                a.frame += 1
            user_data.alien_anim_timer = now

        # Speed scales up as aliens are destroyed
        speed_mult = 1.0 + (1.0 - len(alien_alive) / (ALIEN_ROWS * ALIEN_COLS)) * 2.0
        alien_speed = ALIEN_SPEED_BASE * speed_mult

        # Edge check
        hit_edge = any(
            a.x + alien_speed * user_data.alien_direction < 30 or
            a.x + alien_speed * user_data.alien_direction > width - 30
            for a in alien_alive
        )

        if hit_edge:
            user_data.alien_direction *= -1
            for a in alien_alive:
                a.y += ALIEN_DESCENT
                if a.y >= user_data.player_y - 30:
                    # Aliens reached the player
                    user_data.game_over = True
                    user_data.game_over_time = now
                    return Gst.FlowReturn.OK
        else:
            for a in alien_alive:
                a.x += alien_speed * user_data.alien_direction

        # Alien shoots
        if random.random() < ALIEN_SHOOT_CHANCE:
            shooter = random.choice(alien_alive)
            user_data.alien_bullets.append(
                Bullet(shooter.x, shooter.y + 18, is_alien=True)
            )

    # --- Update bullets ---
    for b in user_data.bullets:
        b.y -= BULLET_SPEED
        if b.y < 0:
            b.alive = False
    for b in user_data.alien_bullets:
        b.y += BULLET_SPEED * 0.6
        if b.y > height:
            b.alive = False

    # --- Collisions: player bullets vs aliens ---
    for b in user_data.bullets:
        if not b.alive:
            continue
        for a in user_data.aliens:
            if not a.alive:
                continue
            if math.hypot(b.x - a.x, b.y - a.y) < 22:
                b.alive = False
                a.alive = False
                user_data.score += a.points
                user_data.popups.append(
                    Popup(f"+{a.points}", a.x, a.y, a.color)
                )
                logger.info("Hit alien (+%d) — score: %d", a.points, user_data.score)
                break

    # --- Collisions: alien bullets vs player ---
    for b in user_data.alien_bullets:
        if not b.alive:
            continue
        if math.hypot(b.x - user_data.player_x, b.y - user_data.player_y) < 20:
            b.alive = False
            user_data.lives -= 1
            user_data.popups.append(
                Popup("-1 LIFE", user_data.player_x, user_data.player_y - 40, BGR_RED)
            )
            logger.info("Player hit! Lives: %d", user_data.lives)
            if user_data.lives <= 0:
                user_data.game_over = True
                user_data.game_over_time = now
                return Gst.FlowReturn.OK

    # --- Cleanup ---
    user_data.bullets = [b for b in user_data.bullets if b.alive]
    user_data.alien_bullets = [b for b in user_data.alien_bullets if b.alive]

    # --- Frame-skip: only render every RENDER_EVERY frames ---
    if RENDER_EVERY > 1 and (user_data._frame_counter % RENDER_EVERY) != 0:
        return Gst.FlowReturn.OK

    _render(user_data, width, height, now, game_over=False)
    return Gst.FlowReturn.OK


# ═══════════════════════════════════════════════════════════════════════════
# Rendering (with pre-allocated buffer)
# ═══════════════════════════════════════════════════════════════════════════
def _render(user_data, width, height, now, game_over=False):
    """Render the full game frame into a pre-allocated buffer."""
    # Get/create reusable buffer
    if (user_data._render_buffer is None or
            user_data._render_buffer_shape != (height, width)):
        user_data._render_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        user_data._render_buffer_shape = (height, width)
    output = user_data._render_buffer
    output[:] = BGR_DARK_BG

    # Twinkling stars
    for sx, sy, brightness, size in user_data.stars:
        twinkle = max(50, min(255, brightness + int(30 * math.sin(now * 2 + sx * 0.1))))
        cv2.circle(output, (sx, sy), size, (twinkle, twinkle, twinkle), -1, cv2.LINE_AA)

    # Aliens
    for alien in user_data.aliens:
        if alien.alive:
            draw_alien(output, alien)

    # Bullets
    for b in user_data.bullets:
        draw_bullet(output, b)
    for b in user_data.alien_bullets:
        draw_bullet(output, b)

    # Player ship
    draw_player_ship(output, user_data.player_x, user_data.player_y)

    # Ship tracking indicator
    if user_data.player_x > 0:
        cv2.circle(output, (int(user_data.player_x), int(user_data.player_y)),
                   30, (0, 80, 0), 1, cv2.LINE_AA)

    # Popups
    user_data.popups = [p for p in user_data.popups if p.alive()]
    for p in user_data.popups:
        alpha = p.alpha()
        color = tuple(int(c * alpha) for c in p.colour)
        cv2.putText(output, p.text, (int(p.x) - 25, p.current_y()),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # HUD
    if not game_over:
        remaining = max(0.0, GAME_DURATION - (now - user_data.game_start))
        draw_hud(output, user_data.score, user_data.lives, remaining, width)
    else:
        draw_game_over(output, user_data.score, width, height)

    # Convert in-place and push to display
    cv2.cvtColor(output, cv2.COLOR_RGB2BGR, dst=output)
    user_data.set_frame(output)


# ═══════════════════════════════════════════════════════════════════════════
# App class
# ═══════════════════════════════════════════════════════════════════════════
class SpaceInvadersApp(GStreamerPoseEstimationApp):
    """Pose-estimation pipeline with Space Invaders overlay."""

    def __init__(self, app_callback, user_data, parser=None):
        super().__init__(app_callback, user_data, parser)
        # Force use_frame after parent constructor
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
