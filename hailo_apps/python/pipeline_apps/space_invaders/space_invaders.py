"""Space Invaders — steer with your head, raise hands to fire, defend Earth from aliens.

Architecture:
  - Camera thread (GStreamer callback): extracts pose data, stores in SharedState.
  - Game thread (main): reads pose from SharedState, runs game loop at fixed FPS,
    renders, and pushes the output frame to the display.
  - SharedState uses threading.Lock to protect the image frame and pose data
    for safe cross-thread access.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import hailo
import math
import numpy as np
import random
import threading
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
GAME_DURATION = 90
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
ALIEN_POINTS = [40, 30, 20, 10]
ALIEN_COLORS = [
    (255, 60, 60), (255, 160, 40), (60, 220, 60), (60, 180, 255),
]
NUM_STARS = 80
RESTART_DELAY = 5
GAME_FPS = 60

# --- Tunables ---
MIRROR_X = True

# --- Colours (BGR) ---
BGR_BLACK = (0, 0, 0)
BGR_DARK_BG = (8, 8, 24)
BGR_WHITE = (255, 255, 255)
BGR_GREEN = (0, 220, 0)
BGR_RED = (60, 60, 255)
BGR_HUD_BG = (30, 30, 30)


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
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════
def draw_alien(output, alien):
    x, y = int(alien.x), int(alien.y)
    size = 18
    color = alien.color
    body_pts = np.array([
        [x - size, y - size // 2], [x + size, y - size // 2],
        [x + size, y + size // 2], [x - size, y + size // 2],
    ], np.int32)
    cv2.fillPoly(output, [body_pts], color, lineType=cv2.LINE_AA)
    cv2.circle(output, (x - 7, y - 3), 4, BGR_BLACK, -1, cv2.LINE_AA)
    cv2.circle(output, (x + 7, y - 3), 4, BGR_BLACK, -1, cv2.LINE_AA)
    cv2.circle(output, (x - 6, y - 4), 1, BGR_WHITE, -1, cv2.LINE_AA)
    cv2.circle(output, (x + 8, y - 4), 1, BGR_WHITE, -1, cv2.LINE_AA)
    anim = 4 if alien.frame % 2 == 0 else -4
    cv2.line(output, (x - 8, y - size // 2), (x - 14, y - size // 2 - 8 + anim), color, 2, cv2.LINE_AA)
    cv2.line(output, (x + 8, y - size // 2), (x + 14, y - size // 2 - 8 + anim), color, 2, cv2.LINE_AA)
    if alien.frame % 2 == 0:
        cv2.line(output, (x - 10, y + size // 2), (x - 14, y + size // 2 + 8), color, 2, cv2.LINE_AA)
        cv2.line(output, (x + 10, y + size // 2), (x + 14, y + size // 2 + 8), color, 2, cv2.LINE_AA)
    else:
        cv2.line(output, (x - 10, y + size // 2), (x - 6, y + size // 2 + 8), color, 2, cv2.LINE_AA)
        cv2.line(output, (x + 10, y + size // 2), (x + 6, y + size // 2 + 8), color, 2, cv2.LINE_AA)
    cv2.polylines(output, [body_pts], True, BGR_WHITE, 1, cv2.LINE_AA)


def draw_player_ship(output, x, y):
    x, y = int(x), int(y)
    pts = np.array([[x, y - 22], [x - 18, y + 12], [x + 18, y + 12]], np.int32)
    cv2.fillPoly(output, [pts + 2], (40, 120, 40), cv2.LINE_AA)
    cv2.fillPoly(output, [pts], BGR_GREEN, cv2.LINE_AA)
    cv2.polylines(output, [pts], True, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.circle(output, (x, y - 4), 5, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(output, (x, y - 4), 5, (200, 255, 200), 1, cv2.LINE_AA)


def draw_bullet(output, bullet):
    x, y = int(bullet.x), int(bullet.y)
    if bullet.is_alien:
        pts = np.array([[x, y - 6], [x + 3, y], [x, y + 6], [x - 3, y]], np.int32)
        cv2.fillPoly(output, [pts], BGR_RED, cv2.LINE_AA)
        cv2.polylines(output, [pts], True, (150, 150, 255), 1, cv2.LINE_AA)
    else:
        cv2.line(output, (x, y - 6), (x, y + 6), BGR_GREEN, 3, cv2.LINE_AA)
        cv2.line(output, (x, y - 6), (x, y + 6), (200, 255, 200), 1, cv2.LINE_AA)


def draw_hud(output, score, lives, remaining, width):
    bar_h = 50
    output[:bar_h, :, :] = BGR_HUD_BG
    frac = max(0.0, min(1.0, remaining / GAME_DURATION))
    bar_w = int((width - 260) * frac)
    bar_color = (80, 220, 80) if remaining > 20 else (80, 80, 255)
    cv2.rectangle(output, (130, 10), (130 + bar_w, 34), bar_color, -1, cv2.LINE_AA)
    cv2.rectangle(output, (130, 10), (width - 130, 34), (180, 180, 180), 1, cv2.LINE_AA)
    mins = int(remaining) // 60
    secs = int(remaining) % 60
    cv2.putText(output, f"{mins}:{secs:02d}", (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BGR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(output, f"SCORE: {score}", (width - 200, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BGR_GREEN, 2, cv2.LINE_AA)
    for i in range(lives):
        lx = width - 250 + i * 30
        pts = np.array([[lx, 18], [lx - 8, 32], [lx + 8, 32]], np.int32)
        cv2.fillPoly(output, [pts], BGR_GREEN, cv2.LINE_AA)


def draw_game_over(output, score, width, height):
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)
    cv2.putText(output, "GAME OVER", (width // 2 - 160, height // 4), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (60, 220, 255), 4, cv2.LINE_AA)
    cv2.putText(output, f"FINAL SCORE: {score}", (width // 2 - 140, height // 4 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BGR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(output, "Restarting...", (width // 2 - 100, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
# Thread-safe shared state between camera and game threads
# ═══════════════════════════════════════════════════════════════════════════
class SharedState:
    """Thread-safe container for data shared between the camera and game threads."""

    def __init__(self):
        # Protects the image frame
        self._frame_lock = threading.Lock()
        self._frame = None
        self._width = 0
        self._height = 0

        # Protects pose data
        self._pose_lock = threading.Lock()
        self._head_x_norm = None
        self._both_hands_up = False

        # Running flag (simple bool — atomic in CPython)
        self.running = True

    def set_frame(self, frame, width, height):
        """Camera thread: store latest frame (RGB numpy)."""
        with self._frame_lock:
            self._frame = frame
            self._width = width
            self._height = height

    def get_frame(self):
        """Game thread: get a copy of the latest frame, or None."""
        with self._frame_lock:
            if self._frame is None:
                return None, 0, 0
            return self._frame.copy(), self._width, self._height

    def set_pose(self, head_x_norm, both_hands_up):
        """Camera thread: store latest pose data."""
        with self._pose_lock:
            self._head_x_norm = head_x_norm
            self._both_hands_up = both_hands_up

    def get_pose(self):
        """Game thread: get latest pose data."""
        with self._pose_lock:
            return self._head_x_norm, self._both_hands_up

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════════
# Game state (game thread only — no lock needed)
# ═══════════════════════════════════════════════════════════════════════════
class GameState:
    def __init__(self):
        self.score = 0
        self.lives = INITIAL_LIVES
        self.game_start = None
        self.game_over = False
        self.game_over_time = None
        self.player_x = 0.0
        self.player_y = 0.0
        self.aliens = []
        self.alien_direction = 1
        self.alien_anim_timer = 0.0
        self.alien_anim_interval = 0.4
        self.bullets = []
        self.alien_bullets = []
        self.last_fire_time = 0.0
        self.popups = []
        self.stars = []
        self.fw = 1280
        self.fh = 720
        self._render_buffer = None
        self._render_buffer_shape = None

    def init_aliens(self):
        self.aliens = []
        total = ALIEN_COLS * ALIEN_SPACING_X
        start_x = (self.fw - total) // 2 + ALIEN_SPACING_X // 2
        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                self.aliens.append(Alien(row, col,
                                          start_x + col * ALIEN_SPACING_X,
                                          80 + row * ALIEN_SPACING_Y))

    def init_stars(self):
        self.stars = []
        for _ in range(NUM_STARS):
            self.stars.append((
                random.randint(0, self.fw),
                random.randint(0, self.fh),
                random.randint(80, 220),
                random.choice([1, 1, 1, 2]),
            ))

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


# ═══════════════════════════════════════════════════════════════════════════
# Camera callback — runs in GStreamer thread
# Fast: only extracts pose data and stores it. No rendering.
# ═══════════════════════════════════════════════════════════════════════════
def camera_callback(element, buffer, user_data):
    """Camera thread: extract pose data and store in SharedState."""
    shared = user_data.shared
    pad = element.get_static_pad("src")
    fmt, width, height = get_caps_from_pad(pad)

    # Store raw frame (optional — game can use it if needed)
    if user_data.use_frame and fmt and width and height:
        try:
            frame = get_numpy_from_buffer(buffer, fmt, width, height)
            if frame is not None:
                shared.set_frame(frame, width, height)
        except Exception as e:
            logger.debug("Frame extraction error: %s", e)

    # Extract pose data
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        head_x_norm = None
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

            nose = points[NOSE]
            head_x_norm = nose.x() * bbox.width() + bbox.xmin()
            if MIRROR_X:
                head_x_norm = 1.0 - head_x_norm
            nose_y = (nose.y() * bbox.height() + bbox.ymin()) * height

            lw_y = (points[LEFT_WRIST].y() * bbox.height() + bbox.ymin()) * height
            rw_y = (points[RIGHT_WRIST].y() * bbox.height() + bbox.ymin()) * height
            if lw_y < nose_y and rw_y < nose_y:
                both_hands_up = True
            break

        shared.set_pose(head_x_norm, both_hands_up)
    except Exception as e:
        logger.debug("Pose extraction error: %s", e)

    return Gst.FlowReturn.OK


# ═══════════════════════════════════════════════════════════════════════════
# Game thread — independent fixed-FPS loop
# ═══════════════════════════════════════════════════════════════════════════
def game_loop(user_data):
    """Main game loop running at fixed FPS, independent of camera."""
    game = GameState()
    shared = user_data.shared
    frame_dt = 1.0 / GAME_FPS

    logger.info("Game thread started (target %d FPS)", GAME_FPS)

    while shared.running:
        loop_start = time.perf_counter()
        now = time.time()

        # Lazy init
        if game.game_start is None:
            game.game_start = now
            game.player_x = game.fw / 2
            game.player_y = game.fh - 60
            game.init_aliens()
            game.init_stars()

        # Read pose from camera thread
        head_x_norm, both_hands_up = shared.get_pose()

        # Game over / restart
        if game.game_over:
            if game.game_over_time and (now - game.game_over_time) >= RESTART_DELAY:
                game.restart()
                continue
            render_frame(game, now, game_over=True, user_data=user_data)
            _sleep_to_fps(loop_start, frame_dt)
            continue

        # Time's up
        elapsed = now - game.game_start
        remaining = max(0.0, GAME_DURATION - elapsed)
        if remaining <= 0:
            game.game_over = True
            game.game_over_time = now
            _sleep_to_fps(loop_start, frame_dt)
            continue

        # Respawn aliens
        if not any(a.alive for a in game.aliens):
            game.init_aliens()
            game.alien_direction = 1

        # Update ship from head
        if head_x_norm is not None:
            target_x = np.clip(head_x_norm * game.fw, 40, game.fw - 40)
            game.player_x += (target_x - game.player_x) * 0.25

        # Fire
        cooldown = FIRE_COOLDOWN_HANDS_UP if both_hands_up else FIRE_COOLDOWN
        if now - game.last_fire_time >= cooldown:
            game.bullets.append(Bullet(game.player_x, game.player_y - 22))
            game.last_fire_time = now

        # Update aliens
        _update_aliens(game, now)

        # Update bullets
        for b in game.bullets:
            b.y -= BULLET_SPEED
            if b.y < 0:
                b.alive = False
        for b in game.alien_bullets:
            b.y += BULLET_SPEED * 0.6
            if b.y > game.fh:
                b.alive = False

        # Collisions
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

        for b in game.alien_bullets:
            if not b.alive:
                continue
            if math.hypot(b.x - game.player_x, b.y - game.player_y) < 20:
                b.alive = False
                game.lives -= 1
                game.popups.append(Popup("-1 LIFE", game.player_x, game.player_y - 40, BGR_RED))
                if game.lives <= 0:
                    game.game_over = True
                    game.game_over_time = time.time()

        # Cleanup
        game.bullets = [b for b in game.bullets if b.alive]
        game.alien_bullets = [b for b in game.alien_bullets if b.alive]

        # Render
        render_frame(game, now, game_over=False, user_data=user_data)

        # Maintain FPS
        _sleep_to_fps(loop_start, frame_dt)


def _sleep_to_fps(loop_start, frame_dt):
    elapsed = time.perf_counter() - loop_start
    sleep_time = frame_dt - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)


def _update_aliens(game, now):
    alive = [a for a in game.aliens if a.alive]
    if not alive:
        return
    if now - game.alien_anim_timer >= game.alien_anim_interval:
        for a in alive:
            a.frame += 1
        game.alien_anim_timer = now

    speed_mult = 1.0 + (1.0 - len(alive) / (ALIEN_ROWS * ALIEN_COLS)) * 2.0
    speed = ALIEN_SPEED_BASE * speed_mult

    hit_edge = any(
        a.x + speed * game.alien_direction < 30 or
        a.x + speed * game.alien_direction > game.fw - 30
        for a in alive
    )

    if hit_edge:
        game.alien_direction *= -1
        for a in alive:
            a.y += ALIEN_DESCENT
            if a.y >= game.player_y - 30:
                game.game_over = True
                game.game_over_time = time.time()
                return
    else:
        for a in alive:
            a.x += speed * game.alien_direction

    if random.random() < ALIEN_SHOOT_CHANCE:
        shooter = random.choice(alive)
        game.alien_bullets.append(Bullet(shooter.x, shooter.y + 18, is_alien=True))


def render_frame(game, now, game_over, user_data):
    """Render the game and push to display via user_data.set_frame."""
    h, w = game.fh, game.fw
    if (game._render_buffer is None or
            game._render_buffer_shape != (h, w)):
        game._render_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        game._render_buffer_shape = (h, w)
    output = game._render_buffer
    output[:] = BGR_DARK_BG

    for sx, sy, brightness, size in game.stars:
        twinkle = max(50, min(255, brightness + int(30 * math.sin(now * 2 + sx * 0.1))))
        cv2.circle(output, (sx, sy), size, (twinkle, twinkle, twinkle), -1, cv2.LINE_AA)

    for alien in game.aliens:
        if alien.alive:
            draw_alien(output, alien)

    for b in game.bullets:
        draw_bullet(output, b)
    for b in game.alien_bullets:
        draw_bullet(output, b)

    draw_player_ship(output, game.player_x, game.player_y)

    if game.player_x > 0:
        cv2.circle(output, (int(game.player_x), int(game.player_y)),
                   30, (0, 80, 0), 1, cv2.LINE_AA)

    game.popups = [p for p in game.popups if p.alive()]
    for p in game.popups:
        alpha = p.alpha()
        color = tuple(int(c * alpha) for c in p.colour)
        cv2.putText(output, p.text, (int(p.x) - 25, p.current_y()),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    if not game_over:
        remaining = max(0.0, GAME_DURATION - (now - game.game_start))
        draw_hud(output, game.score, game.lives, remaining, w)
    else:
        draw_game_over(output, game.score, w, h)

    # Convert in-place to BGR and push to display
    cv2.cvtColor(output, cv2.COLOR_RGB2BGR, dst=output)
    user_data.set_frame(output)


# ═══════════════════════════════════════════════════════════════════════════
# Callback class — wraps SharedState
# ═══════════════════════════════════════════════════════════════════════════
class SpaceInvadersCallback(app_callback_class):
    """Bridges the camera callback thread and the game thread."""

    def __init__(self):
        super().__init__()
        self.use_frame = True
        self.shared = SharedState()

    def set_frame(self, frame):
        """Push a COPY of the frame to the display queue (thread-safe)."""
        frame_copy = frame.copy()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break
        try:
            self.frame_queue.put_nowait(frame_copy)
        except Exception:
            pass

    def stop(self):
        self.shared.stop()


# ═══════════════════════════════════════════════════════════════════════════
# App — runs pipeline in a background thread, game in main thread
# ═══════════════════════════════════════════════════════════════════════════
class SpaceInvadersApp:
    """Runs the GStreamer pose pipeline in a daemon thread, game in main thread."""

    def __init__(self, callback, user_data, parser=None):
        self._gst_app = GStreamerPoseEstimationApp(callback, user_data, parser)
        self._user_data = user_data
        self._game_thread = None

    def run(self):
        # Start game thread (daemon so it dies with the process)
        self._game_thread = threading.Thread(
            target=game_loop, args=(self._user_data,),
            daemon=True, name='GameThread'
        )
        self._game_thread.start()

        # Run the GStreamer pipeline in the main thread (blocks until done)
        try:
            self._gst_app.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._user_data.stop()
            if self._game_thread and self._game_thread.is_alive():
                self._game_thread.join(timeout=2.0)


def main():
    parser = get_pipeline_parser()
    args, _ = parser.parse_known_args()

    user_data = SpaceInvadersCallback()
    app = SpaceInvadersApp(camera_callback, user_data, parser)
    app.run()


if __name__ == "__main__":
    main()
