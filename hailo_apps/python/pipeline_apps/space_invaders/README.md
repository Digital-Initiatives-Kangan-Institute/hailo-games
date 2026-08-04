# Space Invaders

Steer with your head, raise hands to fire, defend Earth from aliens.

## Description

A classic Space Invaders game using Hailo pose estimation. Move your head left and right to steer the player ship across the bottom of the screen. Raise both hands above your head to fire a rapid burst of shots at the descending alien formation. Destroy all aliens to respawn the wave. Each alien row scores different points — the top red row is worth the most. Aliens shoot back, so dodge their red diamond bullets. The game ends when you lose all 3 lives, the aliens reach the bottom, or the 90-second timer runs out.

## How It Works

- A grid of 4 × 6 aliens descends from the top in a side-to-side pattern
- **Move your head** left/right to slide the green ship along the bottom
- **Raise both hands above your head** to fire faster (rapid-fire gesture)
- The ship also auto-fires at a steady rate
- Aliens speed up as you destroy more of them
- Each alien hit awards 10–40 points depending on the row
- Aliens fire back — getting hit costs a life
- Survive the 90-second timer for a high score

## Controls

| Gesture | Action |
|---------|--------|
| Head left/right | Steer the ship |
| Both hands above head | Rapid fire |
| — | Auto-fire when idle |

## Requirements

- Hailo-8/8L/10H accelerator
- USB webcam or Raspberry Pi camera
- Pose estimation model (auto-downloaded)

## Usage

```bash
python3 space_invaders.py --input usb
```

Or use the launch script:

```bash
./play_space_invaders.sh
```
