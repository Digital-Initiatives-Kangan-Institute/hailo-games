# Soccer

3D perspective penalty kick game using Hailo pose estimation.

## Description

A soccer penalty kick simulator with 3D perspective projection. The camera sits behind the player looking toward the goal. Move your foot near the ball to kick it — the ball flies forward with realistic physics (gravity, bounce, friction). An AI goalkeeper tracks the ball and attempts to save your shots. Score goals and track your shooting percentage over a 90-second round.

## How It Works

- A penalty spot ball sits at the center of the field
- **Move your foot** near the ball to kick it
- The ball flies forward with physics (gravity, bounce, friction)
- An AI goalkeeper tries to save your shot
- Score goals, track your shot percentage
- 90-second game timer

## Controls

| Gesture | Action |
|---------|--------|
| Move ankle near ball | Kick the ball |
| — | Ball direction based on foot position relative to ball |

## Requirements

- Hailo-8/8L/10H accelerator
- USB webcam or Raspberry Pi camera
- Pose estimation model (auto-downloaded)

## Usage

```bash
python3 soccer.py --input usb
```
