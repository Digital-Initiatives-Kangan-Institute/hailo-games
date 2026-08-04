# Easter Game

![Easter Game Example](../../../../doc/images/easter_game.gif)

Catch Easter eggs and Afikoman matzahs with your hands for points.

## Description

An interactive Easter Egg and Afikoman catching game using Hailo pose estimation. A custom background is displayed with items appearing one at a time. Catch Easter eggs (20 pts) and Afikoman matzahs (10 pts) by moving your hands to them. Items despawn after 3 seconds. Features a leaderboard with auto-named players, 90-second countdown timer, and auto-restart.

## How It Works

- A custom background image is displayed
- Easter eggs (colorful ovals, 20 pts) and Afikoman matzahs (golden rectangles, 10 pts) appear one at a time at random spots
- Players catch them with their hands (wrist keypoints)
- Eggs appear more often than Afikoman
- If missed after 3 seconds, the next item spawns automatically
- "+20" / "+10" pop-ups appear on catch
- Leaderboard on the right side with auto-named players
- 90-second countdown timer at the top
- Game over screen shows final scores, then auto-restarts

## Controls

| Gesture | Action |
|---------|--------|
| Move hands to item | Catch it |

## Requirements

- Hailo-8/8L/10H accelerator
- USB webcam or Raspberry Pi camera
- Pose estimation model (auto-downloaded)

## Usage

```bash
python3 easter_game.py --input usb
```

Optionally pass a custom background:
```bash
python3 easter_game.py --input usb --background /path/to/background.png
```
