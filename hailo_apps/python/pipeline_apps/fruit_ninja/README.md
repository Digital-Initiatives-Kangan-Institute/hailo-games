# Fruit Ninja

Slice flying fruits with your hands — avoid the bombs!

## Description

A Fruit Ninja-style game using pose estimation. Fruits launch upward from the bottom of the screen — wave your hands to create blade trails that slice them. Bombs also fly up; slicing a bomb freezes you for 3 seconds. Miss 3 fruits and it's game over. Build combos by slicing multiple fruits quickly. Survive the 60-second timer for the highest score.

## How It Works

- Fruits launch upward from the bottom of the screen
- **Wave your hands** to create blade trails that slice fruits
- Bombs also fly up — slicing a bomb freezes you for 3 seconds
- Miss 3 fruits and it's game over
- Combo system: consecutive slices within 1 second earn bonus points
- 60-second timer — survive and get the highest score

## Controls

| Gesture | Action |
|---------|--------|
| Wave left hand | Slice with cyan blade |
| Wave right hand | Slice with orange blade |

## Requirements

- Hailo-8/8L/10H accelerator
- USB webcam or Raspberry Pi camera
- Pose estimation model (auto-downloaded)

## Usage

```bash
python3 fruit_ninja.py --input usb
```
