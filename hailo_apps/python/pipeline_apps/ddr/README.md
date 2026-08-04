# Dance Dance Revolution

Rhythm game — match your limbs to the falling notes.

## Description

A Dance Dance Revolution-style rhythm game using pose estimation. Four colored columns represent your limbs — move the correct limb into the column when a matching note reaches the hit line. Score PERFECT (100pts) or GOOD (50pts) based on timing accuracy. Build combos for bonus points. Speed increases as your score rises.

## How It Works

- Colored notes fall down 4 columns
- Each column corresponds to a limb:
  - **Red** = Left Arm (wrist)
  - **Blue** = Right Arm (wrist)
  - **Green** = Left Leg (knee)
  - **Yellow** = Right Leg (knee)
- Position your limb in the correct column when the note reaches the hit line
- PERFECT (100pts) / GOOD (50pts) scoring based on timing
- Combo multiplier for consecutive hits
- Speed increases as score rises

## Controls

| Gesture | Action |
|---------|--------|
| Left wrist in column | Hit red notes |
| Right wrist in column | Hit blue notes |
| Left knee in column | Hit green notes |
| Right knee in column | Hit yellow notes |

## Requirements

- Hailo-8/8L/10H accelerator
- USB webcam or Raspberry Pi camera
- Pose estimation model (auto-downloaded)

## Usage

```bash
python3 ddr.py --input usb
```
