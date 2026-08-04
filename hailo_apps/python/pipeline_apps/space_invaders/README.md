# Space Invaders

An interactive Space Invaders game using Hailo pose estimation. Control the ship with your body!

## How It Works

- A classic Space Invaders alien formation descends from the top of the screen
- **Move your head** left/right to steer the ship
- **Raise both hands above your head** to fire a burst of shots
- The ship also auto-fires at a steady pace
- Destroy all aliens to respawn the formation; if aliens reach the bottom, it's game over
- Different alien rows are worth different points (10–40 pts)
- Aliens shoot back — getting hit costs a life (3 lives total)
- 90-second countdown timer; game ends when time runs out or you lose all lives
- Auto-restarts after a brief results screen

## Controls

| Gesture | Action |
|---------|--------|
| Move head left/right | Steer the ship |
| Raise both hands above head | Rapid fire |
| — | Auto-fire when idle |

## Usage

```bash
python3 space_invaders.py --input usb
```

## Game Elements

| Element | Description |
|---------|-------------|
| Red aliens (top row) | 40 pts each |
| Orange aliens | 30 pts each |
| Green aliens | 20 pts each |
| Blue aliens (bottom row) | 10 pts each |
| Green bullets | Player shots |
| Red diamonds | Alien shots |
| Green triangle | Player ship |
| Score popup | "+N" floats up on hit |
| "-1 LIFE" popup | Shown when hit by alien bullet |
