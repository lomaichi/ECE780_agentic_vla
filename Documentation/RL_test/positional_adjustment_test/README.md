# Positional adjustment test (interactive pi0 + keyboard)

A clean, first-person (wrist camera) LIBERO window. pi0 drives; you can interrupt at any time, nudge the
gripper in the horizontal plane (and vertically), and hand control back to pi0. The point is to see directly whether a
small in-plane correction (the action space of the pixel-Q corrector in
`failure_agent_prototype/RL/pixel_Q/pixel_Q_RL.md`) is enough for pi0 to recover. Nothing is drawn on
the camera image; the border colour is the only indicator.

## Run

1. pi0 server (`pi0_libero`, NOT `--env LIBERO`, which serves pi0.5):

   ```bash
   cd /opt/openpi && OPENPI_DATA_HOME=/home/DockerShared/.cache/openpi .venv/bin/python scripts/serve_policy.py \
       policy:checkpoint --policy.config=pi0_libero --policy.dir=gs://openpi-assets/checkpoints/pi0_libero
   ```

   Wait for `server listening on 0.0.0.0:8000` (~1 min).

2. Window (on the X display in `$DISPLAY`):

   ```bash
   ./run.sh                    # LIBERO-90 task 2, init state 0
   ./run.sh --init 3           # another of the 50 init states
   ./run.sh --agentview        # also show the third-person camera
   ./run.sh --help             # speeds, resolution, seed, ...
   ```

   The terminal logs each interruption: start step, duration, how far you moved the gripper (cm) and
   rotated it (deg), and the step of success.

## Keys (click the window first so it has focus)

| key | action |
|---|---|
| `space` | start / stop the interruption. Stopping hands back to pi0 |
| arrow keys | translate the gripper horizontally, in the wrist image's frame (up = towards the top of the image) |
| `ctrl` + up / down | move the gripper vertically (world up / down) |
| `ctrl` + left / right | rotate the gripper about the vertical axis (left = turn left, CCW seen from above) |
| `shift` (hold) | fine motion (x0.25) |
| `r` / `n` | restart the episode / go to the next init state |
| `q` / `esc` | quit |

Border: blue = pi0, red = you, green = success (the episode freezes; press `r` or `n`).

## Behaviour details

- **Interrupting**: pi0's queued actions are dropped. The arm holds its pose. Tilt stays locked, and height
  stays locked unless you use `ctrl` + up / down. The gripper keeps pi0's last open/close command.
- **Teleop control**: held keys move a target pose (8 cm/s, 30 deg/s by default). The OSC controller
  servos to that target, so the arm doesn't drift when no key is pressed. The target stays within
  3 cm of the gripper (in 3D), so it can't run away if the arm is blocked, e.g. pushing down into the table.
- **Handing back**: pi0 replans from the current observation, so its first chunk sees your adjusted
  pose.
- **pi0 input**: pi0 gets the same input as in `examples/libero/main.py`: 256 px renders, rotated 180°,
  resized to 224, replanning every 5 steps, 10 wait steps at the start. The window shows a separate
  640 px render of the same camera, unmirrored. Its mean pixel difference from pi0's frame is 1.3/255,
  so you see what pi0 sees, just sharper.
- **Episode length**: there's no step limit, since interruptions add steps. Restart with `r`.
- **Randomness**: pi0 is stochastic, so the same init can play out differently on each run.

## Files

- `run.sh`: entry point.
- `window.py`: Tk window on Python 3.11, standard library only. It launches the simulation and
  forwards your keys to it over a socketpair. Two processes are needed because the LIBERO venv's
  Python 3.8 Tk aborts on this X server (an xcb error), while Python 3.11's Tk works.
- `interactive_pi0.py`: simulation, pi0 client and teleop controller (LIBERO client venv,
  Python 3.8).

## Self-test (no window)

`./run.sh --selftest` drives the same code with scripted keys. pi0 runs for 40 steps, then right is held
for 30 steps, then up for 30, then ctrl+left for 30, then ctrl+up for 30, then ctrl+down for 30, then control goes back to pi0. It checks the
direction of each motion in the wrist camera frame and saves the before/after views to `selftest/`.

Result on 2026-11-18:

```
PASS  right -> image right: +9.1 cm along, dz +0.0 cm
PASS  up -> image up: +8.7 cm along, dz -0.3 cm
PASS  ctrl+left -> CCW from above: yaw +33.7 deg
PASS  ctrl+up -> world up: dz +8.9 cm, |dxy| 0.2 cm
PASS  ctrl+down -> world down: dz -7.6 cm, |dxy| 0.3 cm
```
