# Two-ball imitation-learning experiment

## Problem formulation

The game contains a small blue ball, Ball B, and a larger red target, Ball R.
The policy must pick up Ball B, move it onto Ball R, and release it while the
two circles intersect. The simulation and data recorder update at 10 Hz.

The policy state has five values:

```text
s_t = [x_b, y_b, x_r, y_r, picked_up]
```

- `x_b, y_b`: center of the blue ball.
- `x_r, y_r`: center of the red ball.
- `picked_up`: whether the blue ball was being held at the start of this time
  step (`0.0` or `1.0`).

The policy action has three values:

```text
a_t = [dx_b, dy_b, pick_up]
```

- `dx_b, dy_b`: blue-ball displacement during this time step, in pixels.
- `pick_up`: whether the blue ball should be held after this action (`0.0` or
  `1.0`). During inference this is represented by a probability.

### `picked_up` versus `pick_up`

These similarly named fields represent different points in time:

- `picked_up` is part of state `s_t`. It describes the ball before the action.
- `pick_up` is part of action `a_t`. It describes the commanded hold state
  after the action.

This distinction lets one CSV row represent state transitions explicitly:

| Event | State `picked_up` | Action `pick_up` | Meaning |
| --- | ---: | ---: | --- |
| Initial pickup | 0 | 1 | Start controlling the ball |
| Continue dragging | 1 | 1 | Keep controlling the ball |
| Release | 1 | 0 | Let go of the ball |

Consequently, the next row's `picked_up` normally equals the current row's
`pick_up`.

## Training-data collection

Training data is stored under:

```text
training_data/
└── data_YYMMDD_HHMM/
    ├── 1.csv
    ├── 2.csv
    └── ...
```

Each timestamped directory is one collection session. Each numbered CSV is one
game round. Continuing the latest session finds the largest numeric filename
and creates the next one. A successful drop closes the current CSV, increments
the recorded-round count, randomizes both balls, and immediately starts the
next CSV.

The CSV column order is:

```csv
x_b,y_b,x_r,y_r,picked_up,dx_b,dy_b,pick_up
```

For each 10 Hz tick, the recorder writes the state before that tick's action
and the action observed during the tick:

```text
dx_b = current_x_b - previous_x_b
dy_b = current_y_b - previous_y_b
```

For example, a short demonstration can look like this:

```csv
x_b,y_b,x_r,y_r,picked_up,dx_b,dy_b,pick_up
120.0,180.0,500.0,320.0,0.0,0.0,0.0,1.0
120.0,180.0,500.0,320.0,1.0,24.0,9.0,1.0
144.0,189.0,500.0,320.0,1.0,20.0,8.0,1.0
164.0,197.0,500.0,320.0,1.0,10.0,5.0,0.0
```

The first row is the pickup transition. The middle rows keep the ball picked
up while moving it. The final row includes the last movement and releases it.

### Idle-sample handling

The recorder does not write the waiting period before the user's first pickup.
Without this rule, the same initial state is recorded repeatedly with
`pick_up=0`, teaching an autonomous controller to remain idle.

For compatibility with previously collected sessions, training also filters
rows satisfying all of the following:

```text
picked_up == 0
pick_up == 0
dx_b == 0
dy_b == 0
```

This removes old pre-pickup idle rows and idle rows following an unsuccessful
release. Pickup transitions, drag samples, and release transitions remain.

## Network design and training

`BallPolicy` is a multilayer perceptron:

```text
5 inputs -> 64 -> ReLU -> 64 -> ReLU -> 3 outputs
```

The first two outputs represent normalized `dx_b, dy_b`. The third output is a
pickup/release logit.

Training reads every numbered CSV from the newest `data_*` directory. The
current settings are constants near the top of `policy.py`:

```python
TRAINING_EPOCHS = 200
TRAINING_BATCH_SIZE = 64
TRAINING_LEARNING_RATE = 1e-3
TRAINING_RANDOM_SEED = 110
TRAINING_PROGRESS_REPORTS = 100
FILTER_UNPICKED_IDLE_SAMPLES = True
RELEASE_LOSS_WEIGHT = 8.0
```

All five state fields are standardized using the training-set mean and
standard deviation. The two movement targets are standardized separately.
This prevents pixel-scale movement errors from overwhelming the binary
pickup/release objective.

The total batch loss is:

```text
loss = MSE(normalized predicted movement, normalized human movement)
     + weighted BCEWithLogits(predicted pickup logit, human pick_up)
```

Release rows (`pick_up=0`) receive `RELEASE_LOSS_WEIGHT`. A demonstration has
many dragging ticks but only one release tick, so this weighting prevents the
model from learning to hold the ball forever.

## Checkpoints

Training saves timestamped checkpoints under:

```text
weight/weight_YYMMDD_HHMMSS.pt
```

Each checkpoint contains:

- Model parameters.
- State mean and standard deviation.
- Movement mean and standard deviation.
- State and action field names.
- Source data-directory name.
- Sample count, epoch count, and final loss.

These normalization values are required for inference; a checkpoint is more
than the raw neural-network weights.

## Run Mode

Run Mode loads the newest timestamped checkpoint. At every 10 Hz tick it:

1. Builds `[x_b, y_b, x_r, y_r, picked_up]`.
2. Standardizes the state with the checkpoint statistics.
3. Runs the MLP.
4. Converts the first two outputs back to pixel deltas.
5. Converts the third output to a probability with `sigmoid`.
6. Treats probability `>= 0.5` as picked up.

Movement is applied if the ball was picked up before the action or is picked up
by the new action. This preserves the final movement delta on a release tick.
When the output transitions from picked up to released inside the red circle,
the game calculates a score, automatically randomizes the balls, and continues
Run Mode.
