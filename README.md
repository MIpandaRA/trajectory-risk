# Trajectory prediction and collision-risk alerts

[![tests](https://github.com/MIpandaRA/trajectory-risk/actions/workflows/tests.yml/badge.svg)](https://github.com/MIpandaRA/trajectory-risk/actions/workflows/tests.yml)

A live system that watches a car park, predicts where every car and pedestrian
will go over the next few seconds, and raises an alert when a car and a person
are likely to come too close. Cars are predicted by **ContextVAE**, which reads
the road map; pedestrians by **SocialVAE**, which accounts for people nearby.
Both draw hundreds of possible futures and summarise them as a few candidate
paths with confidences. It runs on a CPU, with a simulator feeding a live
dashboard at 60 updates a second.

![Constant velocity vs ContextVAE on a straight road and a turn](docs/figures/examples_cars.png)

## What it does

* **Predicts several futures per agent, live.** A single "best guess" model
  averages left and right turns into a path nobody takes
  ([why](docs/figures/cvae_toy.png)); these models keep both.
* **Scores risk over time, not just geometry.** Every candidate car path is
  compared with every candidate pedestrian path *at the same moments* over the
  next 3 s. Two paths that cross 5 s apart are safe.
* **Alerts without flickering.** An alert needs several high readings in a row
  to switch on and several low ones to switch off.
* **Stays responsive when the models are slow.** Models run on background
  threads; a new request replaces a waiting one instead of queueing behind it.
* **Is measured against a simple baseline.** Every result is compared with
  constant velocity: "everyone keeps going straight at their current speed".

## Results (simulation)

All results below are from the built-in simulator, on two maps: **T11**, a
small car park, and **OneNorth**, a nuScenes map (the data ContextVAE was
trained on). They are not real-world performance; see
[Limitations](#limitations).

### Prediction accuracy

![Average error of the top guess, straight vs turning, cars and pedestrians](docs/figures/results.png)

Average distance between the most confident predicted path and what actually
happened (lower is better):

* **On straight roads, the baseline wins.** Simulated cars never change speed,
  which is exactly constant velocity's assumption.
* **On turns, ContextVAE wins for cars**: its predictions bend with the road.
  On OneNorth, where cars are faster, the gap is larger: 8.0 m vs 9.8 m on
  turns, while on straight roads the baseline is far ahead (0.78 m vs 2.61 m).
* **For pedestrians, the baseline wins everywhere.** Simulated pedestrians walk
  in straight lines; SocialVAE, trained on real people who wander, predicts a
  spread they never show.
* **The learned models usually have a good guess, but not first.** The black
  line in each blue bar is the best of all their candidates.

![Constant velocity vs SocialVAE, straight and turning](docs/figures/examples_pedestrians.png)

### Alerts

Does the alert fire when a car and a pedestrian really come within 2.2 m in the
next 3 s, and how early? All scenarios, with the default alert settings:

| | Learned models | Constant velocity |
|---|---|---|
| Danger events caught | **41 of 42** | 39 of 42 |
| False alarms per minute | 3.5 | **0.8** |
| Median warning before closest approach | **2.6 s** | 2.2 s |
| AUROC (how well the score ranks dangerous moments above safe ones) | **0.995** | 0.902 |

The learned models catch more and warn earlier, at about four times the false
alarms. Full numbers: [`eval/results/`](eval/results/).

## How it was verified

This repository is a rebuild of an earlier prototype, checked piece by piece
rather than trusted:

* **Against the original model code.** The network files in
  `src/trajectory_risk/third_party/` are byte-identical to the upstream
  ContextVAE and SocialVAE repositories, and the input preparation (rotating
  each car into its own frame, cropping the map) was checked line by line
  against ContextVAE's training code.
* **Against the prototype's frozen output.** Given the same input and random
  seed, both models reproduce the prototype's 2,000 and 740 sampled futures to
  within 0.02 mm (`tests/test_models.py`).
* **Against the earlier results.** Rerunning the full evaluation reproduces the
  earlier constant-velocity and alert numbers exactly, and the learned-model
  accuracy to within 0.01 m.
* **On a clean machine.** CI installs the package from scratch and runs the
  test suite on every push.

## How it works

```
simulator ──ticks (JSON over TCP)──▶ protocol ──▶ tracking ──▶ predictors ──▶ risk ──▶ dashboard
  (60/s, world metres)            (validate)   (one sample    (ContextVAE,   (closest
                                               every 0.5 s /   SocialVAE,     approach,
                                               0.4 s)          baseline)      alerts)
```

Module-by-module description: [docs/architecture.md](docs/architecture.md).

## Quick start

Python 3.10 (the model weights need torch 1.11). Using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/MIpandaRA/trajectory-risk.git
cd trajectory-risk
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install "torch==1.11.0+cpu" "torchvision==0.12.0+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]"
```

**Fedora and other distributions with glibc 2.41+:** torch 1.11 fails to load
with "cannot enable executable stack". Clear the flag once:

```bash
uv pip install patchelf
for f in .venv/lib/python3.10/site-packages/torch/lib/*.so*; do patchelf --clear-execstack "$f"; done
```

**Model weights** are not in this repository (`checkpoints/` is ignored):

* SocialVAE: `models/sdd/ckpt-best` from the
  [SocialVAE repository](https://github.com/xupei0610/SocialVAE), copied to
  `checkpoints/socialvae/ckpt-best`.
* ContextVAE: `checkpoints/contextvae/ckpt-best`, a model trained with
  ContextVAE's nuScenes configuration. The upstream repository does not
  publish weights.

Without weights, everything still runs with the constant-velocity baseline
(`--models cv`).

### Run it live

Two terminals:

```bash
traj-producer --map T11 --num-pedestrians 10 --crossing-bias 0.6 --autostart   # simulator
traj-dashboard --map T11                                                       # predictions and alerts
```

In the dashboard: **D** debug overlay (model latency, superseded requests),
**A** alternative candidates, **Esc** quit.

### Reproduce the results

```bash
python -m eval.run_predictions        # all scenarios, both model sets (about 30 min on a laptop CPU)
python -m eval.accuracy               # eval/results/accuracy_*.csv
python -m eval.risk_eval              # eval/results/risk_*.csv
python -m eval.compare_vehicles       # tables behind docs/figures/results.png
python -m eval.compare_pedestrians
python -m eval.make_figures           # docs/figures/*.png
```

The OneNorth scenario needs `maps/OneNorthRoundabout/semantic_map.pkl` (85 MB,
not in the repository).

### Tests

```bash
pytest
```

Tests that need the model weights or the OneNorth map are skipped when those
files are absent.

## Limitations

* **Simulation only, so far.** Simulated cars never brake or change speed and
  pedestrians walk in straight lines, which flatters the constant-velocity
  baseline. Advance warning in simulation does not imply advance warning on
  real traffic; a real-data evaluation is the next step.
* **Car and pedestrian are assumed not to react to each other.** The risk score
  multiplies the car's and the pedestrian's candidate confidences, which is
  only valid if their futures are independent. In reality a pedestrian who sees
  a car swerve is less likely to step out. The models cannot supply that
  dependence: ContextVAE only sees other cars as neighbours and SocialVAE only
  other pedestrians, and each predicts on its own. Capturing it would need a
  **joint prediction model** that samples both agents' futures together as
  consistent pairs.
* **Confidences are not probabilities.** A candidate's confidence is the share
  of the model's samples near it, so the risk score is a score, not a
  calibrated chance of collision. The alert threshold is therefore set
  empirically, not from costs.
* **Default alert settings are the prototype's.** They trade many false alarms
  for early warning; other thresholds are in `eval/results/risk_threshold_sweep.csv`.
* **CPU latency.** On a laptop CPU, ContextVAE takes about half a second per
  prediction, so live predictions can be over a second old by the time they
  are scored.
* **Not a safety system.** This is a study of prediction and risk scoring, not
  something to put in front of real vehicles.

## Acknowledgements and licences

* Code in this repository: [MIT](LICENSE), © 2026 MIpandaRA.
* **ContextVAE** network (`src/trajectory_risk/third_party/contextvae`): MIT,
  © 2023 徐霈 (Pei Xu). Xu, Hayet, Karamouzas, "Context-Aware Timewise VAEs for
  Real-Time Vehicle Trajectory Prediction", IEEE RA-L 8(9), 2023.
* **SocialVAE** network (`src/trajectory_risk/third_party/socialvae`): MIT,
  © 2021 Pei Xu. Xu, Hayet, Karamouzas, "SocialVAE: Human Trajectory Prediction
  using Timewise Latents", ECCV 2022.
* **OneNorth map** (`maps/OneNorthRoundabout`): derived from the
  [nuScenes](https://www.nuscenes.org/) map expansion, **CC BY-NC-SA 4.0**
  (non-commercial, share-alike).
* **T11 road annotations** (`maps/T11`): CC BY 4.0, by MIpandaRA.
* **Model weights** (not distributed here) were trained on nuScenes and the
  Stanford Drone Dataset, whose non-commercial terms may extend to models
  trained on them; treat them as non-commercial.
