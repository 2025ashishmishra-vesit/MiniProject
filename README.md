# SUMO RL Traffic Signal Controller

A **Reinforcement Learning** agent that controls signalized intersections in the [SUMO](https://eclipse.dev/sumo/) traffic simulator.  
The agent can train in pure **RL mode** or optional **hybrid mode** with fixed-time control during light traffic.

---

## Architecture

```
SUMO (TraCI)
  â”œâ”€â”€ N/S/E/W lanearea detectors + signal context  â†’  state (14 features)
  â””â”€â”€ Traffic light (TLS)         â†  action (select green phase)

DQN Agent (Dueling + Double DQN)
  â”œâ”€â”€ Online network  Q(s,a)
  â”œâ”€â”€ Target network  (soft-updated)
  â””â”€â”€ Replay buffer   50 k transitions

Control (per step)
  â”œâ”€â”€ pure RL: agent selects a configured green phase every decision
  â””â”€â”€ hybrid: fixed-time control can handle light traffic
```

### State vector (14 features, normalised 0â€“1)
| Index | Approach | Feature        |
|-------|----------|----------------|
| 0     | N        | queue length   |
| 1     | N        | mean speed     |
| 2     | N        | vehicle count  |
| 3â€“5   | S        | same           |
| 6â€“8   | E        | same           |
| 9â€“11  | W        | same           |
| 12    | Signal   | current green phase index |
| 13    | Signal   | elapsed green time, capped at 60 seconds |

### Actions
| Action | Meaning |
|--------|---------|
| 0â€“3   | Select the configured green phase at this index; selecting the active index holds it |

The DQN always has four outputs. In a 3-way scenario action 3 is masked; in a
4-way scenario all four actions map to the configured green phases. Before
minimum green time expires, only the active phase action is selectable.

Yellow phases are handled automatically by the environment during a switch.

### Reward
```
reward = previous_pressure - current_pressure
pressure = alpha * total_queue + beta * total_waiting_time
```

The reward is positive when the latest decision reduces queue/waiting pressure.

---

## Installation

### 1 â€” Install SUMO
```bash
# Ubuntu / Debian
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update && sudo apt-get install sumo sumo-tools sumo-doc

# macOS (Homebrew)
brew install sumo

# Or via pip (headless)
pip install eclipse-sumo
```

Set the environment variable:
```bash
export SUMO_HOME="/usr/share/sumo"          # adjust to your install path
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

### 2 â€” Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## Setting up your SUMO network

### Option A â€” Generate a simple 4-way grid (quickstart)
```bash
cd sumo_net

# Generate network
netgenerate --grid --grid.number 2 --grid.length 200 \
            --tls.guess --output-file intersection.net.xml

# Generate random vehicle demand (9000 s simulation)
python $SUMO_HOME/tools/randomTrips.py \
    -n intersection.net.xml \
    -r routes.rou.xml \
    -e 9000 -p 1.2 --fringe-factor 10
```

### Option B â€” Use your own network
Replace `sumo_net/intersection.net.xml` and `routes.rou.xml` with your files.

## Reward Tuning Experiments

Run controlled reward-setting trials with separate model/result folders:

```bash
python reward_tuning_experiments.py
```

Default trial length is 300 episodes per reward setting. Each run is saved under:

```text
experiments/reward_tuning/<timestamp>/<setting_name>/
```

Folder names include the reward settings:

```text
tw_<throughput_weight>__sp_<switch_penalty>__mg_<min_green>__ep_<episodes>
```

The master comparison is written to:

```text
experiments/reward_tuning/<timestamp>/summary.csv
```

### Add detectors
Edit `sumo_net/detectors.add.xml` and replace the lane IDs with your actual lane IDs.  
Find lane IDs in SUMO GUI â†’ right-click on a lane â†’ "Copy lane ID".

Add the detector file to `intersection.sumocfg`:
```xml
<additional-files value="detectors.add.xml"/>
```

### Update `config/settings.py`
```python
detector_ids = {
    "N": ["det_N_0", "det_N_1"],  # one E2 detector per incoming lane
    "S": ["det_S_0", "det_S_1"],
    "E": ["det_E_0", "det_E_1"],
    "W": ["det_W_0", "det_W_1"],
}
tls_id = "center"  # traffic light ID in your .net.xml
```

---

## Training

```bash
# Standard training (no GUI, fastest)
python train.py

# With SUMO GUI so you can watch
python train.py --gui

# Resume from checkpoint
python train.py --resume models/checkpoints/ep_100.pt

# Override episode count
python train.py --episodes 500
```

Training output (per episode):
```
Ep  12 | reward= -1234.5 | avg_queue= 8.2 | RL%=  67.3 | Îµ=0.9820 | loss=0.0341 | 48.2s
         â†³ EVAL reward=-980.1  queue=6.4
         â†³ New best model saved (-980.1)
```

Logs are saved to `logs/train_YYYYMMDD_HHMMSS.csv` and `.json`.

---

## Evaluation only

```bash
python train.py --eval --resume models/best/best.pt
```

---

## Configuration reference

All tuneable parameters live in `config/settings.py`.

| Parameter | Default | Description |
|---|---|---|
| `delta_t` | 5 | Seconds of simulation per RL step |
| `max_steps` | 1800 | Steps per episode |
| `normal_queue_threshold` | 3 | Max queue (veh) before RL activates |
| `fixed_green_time` | 30 | Seconds each direction is green (fixed mode) |
| `alpha` | 0.5 | Queue penalty weight in reward |
| `beta` | 0.1 | Waiting-time penalty weight in reward |
| `epsilon_decay` | 0.9995 | Per-step Îµ decay (multiplicative) |
| `gamma` | 0.99 | Discount factor |
| `buffer_size` | 50 000 | Replay buffer capacity |
| `hidden_layers` | [256,256,128] | DQN network hidden layer sizes |

---

## Project structure

```
sumo_rl_traffic/
â”œâ”€â”€ train.py                  â† main training / eval entry point
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ config/
â”‚   â””â”€â”€ settings.py           â† EnvConfig, AgentConfig, TrainConfig
â”œâ”€â”€ environment/
â”‚   â””â”€â”€ sumo_env.py           â† Gym-style SUMO wrapper
â”œâ”€â”€ agent/
â”‚   â””â”€â”€ dqn_agent.py          â† DQN, replay buffer, Q-network
â”œâ”€â”€ utils/
â”‚   â”œâ”€â”€ logger.py             â† CSV / JSON logger
â”‚   â””â”€â”€ metrics.py            â† EpisodeMetrics
â”œâ”€â”€ sumo_net/
â”‚   â”œâ”€â”€ intersection.sumocfg  â† SUMO config template
â”‚   â””â”€â”€ detectors.add.xml     â† E2 detector definitions
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ checkpoints/          â† periodic saves
â”‚   â””â”€â”€ best/                 â† best eval model
â””â”€â”€ logs/                     â† training CSV + JSON
```

---

## Tips

- **Start with GUI on** to verify the detector positions and phase logic before long runs.
- If RL never activates, lower `normal_queue_threshold` or increase traffic demand (`-p` in randomTrips).
- If reward stays very negative, try increasing `alpha` / `beta` or tuning `fixed_green_time`.
- For multi-intersection networks, instantiate one `DQNAgent` per intersection and share a replay buffer (MARL setup).

