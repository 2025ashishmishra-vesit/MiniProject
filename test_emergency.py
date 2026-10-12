from config.settings import EnvConfig, build_scenarios
from environment.sumo_env import SumoTrafficEnv

env_cfg = EnvConfig()

# Load 4-way scenarios
env_cfg.scenarios = build_scenarios(
    "4_way",
    split="train"
)

env_cfg.scenario_selection = "cycle"

# IMPORTANT:
# Let normal traffic use fixed-time signals.
env_cfg.control_mode = "fixed"

env = SumoTrafficEnv(
    cfg=env_cfg,
    gui=True
)

state = env.reset()

print("\nScenario being tested:\n")
print(env.current_scenario)

done = False

while not done:

    # This value does not matter during normal fixed-time control.
    # If an emergency occurs, your emergency_action will override control.
    action = 0

    state, reward, done, info = env.step(action)

    print(
        f"time={info['sim_time']:.1f} | "
        f"mode={info['mode']} | "
        f"phase={info['phase']} | "
        f"queue={info['total_queue']}"
    )

env.close()