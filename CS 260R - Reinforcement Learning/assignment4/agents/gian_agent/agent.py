import pathlib

# Use dot here to denote importing the file in the folder hosting this file.
from .ppo_trainer import PPOTrainer, PPOConfig

FOLDER_ROOT = pathlib.Path(__file__).parent  # The path to the folder hosting this file.


class Policy:

    CREATOR_NAME = "Gian Zignago"
    CREATOR_UID = "706294998"

    def __init__(self):
        config = PPOConfig()
        self.agent = PPOTrainer(config=config)
        self.agent.load_w(log_dir=FOLDER_ROOT, suffix="iter275")

    def __call__(self, obs):
        value, action, action_log_prob = self.agent.compute_action(obs)
        action = action.detach().cpu().numpy()
        return action
