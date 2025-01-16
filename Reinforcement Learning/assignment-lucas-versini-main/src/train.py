from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

import torch.nn as nn
import torch
import os
import random
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

class ReplayBuffer:
    """ Replay buffer to store past experiences that the agent can use for training data. """
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, state, action, reward, next_state, done):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return tuple(map(lambda x: torch.Tensor(np.array(x)).to(self.device), zip(*batch)))
    
    def __len__(self):
        return len(self.data)
    
class PolicyNetwork(nn.Module):
    """ NN that approximates the Q function. Input: state, output: Q values for each action. """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.value_branch = self._build_branch(state_dim, hidden_dim, action_dim)
        self.advantage_branch = self._build_branch(state_dim, hidden_dim, action_dim)

    def _build_branch(self, input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        value = self.value_branch(x)
        advantage = self.advantage_branch(x)
        return value + advantage - advantage.mean(dim = 1, keepdim = True)

class ProjectAgent:
    def __init__(self, save_path = None):
        self.config = {
            "batch_size": 1024,
            "gradient_num_steps": 2,
            "gamma": 0.98,
            "buffer_capacity": 1_000_000,
            "initial_buffer_fill": 1024,
            "epsilon_end": 1e-2,
            "epsilon_start": 1.0,
            "epsilon_decay_steps": int(1e4),
            "epsilon_delay": 400,
            "max_episodes": 300,
            "state_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.n,
            "hidden_dim": 512,
            "lr": 1e-3,
            "target_update_freq": 600,
            "target_update_tau": 1e-3,
            "save_frequency": 50
        }

        # Initialize epsilon
        self.epsilon = self.config["epsilon_start"]
        self.epsilon_step = (self.config["epsilon_start"] - self.config["epsilon_end"]) / self.config["epsilon_decay_steps"]
        self.step_count = 0
        self.episode_count = 0

        # Initialize model and target model
        self.policy_net = PolicyNetwork(
            self.config["state_dim"], 
            self.config["hidden_dim"], 
            self.config["action_dim"]
        ).to(device)
        self.target_net = PolicyNetwork(
            self.config["state_dim"], 
            self.config["hidden_dim"], 
            self.config["action_dim"]
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = self.config["lr"])
        self.criterion = nn.SmoothL1Loss()

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.config["buffer_capacity"], device)
        self.save_path = save_path if save_path is not None else "model.pth"

    def save(self, path = None):
        """ Save the model and optimizer state to a file. """
        if path is None:
            path = self.save_path
        checkpoint = {
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'replay_buffer': self.replay_buffer.data,
            'buffer_index': self.replay_buffer.index
        }
        torch.save(checkpoint, path)

    def load(self, path = None):
        """ Load the model and optimizer state from a file. """
        if path is None:
            path = self.save_path
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location = device)
            self.policy_net.load_state_dict(checkpoint['policy_state'])
            self.target_net.load_state_dict(checkpoint['target_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.config = checkpoint['config']
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.episode_count = checkpoint['episode_count']
            self.replay_buffer.data = checkpoint['replay_buffer']
            self.replay_buffer.index = checkpoint['buffer_index']
            print(f"Loaded agent at episode {self.episode_count}.")
            return True
        print(f"No checkpoint found at {path}.")
        return False

    def act(self, observation, use_random = False):
        with torch.no_grad():
            q_values = self.policy_net(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(q_values).item()
    
    def train(self, env):
        """ Train the agent using DQN. """
        if self.episode_count == 0:
            self.populate_replay_buffer(env)

        episode_cumul_reward = 0

        state, _ = env.reset()

        while self.episode_count < self.config["max_episodes"]:
            # Update lr
            # if self.step_count % 100 == 0:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = max(1e-4, param_group['lr'] * 0.95)

            # Update epsilon
            if self.step_count > self.config["epsilon_delay"]:
                self.epsilon = max(self.config["epsilon_end"], self.epsilon - self.epsilon_step)

            # Epsilon greedy policy
            action = self.act(state) if random.random() > self.epsilon else env.action_space.sample()

            # Take action, accumulate reward and store experience
            next_state, reward, done, truncated, _ = env.step(action)
            self.replay_buffer.append(state, action, reward, next_state, done)
            episode_cumul_reward += reward

            # Update model
            if len(self.replay_buffer) >= self.config["batch_size"]:
                for _ in range(self.config["gradient_num_steps"]):
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config["batch_size"])

                    evaluated_action = torch.argmax(self.policy_net(next_states).detach(), dim = 1)
                    Q_target = self.target_net(next_states).detach()
                    Q_target_max = Q_target.gather(1, evaluated_action.unsqueeze(1)).squeeze(1)

                    update = rewards + self.config["gamma"] * (1 - dones) * Q_target_max
                    Q_net = self.policy_net(states).gather(1, actions.unsqueeze(1).long())

                    loss = self.criterion(Q_net, update.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
            # Update target network
            target_state_dict = self.target_net.state_dict()
            policy_state = self.policy_net.state_dict()
            for k in target_state_dict.keys():
                target_state_dict[k] = self.config["target_update_tau"] * policy_state[k] + (1 - self.config["target_update_tau"]) * target_state_dict[k]
            self.target_net.load_state_dict(target_state_dict)

            self.step_count += 1

            # Reset environment if done
            if done or truncated:
                self.episode_count += 1
                if self.episode_count % self.config["save_frequency"] == 0:
                    self.save(self.save_path)
                print(f"Episode {self.episode_count} | epsilon {self.epsilon:.3f} | memory size {len(self.replay_buffer)} | cumulative reward {episode_cumul_reward:.3e} - loss {loss:.3e}")
                
                state, _ = env.reset()
                episode_cumul_reward = 0
            else:
                state = next_state
   
        self.save(self.save_path)
        print("Training finished.")
    
    def populate_replay_buffer(self, env):
        """ Fill the replay buffer with random experiences. """
        state, _ = env.reset()
        for _ in range(self.config["initial_buffer_fill"]):
            action = env.action_space.sample()
            next_state, reward, done, truncated, _ = env.step(action)

            self.replay_buffer.append(state, action, reward, next_state, done)
            state = next_state 
            if done or truncated:
                state, _ = env.reset()
            else:
                state = next_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", type = bool, default = False)
    args = parser.parse_args()
    seed_everything(seed = 42)

    agent = ProjectAgent()
    if not agent.load() or args.train:
        print("Training the agent")
        agent.train(env)
        agent.save(agent.save_path)
        print("Agent trained and saved")