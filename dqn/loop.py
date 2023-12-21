import torch
from trainer import Trainer
from replay_memory import Memory
from itertools import count
print(torch.cuda.is_available())


def train(trainer: Trainer):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_episodes = 500
    episode_durations = []
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = Trainer.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = trainer.select_action(state)
            observation, reward, terminated, truncated, _ = Trainer.env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            Trainer.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            trainer.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = trainer.target_net.state_dict()
            policy_net_state_dict = trainer.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*Trainer.TAU + target_net_state_dict[key]*(1-Trainer.TAU)
            trainer.target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                trainer.plot_durations()
                break
    print('Complete')