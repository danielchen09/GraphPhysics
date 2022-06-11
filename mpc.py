import torch


from utils import *

class CEMMPC:
    def __init__(self, env, horizon, total_samples, top_samples):
        self.env = env
        self.env.physics.data.qacc_warmstart[:] = 0 # deterministic
        self.env.reset()
        self.initial_state = self.env.physics.get_state()

        self.action_spec = env.action_spec()
        self.action_shape = self.action_spec.shape[0]
        self.horizon = horizon
        self.total_samples = total_samples # G
        self.top_samples = top_samples # K
    
    def reset_env(self):
        self.env.reset()
        self.env.physics.set_state(self.initial_state)

    def rollout(self, actions):
        # actions: G x T x A
        # returns: G

        returns = []
        for sample_idx in range(self.total_samples):
            self.reset_env()
            reward = 0
            for step in range(self.horizon):
                reward += self.env.step(actions[sample_idx][step]).reward
            returns.append(reward)
        return torch.tensor(returns)

    def solve(self, num_iters):
        action_mean = torch.randn(1, self.horizon, self.action_shape)
        action_std = torch.randn(1, self.horizon, self.action_shape)
        actions = action_mean + action_std * torch.randn(self.total_samples, self.horizon, self.action_shape)
        
        mean_returns = []
        for t in range(num_iters):
            # calculate return
            returns = self.rollout(actions)
            mean_ret = returns.mean().item()
            print(f'epoch {t}: {mean_ret}')
            mean_returns.append(mean_ret)

            # update action mean and std with top K action sequences
            _, top_idx = returns.topk(self.top_samples, dim=0, largest=True, sorted=False)
            top_actions = actions[top_idx]         
            action_mean = top_actions.mean(dim=0, keepdim=True)
            action_std = top_actions.std(dim=0, keepdim=True, unbiased=False)

            # replace bottom G - K action sequences sampled from N(action_mean, action_std)
            bot_samples = self.total_samples - self.top_samples
            _, bot_idx = returns.topk(bot_samples, dim=0, largest=False, sorted=False)
            actions[bot_idx] = action_mean + action_std * torch.randn(bot_samples, self.horizon, self.action_shape)

        return action_mean, action_std, mean_returns

def run_mpc():
    from dm_control import suite
    import matplotlib.pyplot as plt
    import numpy as np

    env = suite.load('cheetah', 'run')
    H, G, K, T = 100, 100, 40, 500
    solver = CEMMPC(env, H, G, K)
    mu, std, rets = solver.solve(T)
    save_pickle('mpc_results/cheetah.pkl', {
        'mean': mu,
        'std': std,
        'rets': rets
    })
    plt.plot(np.arange(T), rets)
    plt.show()


def render_mpc(filepath):
    from render import generate_video
    from env import CheetahEnvCreator
    from dm_control import suite
    from tqdm import tqdm

    env = suite.load('cheetah', 'run')

    mpc_result = load_pickle(filepath)
    action_mean = mpc_result['mean']
    action_std = mpc_result['std']
    H = 100
    actions = action_mean + action_std * torch.randn(1, H, env.action_spec().shape[0])
    actions = actions.squeeze(0)

    frames = []
    for t in tqdm(range(H)):
        env.step(actions[t])
        frames.append(env.physics.render(camera_id=0, height=200, width=200))

    generate_video(frames, 'mpc_results/cheetah.mp4')

def plot_mpc(filepath):
    import matplotlib.pyplot as plt

    mpc_result = load_pickle(filepath)
    rets = mpc_result['rets']
    plt.plot(np.arange(500), rets)
    plt.title('Total return over time')
    plt.xlabel('Iteration')
    plt.ylabel('Total return')
    plt.show()

if __name__ == '__main__':
    plot_mpc('mpc_results/cheetah.pkl')
