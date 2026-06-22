import numpy as np
import matplotlib.pyplot as plt
import torch


def sample_line(n_points):
    ts = np.linspace(0, 1, n_points)
    return ts

def sample_simplex_grid(n_points):
    """
    Samples (t, s) over a 2D simplex:
    t >= 0, s >= 0, t + s <= 1
    """
    ts = []
    ss = []

    grid = np.linspace(0, 1, n_points)

    for t in grid:
        for s in grid:
            if t + s <= 1.0:
                ts.append(t)
                ss.append(s)
    return np.array(ts), np.array(ss)

def evaluate_line(agent, algo, env, n_points=50):

    ts = sample_line(n_points)

    all_returns = []

    for t in ts:

        w = np.zeros(agent.reward_dim, dtype=np.float32)
        w[0] = t
        w[1] = 1.0 - t   # ONLY 2 OBJECTIVES

        obs, _ = env.reset()
        done = False

        ep_return = np.zeros(agent.reward_dim, dtype=np.float32)

        while not done:

            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            w_tensor = torch.tensor(w, dtype=torch.float32)

            with torch.no_grad():
                if "gpi" in algo.lower():
                    action = agent.gpi_action(obs_tensor, w_tensor)
                elif "env" in algo.lower():
                    q = agent.q_net(obs_tensor.unsqueeze(0), w_tensor.unsqueeze(0))
                    action = (q * w_tensor.unsqueeze(1)).sum(dim=2).argmax(dim=1).item()

            obs, vec_reward, terminated, truncated, _ = env.step(action)

            ep_return += vec_reward
            done = terminated or truncated

        all_returns.append(ep_return)

    return ts, np.array(all_returns)

def evaluate_simplex(agent, algo, env, n_points=10):

    ts, ss = sample_simplex_grid(n_points)

    all_returns = []

    for t, s in zip(ts, ss):

        w = np.zeros(agent.reward_dim, dtype=np.float32)
        w[0] = t
        w[1] = s
        w[2] = 1.0 - t - s   # simplex constraint

        print(f"Evaluating w = {w}")

        obs, _ = env.reset()
        done = False

        ep_return = np.zeros(agent.reward_dim, dtype=np.float32)

        while not done:

            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            w_tensor = torch.tensor(w, dtype=torch.float32)

            with torch.no_grad():
                if "gpi" in algo.lower():
                    action = agent.gpi_action(obs_tensor, w_tensor)
                elif "env" in algo.lower():
                    q = agent.q_net(obs_tensor.unsqueeze(0), w_tensor.unsqueeze(0))
                    action = (q * w_tensor.unsqueeze(1)).sum(dim=2).argmax(dim=1).item()
                

            obs, vec_reward, terminated, truncated, _ = env.step(action)

            ep_return += vec_reward
            done = terminated or truncated

        all_returns.append(ep_return)

    return np.array(ts), np.array(ss), np.array(all_returns)


def plot_preferences(agent, algo, env, n_points=50, exp_note=""):

    dim = agent.reward_dim

    if dim == 2:

        ts, returns = evaluate_line(agent, algo, env, n_points)

        plt.figure(figsize=(7, 5))

        for obj in range(dim):
            plt.plot(ts, returns[:, obj], label=f"Obj {obj}")

        plt.xlabel("t (w0)")
        plt.ylabel("Return")
        plt.title("Preference Line (2-objective MORL)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("pref_line_" + exp_note + ".png")

    else:

        ts, ss, returns = evaluate_simplex(agent, algo, env, n_points)

        fig, axes = plt.subplots(
            1,
            dim,
            figsize=(6 * dim, 5),
            sharex=True,
            sharey=True
        )

        if dim == 1:
            axes = [axes]

        for obj, ax in enumerate(axes):

            sc = ax.scatter(
                ts,
                ss,
                c=returns[:, obj],
                cmap="viridis"
            )

            ax.set_title(f"Objective {obj}")
            ax.set_xlabel("t (w0)")
            ax.set_ylabel("s (w1)")

            plt.colorbar(sc, ax=ax)

        plt.tight_layout()
        plt.savefig("pref_simplex_" + exp_note + ".png")