# train_sac_robotarium_att_def.py
# SAC + Gaussian PenaltyNet, with entropy term (与原 v6_3_final 结构一致)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed, will not save .mat logs.")

# 只导入环境，不导入那里的 PenaltyNet
from compcbf_env_v2 import UnicycleHOCBFEnvRobotarium as UnicycleHOCBFEnv


# ---------------- Replay Buffer ----------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# ---------------- SAC Actor / Critic ----------------
class ActorGaussian(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(128, 128)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes),
                       activation=nn.ReLU,
                       output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = self.act_limit * y_t

        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.act_limit * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mu_action = self.act_limit * torch.tanh(mu)
        return action, log_prob, mu_action


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1],
                     activation=nn.ReLU,
                     output_activation=nn.Identity)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit, device,
                 gamma=0.99, polyak=0.995, lr=3e-4, alpha=0.1):

        self.device = device
        self.gamma = gamma
        self.polyak = polyak
        self.act_limit = act_limit
        self.alpha = alpha

        self.actor = ActorGaussian(obs_dim, act_dim, act_limit).to(device)
        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.pi_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, std = self.actor(obs_t)
                action = self.act_limit * torch.tanh(mu)
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=128):
        batch = replay_buffer.sample_batch(batch_size)
        obs = batch['obs'].to(self.device)
        obs2 = batch['obs2'].to(self.device)
        act = batch['act'].to(self.device)
        rew = batch['rew'].unsqueeze(-1).to(self.device)
        done = batch['done'].unsqueeze(-1).to(self.device)

        # --- 1. Q 更新 ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(obs2)
            q1_target_val = self.q1_target(obs2, next_action)
            q2_target_val = self.q2_target(obs2, next_action)
            q_target_min = torch.min(q1_target_val, q2_target_val)
            target = rew + self.gamma * (1 - done) * (q_target_min - self.alpha * next_log_prob)

        q1_val = self.q1(obs, act)
        q2_val = self.q2(obs, act)
        q1_loss = ((q1_val - target) ** 2).mean()
        q2_loss = ((q2_val - target) ** 2).mean()

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # --- 2. Actor 更新 ---
        new_action, log_prob, _ = self.actor.sample(obs)
        q1_pi = self.q1(obs, new_action)
        q2_pi = self.q2(obs, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = (self.alpha * log_prob - q_pi).mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        # --- 3. target 网络 ---
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return q1_loss.item(), q2_loss.item(), pi_loss.item()


# --------------- Gaussian PenaltyNet（带熵） ----------------
class GaussianPenaltyNet(nn.Module):
    """
    输入: obs (B, obs_dim)
    输出: 对 latent z 的高斯分布 N(mu, sigma^2)，维度 D = 2*num_obs + 1
    通过 softplus(z) + eps 得到真正的 CBF 参数 (k1_vec, k2_vec, alpha_c) > 0
    """
    def __init__(self, obs_dim, num_obs, hidden=64):
        super().__init__()
        self.num_obs = num_obs
        self.latent_dim = 2 * num_obs + 1

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(hidden, self.latent_dim)
        self.log_std_layer = nn.Linear(hidden, self.latent_dim)

    def forward(self, obs):
        h = self.backbone(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mu, log_std

    def sample_params(self, obs):
        mu, log_std = self(obs)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + std * eps
        params = F.softplus(z, beta=0.5) + 1e-3
        return params, mu, log_std


# ---------------- 训练主函数 ----------------
def train_sac_penalty(num_episodes=300,
                      max_ep_steps=5000,
                      replay_size=int(5e4),
                      start_steps=2000,
                      update_after=1000,
                      update_every=1,
                      batch_size=32,
                      seed=0,
                      pen_lr=3e-5):

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = UnicycleHOCBFEnv(T_max=max_ep_steps * 0.01,
                           safe_margin=0.1)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    act_limit = 2.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    agent = SACAgent(obs_dim, act_dim, act_limit, device=device)
    penalty_net = GaussianPenaltyNet(obs_dim, env.num_obs).to(device)
    pen_opt = optim.Adam(penalty_net.parameters(), lr=pen_lr)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)

    total_steps = 0
    all_returns = []
    all_q1_loss = []
    all_q2_loss = []
    all_pi_loss = []
    all_pn_loss = []

    all_r_dist = []
    all_r_heading = []
    all_r_time = []
    all_r_near_goal = []

    alpha_e = 1e-3
    beta_B = 0.1
    lambda_reg = 1e-4

    for ep in range(num_episodes):
        obs = env.reset().astype(np.float32)

        ep_ret = 0.0
        ep_q1_list, ep_q2_list, ep_pi_list = [], [], []

        state_pen_list = []
        safe_u_list = []

        ep_r_dist_sum = 0.0
        ep_r_heading_sum = 0.0
        ep_r_time_sum = 0.0
        ep_r_near_goal_sum = 0.0
        ep_step_count = 0

        for t in range(max_ep_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # 1) 行为策略：先采样 RL 动作（标称控制）
            if total_steps < start_steps:
                act = np.random.uniform(-act_limit, act_limit, size=act_dim).astype(np.float32)
            else:
                act = agent.select_action(obs, deterministic=False).astype(np.float32)

            # 2) PenaltyNet 采样 CBF 参数（Gaussian + softplus）
            with torch.no_grad():
                pen_params, _, _ = penalty_net.sample_params(obs_t)
            pen_params_np = pen_params.cpu().numpy().squeeze(0)

            k1_vec = pen_params_np[:env.num_obs]
            k2_vec = pen_params_np[env.num_obs:2 * env.num_obs]
            alpha_c = pen_params_np[-1]

            # 3) 与环境交互：获得安全控制 omega_hat 和奖励
            next_obs, r, done, info = env.step(act[0], k1_vec, k2_vec, alpha_c)
            next_obs = next_obs.astype(np.float32)

            replay_buffer.store(obs, act, r, next_obs, float(done))
            ep_ret += r
            total_steps += 1
            ep_step_count += 1

            ep_r_dist_sum += info.get("r_dist", 0.0)
            ep_r_heading_sum += info.get("r_heading", 0.0)
            ep_r_time_sum += info.get("r_time", 0.0)
            ep_r_near_goal_sum += info.get("r_near_goal", 0.0)

            state_pen_list.append(obs.copy())
            safe_u_list.append(info["omega_hat"])

            obs = next_obs

            # 4) SAC 更新
            if total_steps >= update_after and total_steps % update_every == 0:
                q1_l, q2_l, pi_l = agent.update(replay_buffer, batch_size)
                ep_q1_list.append(q1_l)
                ep_q2_list.append(q2_l)
                ep_pi_list.append(pi_l)

            if done:
                break

        # ---- episode 级别统计 ----
        all_returns.append(ep_ret)
        all_q1_loss.append(np.mean(ep_q1_list) if len(ep_q1_list) > 0 else 0.0)
        all_q2_loss.append(np.mean(ep_q2_list) if len(ep_q2_list) > 0 else 0.0)
        all_pi_loss.append(np.mean(ep_pi_list) if len(ep_pi_list) > 0 else 0.0)

        if ep_step_count > 0:
            all_r_dist.append(ep_r_dist_sum / ep_step_count)
            all_r_heading.append(ep_r_heading_sum / ep_step_count)
            all_r_time.append(ep_r_time_sum / ep_step_count)
            all_r_near_goal.append(ep_r_near_goal_sum / ep_step_count)
        else:
            all_r_dist.append(0.0)
            all_r_heading.append(0.0)
            all_r_time.append(0.0)
            all_r_near_goal.append(0.0)

        # ---- PenaltyNet 更新 ----
        if len(state_pen_list) > 1:
            states_pen_np = np.array(state_pen_list, dtype=np.float32)
            safe_u_np = np.array(safe_u_list, dtype=np.float32)

            states_pen = torch.as_tensor(states_pen_np, dtype=torch.float32, device=device)
            u_safe = torch.as_tensor(safe_u_np, dtype=torch.float32, device=device).unsqueeze(-1)

            mu_z, log_std_z = penalty_net(states_pen)
            const = 0.5 * np.log(2 * np.pi * np.e)
            entropy_per_sample = torch.sum(const + log_std_z, dim=1)
            H_mean = entropy_per_sample.mean()

            q1_val = agent.q1(states_pen, u_safe)
            q2_val = agent.q2(states_pen, u_safe)
            q_min = torch.min(q1_val, q2_val)
            loss_q = - q_min.mean()

            u_prev = torch.roll(u_safe, shifts=1, dims=0)
            u_prev[0] = u_safe[0]
            loss_B = ((u_safe - u_prev) ** 2).mean()

            params_sample, _, _ = penalty_net.sample_params(states_pen)
            loss_reg = lambda_reg * (params_sample ** 2).mean()

            pn_loss = -alpha_e * H_mean + loss_q + beta_B * loss_B + loss_reg

            pen_opt.zero_grad()
            pn_loss.backward()
            nn.utils.clip_grad_norm_(penalty_net.parameters(), 0.5)
            pen_opt.step()

            pn_loss_val = pn_loss.item()
        else:
            pn_loss_val = 0.0

        all_pn_loss.append(pn_loss_val)

        if ep % 10 == 0:
            print(f"[Ep {ep:03d}] Return={ep_ret:.2f}, "
                  f"Q1={all_q1_loss[-1]:.4f}, Q2={all_q2_loss[-1]:.4f}, "
                  f"Pi={all_pi_loss[-1]:.4f}, PN={pn_loss_val:.4f}, "
                  f"r_dist={all_r_dist[-1]:.3f}, r_head={all_r_heading[-1]:.3f}, "
                  f"r_time={all_r_time[-1]:.3f}, r_ng={all_r_near_goal[-1]:.3f}")

    # ---------- 保存模型 ----------
    torch.save(agent.actor.state_dict(), "actor_sac_robotarium_att_def_v2.pth")
    torch.save(agent.q1.state_dict(), "q1_sac_robotarium_att_def_q1_v2.pth")
    torch.save(agent.q2.state_dict(), "q2_sac_robotarium_att_def_q2_v2.pth")
    torch.save(penalty_net.state_dict(), "penalty_sac_robotarium_att_def_v2.pth")

    # ---------- 保存训练日志 ----------
    if HAS_SCIPY:
        sio.savemat("train_log_sac_robotarium_att_def_v2.mat", {
            "returns": np.array(all_returns, dtype=np.float64),
            "q1_loss": np.array(all_q1_loss, dtype=np.float64),
            "q2_loss": np.array(all_q2_loss, dtype=np.float64),
            "pi_loss": np.array(all_pi_loss, dtype=np.float64),
            "pn_loss": np.array(all_pn_loss, dtype=np.float64),
            "r_dist": np.array(all_r_dist, dtype=np.float64),
            "r_heading": np.array(all_r_heading, dtype=np.float64),
            "r_time": np.array(all_r_time, dtype=np.float64),
            "r_near_goal": np.array(all_r_near_goal, dtype=np.float64),
        })
        print("Saved train_log_sac_robotarium_att_def_v2.mat")

    # ---------- 画奖励分解曲线 ----------
    episodes = np.arange(len(all_returns))

    plt.figure()
    plt.plot(episodes, all_r_dist, label="r_dist (distance)")
    plt.plot(episodes, all_r_heading, label="r_heading (heading)")
    plt.plot(episodes, all_r_time, label="r_time (time)")
    plt.plot(episodes, all_r_near_goal, label="r_near_goal (near goal)")
    plt.xlabel("Episode")
    plt.ylabel("Average reward component per step")
    plt.title("Reward Decomposition during Training (Robotarium attacker-defender)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return env, agent, penalty_net, all_returns


if __name__ == "__main__":
    env, agent, penalty_net, returns = train_sac_penalty(num_episodes=200)
