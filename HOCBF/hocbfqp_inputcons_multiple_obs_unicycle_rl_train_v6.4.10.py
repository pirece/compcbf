# train_sac_hocbf_qp.py
# SAC + Gaussian PenaltyNet for original HOCBF-QP
# - 使用 hocbfqp_multiple_obs_unicycle_rl_env_v6410 环境
# - PenaltyNet 输出 k1_vec, k2_vec（无 alpha_c）
# - PenaltyNet loss = Q-loss + smooth + entropy + regularization

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

# ====== HOCBF-QP 环境 ======
from hocbfqp_env_v2 import \
    UnicycleHOCBFEnvRobotarium as UnicycleHOCBFEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# Replay Buffer
# ============================================================
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
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                for k, v in batch.items()}


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# ============================================================
# SAC Actor / Critic
# ============================================================
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
        """
        obs: [B, obs_dim]
        return:
          action:  [B, act_dim]
          log_prob:[B, 1]
          mu_action: [B, act_dim]
        """
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
        return self.q(torch.cat([obs, act], dim=-1))  # [B,1]


class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit,
                 gamma=0.99, polyak=0.995, lr=3e-4, alpha=0.1):

        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.act_limit = act_limit

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
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(obs_t)
                action = self.act_limit * torch.tanh(mu)
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=128):
        batch = replay_buffer.sample_batch(batch_size)
        obs = batch["obs"]
        obs2 = batch["obs2"]
        act = batch["act"]
        rew = batch["rew"].unsqueeze(-1)    # [B,1]
        done = batch["done"].unsqueeze(-1)  # [B,1]

        # ---------- Q-network 更新 ----------
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(obs2)
            q1_target_val = self.q1_target(obs2, next_action)
            q2_target_val = self.q2_target(obs2, next_action)
            q_target_min = torch.min(q1_target_val, q2_target_val)
            target = rew + self.gamma * (1 - done) * \
                (q_target_min - self.alpha * next_log_prob)

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

        # ---------- Actor 更新 ----------
        new_action, log_prob, _ = self.actor.sample(obs)
        q1_pi = self.q1(obs, new_action)
        q2_pi = self.q2(obs, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = (self.alpha * log_prob - q_pi).mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        # ---------- Target 网络软更新 ----------
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(),
                                 self.q1_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.q2.parameters(),
                                 self.q2_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return q1_loss.item(), q2_loss.item(), pi_loss.item()


# ============================================================
# Gaussian PenaltyNet (只输出 k1_vec, k2_vec)
# ============================================================
class GaussianPenaltyNet(nn.Module):
    """
    输入: obs (B, obs_dim)
    输出: 2*num_obs 维 z → softplus(z) 作为 (k1_vec, k2_vec) > 0
    """
    def __init__(self, obs_dim, num_obs, hidden=64):
        super().__init__()
        self.num_obs = num_obs
        self.out_dim = 2 * num_obs

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(hidden, self.out_dim)
        self.log_std_layer = nn.Linear(hidden, self.out_dim)

    def forward(self, obs):
        h = self.backbone(obs)
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), -5.0, 2.0)
        return mu, log_std

    def sample_params(self, obs):
        mu, log_std = self(obs)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + std * eps
        params = F.softplus(z, beta=0.5) + 1e-3  # >0
        return params, mu, log_std


# ============================================================
# PenaltyNet 任务驱动 Loss（不使用 H、omega_hat、psi2_min）
# ============================================================
def penaltynet_loss(penalty_net, agent, states, u_safe_list,
                    lambda_Q=1.0, lambda_smooth=0.1,
                    lambda_reg=1e-4, lambda_ent=1e-3):
    """
    states: (T, obs_dim)  episode 内所有状态
    u_safe_list: (T,)     episode 内的安全控制 omega（QP 输出）
    """

    # -------- 1) 熵：鼓励参数分布有一定探索 --------
    mu_z, log_std_z = penalty_net(states)
    std_z = torch.exp(log_std_z)

    # 高斯熵公式：0.5 * log(2πeσ^2)
    ent_per_sample = torch.sum(
        0.5 * torch.log(2 * torch.pi * torch.e * std_z**2),
        dim=1
    )
    loss_entropy = - lambda_ent * ent_per_sample.mean()

    # -------- 2) Q-loss：鼓励选择 Q(s,u_safe) 大的参数 --------
    u_safe = torch.as_tensor(
        u_safe_list, dtype=torch.float32,
        device=states.device
    ).unsqueeze(-1)
    q1 = agent.q1(states, u_safe)
    q2 = agent.q2(states, u_safe)
    q_min = torch.min(q1, q2)
    loss_Q = - lambda_Q * q_min.mean()

    # -------- 3) 平滑性：鼓励 u_safe 在时间上平滑 --------
    u_prev = torch.roll(u_safe, shifts=1, dims=0)
    u_prev[0] = u_safe[0]
    loss_smooth = lambda_smooth * ((u_safe - u_prev) ** 2).mean()

    # -------- 4) 正则：防止参数过大 --------
    params_sample, _, _ = penalty_net.sample_params(states)
    loss_reg = lambda_reg * (params_sample ** 2).mean()

    # 总损失
    pn_loss = loss_Q + loss_smooth + loss_reg + loss_entropy
    return pn_loss


# ============================================================
# Training Loop
# ============================================================
def train_sac_penalty(num_episodes=200,
                      max_ep_steps=5000,
                      replay_size=int(5e4),
                      start_steps=2000,
                      update_after=1000,
                      update_every=1,
                      batch_size=64,
                      pen_lr=3e-5,
                      seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = UnicycleHOCBFEnv(T_max=max_ep_steps * 0.01)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    act_limit = 2.0

    agent = SACAgent(obs_dim, act_dim, act_limit)
    penalty_net = GaussianPenaltyNet(obs_dim, env.num_obs).to(device)
    pen_opt = optim.Adam(penalty_net.parameters(), lr=pen_lr)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)

    total_steps = 0

    all_returns = []
    all_q1_loss = []
    all_q2_loss = []
    all_pi_loss = []
    all_pn_loss = []

    # 奖励分解（episode 平均）
    all_r_dist = []
    all_r_time = []

    print("Start training (HOCBF-QP + task-driven PenaltyNet)...")

    for ep in range(num_episodes):
        obs = env.reset().astype(np.float32)

        ep_ret = 0.0
        ep_q1_list, ep_q2_list, ep_pi_list = [], [], []
        ep_pn_list = []

        # 奖励分解统计
        ep_r_dist_sum = 0.0
        ep_r_time_sum = 0.0
        ep_step_count = 0
        ep_infeasible_cnt = 0

        # PenaltyNet 用的轨迹数据
        state_pen_list = []
        u_safe_list = []

        for t in range(max_ep_steps):
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # 1) RL 标称动作
            if total_steps < start_steps:
                act = np.random.uniform(-act_limit, act_limit,
                                        size=act_dim).astype(np.float32)
            else:
                act = agent.select_action(obs, deterministic=False).astype(np.float32)

            # 2) PenaltyNet 输出 HOCBF 参数（k1_vec, k2_vec）
            with torch.no_grad():
                pen_params, _, _ = penalty_net.sample_params(obs_t)
            pen_params_np = pen_params.cpu().numpy().squeeze(0)
            k1_vec = pen_params_np[:env.num_obs]
            k2_vec = pen_params_np[env.num_obs:]

            # 3) 与 HOCBF-QP 环境交互
            next_obs, r, done, info = env.step(act[0], k1_vec, k2_vec)
            next_obs = next_obs.astype(np.float32)

            # 存入 ReplayBuffer
            replay_buffer.store(obs, act, r, next_obs, float(done))

            # 记录奖励
            r_dist = info.get("r_dist", 0.0)
            r_time = info.get("r_time", 0.0)
            ep_infeasible_cnt += info.get("infeasible_cnt", 0) 
            ep_r_dist_sum += r_dist
            ep_r_time_sum += r_time
            ep_step_count += 1
            

            # 每步打印奖励分解
            #print(f"[Ep {ep:03d} Step {t:04d}] "f"r_dist={r_dist:.3f}, r_time={r_time:.3f}, reward={r:.3f}")

            # PenaltyNet 轨迹数据
            state_pen_list.append(obs.copy())
            # u_safe = QP 输出的实际控制，假设 info["omega"]，若不存在用 act[0]
            u_safe_list.append(info.get("omega", float(act[0])))

            obs = next_obs
            ep_ret += r
            total_steps += 1

            # 4) SAC 更新
            if (total_steps >= update_after) and \
               (total_steps % update_every == 0):
                q1_l, q2_l, pi_l = agent.update(replay_buffer, batch_size)
                ep_q1_list.append(q1_l)
                ep_q2_list.append(q2_l)
                ep_pi_list.append(pi_l)

            if done:
                break

        # ---- episode 级别统计 ----
        all_returns.append(ep_ret)
        all_q1_loss.append(np.mean(ep_q1_list) if ep_q1_list else 0.0)
        all_q2_loss.append(np.mean(ep_q2_list) if ep_q2_list else 0.0)
        all_pi_loss.append(np.mean(ep_pi_list) if ep_pi_list else 0.0)

        # 奖励分解：episode 平均
        if ep_step_count > 0:
            avg_r_dist = ep_r_dist_sum 
            avg_r_time = ep_r_time_sum 
        else:
            avg_r_dist = 0.0
            avg_r_time = 0.0

        all_r_dist.append(avg_r_dist)
        all_r_time.append(avg_r_time)

        # ---- PenaltyNet 更新（用整条 episode 序列）----
        if len(state_pen_list) > 1:
            states_pen = torch.as_tensor(
                np.array(state_pen_list, dtype=np.float32),
                dtype=torch.float32, device=device
            )
            u_safe_arr = np.array(u_safe_list, dtype=np.float32)

            pn_loss = penaltynet_loss(
                penalty_net, agent,
                states_pen, u_safe_arr,
                lambda_Q=1.0,
                lambda_smooth=0.1,
                lambda_reg=1e-4,
                lambda_ent=1e-3
            )

            pen_opt.zero_grad()
            pn_loss.backward()
            nn.utils.clip_grad_norm_(penalty_net.parameters(), 0.5)
            pen_opt.step()

            pn_loss_val = pn_loss.item()
        else:
            pn_loss_val = 0.0

        all_pn_loss.append(pn_loss_val)

        print(f"[Ep {ep:03d}] "
              f"Return={ep_ret:.2f}, "
              f"Q1={all_q1_loss[-1]:.4f}, Q2={all_q2_loss[-1]:.4f}, "
              f"Pi={all_pi_loss[-1]:.4f}, PN={pn_loss_val:.4f}, "
              f"r_dist={avg_r_dist:.4f}, r_time={avg_r_time:.4f}, "
              f"infeasible_cnt={ep_infeasible_cnt:.3f}")

    # ---------- 保存模型 ----------
    torch.save(agent.actor.state_dict(), "actor_sac_hocbf_att_def_v2.pth")
    torch.save(agent.q1.state_dict(), "q1_sac_hocbf_att_def_v2.pth")
    torch.save(agent.q2.state_dict(), "q2_sac_hocbf_att_def_v2.pth")
    torch.save(penalty_net.state_dict(), "penaltynet_sac_hocbf_att_def_v2.pth")
    print("Saved actor/q1/q2/penaltynet weights.")

    # ---------- 保存训练日志 ----------
    if HAS_SCIPY:
        sio.savemat("train_log_sac_hocbf_att_def_v2.mat", {
            "returns": np.array(all_returns, dtype=np.float64),
            "q1_loss": np.array(all_q1_loss, dtype=np.float64),
            "q2_loss": np.array(all_q2_loss, dtype=np.float64),
            "pi_loss": np.array(all_pi_loss, dtype=np.float64),
            "pn_loss": np.array(all_pn_loss, dtype=np.float64),
            "r_dist": np.array(all_r_dist, dtype=np.float64),
            "r_time": np.array(all_r_time, dtype=np.float64),
        })
        print("Saved train_log_sac_hocbf_att_def_v2.mat")

    # ---------- 画回报曲线 ----------
    episodes = np.arange(len(all_returns))
    plt.figure()
    plt.plot(episodes, all_returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Return (HOCBF-QP attacker-defender)")
    plt.grid(True)
    plt.tight_layout()

    # ---------- 画奖励分解曲线 ----------
    plt.figure()
    plt.plot(episodes, all_r_dist, label="avg r_dist")
    plt.plot(episodes, all_r_time, label="avg r_time")
    plt.xlabel("Episode")
    plt.ylabel("Avg reward component per step")
    plt.title("Reward Decomposition (HOCBF-QP)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    return env, agent, penalty_net, all_returns


if __name__ == "__main__":
    env, agent, penalty_net, returns = train_sac_penalty(num_episodes=200)
