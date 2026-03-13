# test_sac_robotarium_att_def.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from compcbf_env_v2 import UnicycleHOCBFEnvRobotarium as UnicycleHOCBFEnv

# ==== Robotarium 原生可视化相关 ====
import rps.robotarium as robotarium
from rps.utilities.controllers import create_clf_unicycle_position_controller

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ActorGaussian(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(128, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
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

        return action, log_prob, self.act_limit * torch.tanh(mu)


class GaussianPenaltyNet(nn.Module):
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
        return mu, torch.clamp(log_std, -5.0, 2.0)

    def sample_params(self, obs):
        mu, log_std = self(obs)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + std * eps
        params = F.softplus(z, beta=0.5) + 1e-3
        return params, mu, log_std


def rollout_one_episode(env, actor, penalty_net, max_steps=5000,
                        device="cpu", act_limit=2.0):

    obs = env.reset().astype(np.float32)

    # ====== 轨迹记录 ======
    traj_attacker = [obs[:3].copy()]
    traj_defender = [obs[4:].copy()]
    omega_hist, omega_hat_hist, u_nom_hist = [], [], []
    H_hist, min_dist_hist, min_safe_hist = [], [], []
    h_geom_min_hist, psi2_min_hist = [], []
    phi_min_hist, phi_max_hist = [], []
    reward_hist = []

    # CBF 参数
    k1_hist, k2_hist, alpha_hist = [], [], []
    relax_hist = []

    # 奖励分解
    r_dist_hist, r_heading_hist = [], []
    r_time_hist, r_near_goal_hist = [], []

    for t in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # Actor（用 mean 动作）
        with torch.no_grad():
            mu, std = actor(obs_t)
            u_nom = act_limit * torch.tanh(mu)
            u_nom = float(u_nom.cpu().numpy()[0])

        # PenaltyNet（不采样，直接用均值 softplus）
        with torch.no_grad():
            mu_p, log_std_p = penalty_net(obs_t)
            params = F.softplus(mu_p, beta=0.5) + 1e-3
            params = params.cpu().numpy().squeeze(0)

        num_obs = env.num_obs
        k1_vec = params[:num_obs]
        k2_vec = params[num_obs:2 * num_obs]
        alpha_c = params[-1]

        # 环境 step（纯仿真）
        next_obs, r, done, info = env.step(u_nom, k1_vec, k2_vec, alpha_c)
        next_obs = next_obs.astype(np.float32)

        traj_attacker.append(next_obs[:3].copy())
        traj_defender.append(next_obs[4:].copy())
        obs = next_obs

        omega_hist.append(info["omega"])
        omega_hat_hist.append(info["omega_hat"])
        u_nom_hist.append(info["u_nom"])
        H_hist.append(info["H"])
        min_dist_hist.append(info["min_dist"])
        min_safe_hist.append(info["min_safe_radius"])
        h_geom_min_hist.append(info["h_geom_min"])
        psi2_min_hist.append(info["psi2_min"])
        phi_min_hist.append(info["phi_min"])
        phi_max_hist.append(info["phi_max"])
        reward_hist.append(info["reward"])

        k1_hist.append(info["k1"])
        k2_hist.append(info["k2"])
        alpha_hist.append(info["alpha_c"])

        relax_hist.append(abs(info["omega_hat"] - info["u_nom"]))

        r_dist_hist.append(info["r_dist"])
        r_heading_hist.append(info["r_heading"])
        r_time_hist.append(info["r_time"])
        r_near_goal_hist.append(info["r_near_goal"])

        if done:
            break

    traj_attacker = np.array(traj_attacker)
    traj_defender = np.array(traj_defender)
    time = np.arange(len(omega_hist)) * env.dt

    return dict(
        traj_attacker=traj_attacker,
        traj_defender=traj_defender,
        time=time,
        omega=np.array(omega_hist),
        omega_hat=np.array(omega_hat_hist),
        u_nom=np.array(u_nom_hist),
        H=np.array(H_hist),
        min_dist=np.array(min_dist_hist),
        min_safe=np.array(min_safe_hist),
        h_geom_min=np.array(h_geom_min_hist),
        psi2_min=np.array(psi2_min_hist),
        phi_min=np.array(phi_min_hist),
        phi_max=np.array(phi_max_hist),
        reward=np.array(reward_hist),

        k1=np.array(k1_hist),
        k2=np.array(k2_hist),
        alpha=np.array(alpha_hist),
        relax=np.array(relax_hist),

        r_dist=np.array(r_dist_hist),
        r_heading=np.array(r_heading_hist),
        r_time=np.array(r_time_hist),
        r_near_goal=np.array(r_near_goal_hist),
    )


def plot_all(env, R, train_log_mat=None, save_prefix="test_sac_robotarium_att_def"):

    traj_a = R["traj_attacker"]
    traj_d = R["traj_defender"]
    time = R["time"]

    omega, omega_hat, u_nom = R["omega"], R["omega_hat"], R["u_nom"]
    H = R["H"]
    min_dist, min_safe = R["min_dist"], R["min_safe"]
    h_geom_min = R["h_geom_min"]
    psi2_min = R["psi2_min"]
    phi_min, phi_max = R["phi_min"], R["phi_max"]

    reward = R["reward"]

    k1, k2, alpha, relax = R["k1"], R["k2"], R["alpha"], R["relax"]

    r_dist = R["r_dist"]
    r_heading = R["r_heading"]
    r_time_comp = R["r_time"]
    r_near_goal = R["r_near_goal"]

    num_obs = env.num_obs

    # ========== 1) 轨迹 ==========
    fig, ax = plt.subplots()

    ax.plot(traj_a[:, 0], traj_a[:, 1], label="Attacker")
    ax.plot(traj_d[:, 0], traj_d[:, 1], label="Defender")

    ax.scatter(traj_a[0, 0], traj_a[0, 1], c='r', label="Start A")
    # ---------- 目标区域（真实半径） ----------
    goal_center = env.goal
    goal_radius = env.goal_tol    # 若环境变量名字是 env.goal_R，也可替换这里

    # 真实目标圆
    goal_patch = Circle(
        goal_center, goal_radius,
        edgecolor='green',
        facecolor='none',
        linestyle='--',
        linewidth=1.5,
        label="Goal region"
    )
    ax.add_patch(goal_patch)

    # 目标点（中心）
    ax.scatter(goal_center[0], goal_center[1], c='g', s=40)

    # 目标标签
    ax.text(goal_center[0], goal_center[1] + goal_radius + 0.05,
        "Target", color='green', ha='center', fontsize=9)


    # 静态障碍
    th = np.linspace(0, np.pi*2, 200)
    for i in range(env._num_static_obs):
        cx, cy = env.static_centers[i]
        r0 = env.static_radii[i]
        ax.plot(cx + np.cos(th)*r0, cy + np.sin(th)*r0, 'r--')

    cd = traj_d[-1, :]
    Rc = env.capture_radius
    ax.plot(cd[0] + Rc * np.cos(th),
            cd[1] + Rc * np.sin(th),
            'b--', label="Capture radius (final)")

    ax.set_aspect("equal")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_trajectory.png", dpi=300)

    # ========== 2) 控制量 ==========
    fig, ax = plt.subplots()
    ax.plot(time, u_nom, label="u_nom")
    ax.plot(time, omega_hat, '--', label="omega_hat")
    ax.plot(time, omega, label="omega")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_controls.png", dpi=300)

    # ========== 3) CBF 原始 ==========
    fig, ax = plt.subplots()
    ax.plot(time, h_geom_min, label="min h_geom")
    ax.plot(time, phi_min, label="phi_min")
    ax.plot(time, phi_max, label="phi_max")
    ax.axhline(0, linestyle="--")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_cbf_primitives.png", dpi=300)

    # ========== 4) psi2 / H ==========
    fig, ax = plt.subplots()
    ax.plot(time, psi2_min, label="min psi2")
    ax.plot(time, H, label="H")
    ax.axhline(0, linestyle="--")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_psi2_H.png", dpi=300)

    # ========== 5) 奖励分解 ==========
    fig, ax = plt.subplots()
    ax.plot(time, r_dist, label="r_dist")
    ax.plot(time, r_time_comp, label="r_time")
    ax.plot(time, r_heading, label="r_heading")
    ax.plot(time, r_near_goal, label="r_near_goal")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_reward_decomposition.png", dpi=300)

    # ========== 6) PenaltyNet 参数 ==========
    fig, ax = plt.subplots()
    for i in range(num_obs):
        ax.plot(time, k1[:, i], label=f"k1_{i+1}")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_k1.png", dpi=300)

    fig, ax = plt.subplots()
    for i in range(num_obs):
        ax.plot(time, k2[:, i], label=f"k2_{i+1}")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_k2.png", dpi=300)

    fig, ax = plt.subplots()
    ax.plot(time, alpha)
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_alpha.png", dpi=300)

    # ========== 7) 约束松弛 ==========
    fig, ax = plt.subplots()
    ax.plot(time, relax)
    ax.set_title("|omega_hat - u_nom|")
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_relax.png", dpi=300)

    # ========== 8) 训练 returns ==========
    if train_log_mat is not None and "returns" in train_log_mat:
        fig, ax = plt.subplots()
        ax.plot(train_log_mat["returns"].squeeze(), label="Episode Return")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title("Training Episode Reward Curve")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        fig.savefig(f"{save_prefix}_train_returns.png", dpi=300)

    plt.show()


# ==========================================================
#  新增：用 Robotarium 原生 GUI 回放 RL 生成的轨迹
#  - 显示静态圆障碍物
#  - 显示捕捉半径圈
# ==========================================================
def robotarium_playback(env, R):
    traj_a = R["traj_attacker"]
    traj_d = R["traj_defender"]
    time = R["time"]

    # ===== Robotarium 初始化 =====
    N = 2  # attacker + defender
    initial_conditions = np.array([
        [traj_a[0, 0], traj_d[0, 0]],
        [traj_a[0, 1], traj_d[0, 1]],
        [traj_a[0, 2], traj_d[0, 2]],
    ])

    print(traj_a[0, 2], traj_d[0, 2])

    r = robotarium.Robotarium(
        number_of_robots=N,
        show_figure=True,
        initial_conditions=initial_conditions,
        sim_in_real_time=True
    )

    ax = r.figure.axes[0]
    ax.set_aspect('equal')

    if not hasattr(env, "x_range"):
        env.x_range = [-1.6, 1.6]
    if not hasattr(env, "y_range"):
        env.y_range = [-1.0, 1.0]

    ax.set_xlim(env.x_range)
    ax.set_ylim(env.y_range)

    # ===== 静态障碍物 =====
    for i in range(env._num_static_obs):
        static_center = env.static_centers[i]
        static_radius = env.static_radii[i]

        static_obs_patch = Circle(
            static_center, static_radius,
            color='gray', alpha=0.3, zorder=1
        )
        ax.add_patch(static_obs_patch)

        # Obstacle label
        ax.text(
            static_center[0], static_center[1],
            f"Obs {i+1}",
            color="black", fontsize=9,
            ha="center", va="center",
            zorder=10
        )


    # ===== 捕捉半径圈（随防守者更新） =====
    capture_R = env.capture_radius
    capture_patch = Circle(
        (traj_d[0, 0], traj_d[0, 1]), capture_R,
        edgecolor='blue', facecolor='none',
        linestyle='--', linewidth=1.5, zorder=2
    )
    ax.add_patch(capture_patch)

    # ===== 目标点 & 标签 =====
    
    goal_center = env.goal
    goal_radius = env.goal_tol   # 或 env.goal_R

    target_patch = Circle(
        goal_center, goal_radius,
        edgecolor='green',
        facecolor='none',
        linestyle='--',
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(target_patch)

    # ===== 目标标签 =====
    target_label = ax.text(
        goal_center[0], goal_center[1] ,
        "Target",
        fontsize=10,
        color="green",
        ha="center",
        zorder=10
    )

    # === 创建标签文本（Attacker / Defender） ===
    attacker_label = ax.text(
        traj_a[0,0], traj_a[0,1] - 0.08,
        "Attacker",
        color="red",
        fontsize=10,
        ha="center",
        zorder=10
    )
    defender_label = ax.text(
        traj_d[0,0], traj_d[0,1] + 0.08,
        "Defender",
        color="blue",
        fontsize=10,
        ha="center",
        zorder=10
    )

    # 使用 Robotarium 自带 unicycle 位置控制器
    uni_pos_ctrl = create_clf_unicycle_position_controller()

    # ===== 回放轨迹 =====
    T = len(time)
    for k in range(T):
        x = r.get_poses()  # 3xN

        goals = np.array([
            [traj_a[k, 0], traj_d[k, 0]],
            [traj_a[k, 1], traj_d[k, 1]],
        ])

        dxu = uni_pos_ctrl(x, goals)

        r.set_velocities(np.arange(N), dxu)

        # ---- 更新捕捉圈中心 ----
        capture_patch.center = (x[0, 1], x[1, 1])

        # ---- 更新标签位置 ----
        attacker_label.set_position((x[0,0], x[1,0] - 0.08))
        defender_label.set_position((x[0,1], x[1,1] + 0.08))

        # 刷新界面
        r.step()

    r.call_at_scripts_end()
    print("Robotarium playback finished.")


def test_sac_robotarium_att_def():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = UnicycleHOCBFEnv(T_max=50.0, safe_margin=0.1)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    num_obs = env.num_obs
    act_limit = 2.0

    actor = ActorGaussian(obs_dim, act_dim, act_limit).to(device)
    pn = GaussianPenaltyNet(obs_dim, num_obs).to(device)

    actor.load_state_dict(torch.load("actor_sac_robotarium_att_def_v2.pth", map_location=device))
    pn.load_state_dict(torch.load("penalty_sac_robotarium_att_def_v2.pth", map_location=device))

    actor.eval()
    pn.eval()

    # 1) 先在仿真环境中 roll 一条轨迹 + 记录各类信息
    R = rollout_one_episode(env, actor, pn, max_steps=5000, device=device)

    # 2) 可选：画静态分析图
    train_log_mat = None
    if HAS_SCIPY and os.path.exists("train_log_sac_robotarium_att_def_v2.mat"):
        train_log_mat = sio.loadmat("train_log_sac_robotarium_att_def_v2.mat")
    plot_all(env, R, train_log_mat=train_log_mat, save_prefix="test_sac_robotarium_att_def")

    # 3) 使用 Robotarium 原生 GUI 回放轨迹 + 显示圆障碍 & 捕捉半径
    robotarium_playback(env, R)


if __name__ == "__main__":
    test_sac_robotarium_att_def()
