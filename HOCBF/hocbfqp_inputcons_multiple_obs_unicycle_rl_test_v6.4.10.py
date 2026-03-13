# test_sac_hocbfqp_robotarium_att_def.py
# 原始 HOCBF-QP 版本（无升阶，无 alpha_c，无 omega_hat）

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ===== 使用你的最终 ENV A =====
from hocbfqp_env_v2 import UnicycleHOCBFEnvRobotarium

# ===== Robotarium 动图 =====
import rps.robotarium as robotarium
from rps.utilities.controllers import create_clf_unicycle_position_controller

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ======================================================
# Actor（与训练完全一致，无 alpha_c）
# ======================================================
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
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = self.act_limit * y_t
        return action


# ======================================================
# PenaltyNet：只输出 k1_vec, k2_vec
# ======================================================
class GaussianPenaltyNet(nn.Module):
    def __init__(self, obs_dim, num_obs, hidden=64):
        super().__init__()
        self.num_obs = num_obs
        self.out_dim = 2 * num_obs      # no alpha_c

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
        log_std = torch.clamp(self.log_std_layer(h), -5, 2)
        return mu, log_std

    def sample_params(self, obs):
        mu, log_std = self(obs)
        z = mu                     # 测试使用均值，不采样
        return F.softplus(z, beta=0.5) + 1e-3


# ======================================================
# Rollout（不包含升阶内容）
# ======================================================
def rollout_episode(env, actor, penalty_net, device, max_steps=5000):
    obs = env.reset().astype(np.float32)

    traj_a = []
    traj_d = []
    omega_hist = []
    min_dist_hist = []
    h_geom_hist = []
    reward_hist = []
    k1_list = []
    k2_list = []
    ep_infeasible_cnt = 0
    for t in range(max_steps):
        traj_a.append(obs[:3].copy())
        traj_d.append(obs[4:].copy())

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # ---- actor mean 动作 ----
        with torch.no_grad():
            mu, _ = actor(obs_t)
            u_nom = float(actor.act_limit * torch.tanh(mu).cpu().numpy()[0])

        # ---- penalty net ----
        with torch.no_grad():
            params = penalty_net.sample_params(obs_t).cpu().numpy()[0]

        num_obs = env.num_obs
        k1 = params[:num_obs]
        k2 = params[num_obs:]

        # ---- env step（HOCBF-QP）----
        next_obs, reward, done, info = env.step(u_nom, k1, k2)
        next_obs = next_obs.astype(np.float32)

        omega_hist.append(info["omega"])
        min_dist_hist.append(info["min_dist"])
        h_geom_hist.append(info["h_geom_min"])
        reward_hist.append(reward)
        k1_list.append(k1)
        k2_list.append(k2)
        ep_infeasible_cnt += info.get("infeasible_cnt", 0) 
        
        # 为了不刷屏，可以只在有不可行的时候打印，或者注释掉
        # if info.get("infeasible_cnt", 0) > 0:
        #     print(f"[Ep {t:03d}] infeasible_cnt={ep_infeasible_cnt:03d}")
            
        obs = next_obs
        if done:
            break

    traj_a = np.array(traj_a)
    traj_d = np.array(traj_d)
    time = np.arange(len(omega_hist)) * env.dt

    return dict(
        traj_attacker=traj_a,
        traj_defender=traj_d,
        omega=np.array(omega_hist),
        time=time,
        min_dist=np.array(min_dist_hist),
        h_geom_min=np.array(h_geom_hist),
        reward=np.array(reward_hist),
        k1=np.array(k1_list),
        k2=np.array(k2_list),
    )


# ======================================================
# 轨迹分析图
# ======================================================
def plot_results(env, R, prefix="hocbfqp_test"):

    traj_a = R["traj_attacker"]
    traj_d = R["traj_defender"]
    time = R["time"]

    omega = R["omega"]
    min_dist = R["min_dist"]
    h_geom = R["h_geom_min"]
    reward = R["reward"]

    k1 = R["k1"]
    k2 = R["k2"]

    # ---- 轨迹 ----
    fig, ax = plt.subplots()
    ax.plot(traj_a[:,0], traj_a[:,1], label="Attacker")
    ax.plot(traj_d[:,0], traj_d[:,1], label="Defender")
    ax.scatter(env.goal[0], env.goal[1], c='g', label="Goal")

    # 静态障碍：修改为循环遍历所有静态障碍物
    th = np.linspace(0, np.pi*2, 200)
    for i in range(env._num_static_obs):
        cx, cy = env.static_centers[i]
        r0 = env.static_radii[i]
        ax.plot(cx + np.cos(th)*r0, cy + np.sin(th)*r0, 'r--')

    # 终态捕捉半径
    cx2, cy2 = traj_d[-1,:2]
    Rd = env.capture_radius
    ax.plot(cx2 + np.cos(th)*Rd, cy2 + np.sin(th)*Rd, 'b--')

    ax.set_aspect('equal')
    ax.grid()
    ax.legend()
    fig.savefig(prefix+"_traj.png", dpi=300)

    # ---- 控制输入 ----
    plt.figure()
    plt.plot(time, omega)
    plt.xlabel("t")
    plt.ylabel("omega (QP)")
    plt.grid()
    plt.savefig(prefix+"_omega.png", dpi=300)

    # ---- CBF 几何 h ----
    plt.figure()
    plt.plot(time, h_geom)
    plt.axhline(0, linestyle="--")
    plt.ylabel("min h_geom")
    plt.grid()
    plt.savefig(prefix+"_hgeom.png", dpi=300)

    # ---- reward ----
    plt.figure()
    plt.plot(time, reward)
    plt.ylabel("reward")
    plt.grid()
    plt.savefig(prefix+"_reward.png", dpi=300)

    # ---- PenaltyNet ----
    plt.figure()
    for i in range(k1.shape[1]):
        plt.plot(time, k1[:,i], label=f"k1_{i}")
    plt.grid()
    plt.legend()
    plt.savefig(prefix+"_k1.png", dpi=300)

    plt.figure()
    for i in range(k2.shape[1]):
        plt.plot(time, k2[:,i], label=f"k2_{i}")
    plt.grid()
    plt.legend()
    plt.savefig(prefix+"_k2.png", dpi=300)

    plt.show()


# ======================================================
# Robotarium 动画播放
# ======================================================
def robotarium_playback(env, R):
    """
    Playback attacker & defender trajectories in Robotarium simulator.
    Includes:
    - static obstacle (+label)
    - defender capture radius circle (dynamic)
    - goal (+label)
    - attacker/defender labels
    - trajectory visualization
    """

    traj_a = R["traj_attacker"]
    traj_d = R["traj_defender"]
    time = R["time"]
    T = len(time)

    # ========= Robotarium init =========
    N = 2  # attacker + defender
    initial_conditions = np.array([
        [traj_a[0, 0], traj_d[0, 0]],
        [traj_a[0, 1], traj_d[0, 1]],
        [traj_a[0, 2], traj_d[0, 2]],
    ])

    print(f"[Robotarium] Initial yaw A={traj_a[0,2]:.2f}, D={traj_d[0,2]:.2f}")

    r = robotarium.Robotarium(
        number_of_robots=N,
        show_figure=True,
        initial_conditions=initial_conditions,
        sim_in_real_time=True
    )

    fig = r.figure
    ax = fig.axes[0]
    ax.set_aspect('equal')

    # Set default ranges if environment does not contain them
    if not hasattr(env, "x_range"):
        env.x_range = [-1.6, 1.6]
    if not hasattr(env, "y_range"):
        env.y_range = [-1.0, 1.0]

    ax.set_xlim(env.x_range)
    ax.set_ylim(env.y_range)
    ax.set_title("Robotarium Playback (HOCBF-QP Safety Control)", fontsize=12)

    # ========= Static obstacle visualization =========
    # 修改为循环遍历所有静态障碍物
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

    # ========= Goal region =========
    goal_center = env.goal
    goal_radius = env.goal_tol

    goal_patch = Circle(
        goal_center, goal_radius,
        edgecolor='green', facecolor='none',
        linestyle='--', linewidth=1.5, zorder=2
    )
    ax.add_patch(goal_patch)

    ax.text(
        goal_center[0], goal_center[1] + goal_radius + 0.05,
        "Goal", color="green",
        fontsize=10, ha="center", zorder=10
    )

    # ========= Defender capture radius =========
    capture_R = env.capture_radius
    capture_patch = Circle(
        (traj_d[0,0], traj_d[0,1]), capture_R,
        edgecolor='blue',
        facecolor='none',
        linestyle='--', linewidth=1.5, zorder=2
    )
    ax.add_patch(capture_patch)


    # ========= Labels for robots =========
    attacker_label = ax.text(
        traj_a[0,0], traj_a[0,1] - 0.10,
        "Attacker", color="red",
        fontsize=10, ha="center", zorder=12
    )

    defender_label = ax.text(
        traj_d[0,0], traj_d[0,1] + 0.10,
        "Defender", color="blue",
        fontsize=10, ha="center", zorder=12
    )

    # ========= Trajectory lines =========
    line_attacker, = ax.plot([], [], 'r-', linewidth=1.5, label="Attacker Trajectory")
    line_defender, = ax.plot([], [], 'b-', linewidth=1.5, label="Defender Trajectory")

    ax.legend(loc="upper left")

    # Robotarium unicycle controller
    uni_pos_ctrl = create_clf_unicycle_position_controller()

    # ========= Playback loop =========
    print("[Robotarium] Starting trajectory playback...")

    for k in range(T):
        x = r.get_poses()   # shape 3xN

        # Desired positions (attacker, defender)
        goals = np.array([
            [traj_a[k,0], traj_d[k,0]],
            [traj_a[k,1], traj_d[k,1]],
        ])

        dxu = uni_pos_ctrl(x, goals)
        r.set_velocities(np.arange(N), dxu)

        # --- Update capture circle center ---
        capture_patch.center = (x[0,1], x[1,1])

        # --- Update labels (slightly offset from robot centers) ---
        attacker_label.set_position((x[0,0], x[1,0] - 0.10))
        defender_label.set_position((x[0,1], x[1,1] + 0.10))

        # --- Update trajectories ---
        line_attacker.set_data(traj_a[:k+1, 0], traj_a[:k+1, 1])
        line_defender.set_data(traj_d[:k+1, 0], traj_d[:k+1, 1])

        # Refresh Robotarium GUI
        r.step()

    r.call_at_scripts_end()
    print("[Robotarium] Playback finished.")



# ======================================================
# 主测试函数
# ======================================================
def test_sac_hocbfqp_robotarium():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = UnicycleHOCBFEnvRobotarium()
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    num_obs = env.num_obs
    act_limit = 2.0

    # ---- load models ----
    actor = ActorGaussian(obs_dim, act_dim, act_limit).to(device)
    pn = GaussianPenaltyNet(obs_dim, num_obs).to(device)

    actor.load_state_dict(torch.load("actor_sac_hocbf_att_def_v2.pth", map_location=device))
    pn.load_state_dict(torch.load("penaltynet_sac_hocbf_att_def_v2.pth", map_location=device))

    actor.eval()
    pn.eval()

    # ---- rollout ----
    R = rollout_episode(env, actor, pn, device)

    # ---- plot ----
    plot_results(env, R, prefix="hocbfqp_test")

    # ---- Robotarium 动画 ----
    robotarium_playback(env, R)


if __name__ == "__main__":
    test_sac_hocbfqp_robotarium()