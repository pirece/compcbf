# compcbf_inputcons_multiple_obs_unicycle_rl_test_numpy.py

import os
import numpy as np
from matplotlib.patches import Circle
import scipy.io as sio

# 导入新的环境 v2
from compcbf_env_v2 import UnicycleHOCBFEnvRobotarium as UnicycleHOCBFEnv

# ==== Robotarium 原生可视化与底层控制相关模块 ====
import rps.robotarium as robotarium
from rps.utilities.controllers import create_clf_unicycle_position_controller, create_clf_unicycle_pose_controller
from rps.utilities.misc import at_pose
from rps.utilities.barrier_certificates import create_unicycle_barrier_certificate_with_boundary

# 从外部独立生成的参数文件中导入神经网络权重
try:
    from weights_dict_v2 import HARDCODED_WEIGHTS
except ImportError:
    raise ImportError("无法导入 weights_dict 模块，请先运行 convert_pth_to_py.py 生成 weights_dict.py。")


# =====================================================================
# 1. 纯 NumPy 实现的神经网络前向推理
# =====================================================================

class NumPyActor:
    """基于纯 NumPy 矩阵运算的 Actor 策略网络前向推理类"""
    def __init__(self, weights_dict, act_limit=2.0):
        self.W1 = weights_dict['actor_W1']
        self.b1 = weights_dict['actor_b1']
        self.W2 = weights_dict['actor_W2']
        self.b2 = weights_dict['actor_b2']
        self.W_mu = weights_dict['actor_W_mu']
        self.b_mu = weights_dict['actor_b_mu']
        self.act_limit = act_limit

    def __call__(self, obs):
        h1 = np.maximum(0, obs @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        mu = h2 @ self.W_mu + self.b_mu
        u_nom = self.act_limit * np.tanh(mu)
        return u_nom.item()

class NumPyPenaltyNet:
    """基于纯 NumPy 矩阵运算的惩罚网络前向推理类"""
    def __init__(self, weights_dict, num_obs):
        self.num_obs = num_obs
        self.W1 = weights_dict['penalty_W1']
        self.b1 = weights_dict['penalty_b1']
        self.W2 = weights_dict['penalty_W2']
        self.b2 = weights_dict['penalty_b2']
        self.W_mu = weights_dict['penalty_W_mu']
        self.b_mu = weights_dict['penalty_b_mu']

    def __call__(self, obs):
        h1 = np.tanh(obs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        mu_p = h2 @ self.W_mu + self.b_mu
        
        bx = 0.5 * mu_p
        softplus_out = np.where(bx > 20, mu_p, 2.0 * np.log1p(np.exp(bx)))
        params = softplus_out + 1e-3
        
        # 拆分输出
        k1_vec = params[:self.num_obs]
        k2_vec = params[self.num_obs:2 * self.num_obs]
        alpha_c = params[-1]
        
        return k1_vec, k2_vec, alpha_c

# =====================================================================
# 2. 离线仿真 Rollout 模块
# =====================================================================

def rollout_one_episode(env, actor, penalty_net, max_steps=5000):
    obs = env.reset().astype(np.float64)

    traj_attacker = [obs[:3].copy()]
    traj_defender = [obs[4:].copy()]
    omega_hist, omega_hat_hist, u_nom_hist = [], [], []
    H_hist, min_dist_hist, min_safe_hist = [], [], []
    h_geom_min_hist, psi2_min_hist = [], []
    phi_min_hist, phi_max_hist = [], []
    reward_hist = []
    k1_hist, k2_hist, alpha_hist, relax_hist = [], [], [], []

    for t in range(max_steps):
        u_nom = actor(obs)
        k1_vec, k2_vec, alpha_c = penalty_net(obs)

        next_obs, r, done, info = env.step(u_nom, k1_vec, k2_vec, float(alpha_c))
        next_obs = next_obs.astype(np.float64)

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

        if done:
            break

    time_arr = np.arange(len(omega_hist)) * env.dt
    
    return dict(
        traj_attacker=np.array(traj_attacker),
        traj_defender=np.array(traj_defender),
        time=time_arr,
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
    )

# =====================================================================
# 3. 数据保存接口
# =====================================================================

def save_data_to_mat(R, filename="experiment_results_compcbf.mat"):
    print(f"正在保存实验数据至 {filename} ...")
    sio.savemat(filename, R)
    print("数据保存成功。")

# =====================================================================
# 4. Robotarium 物理系统轨迹回放与部署验证
# =====================================================================

def robotarium_playback(env, R, align_to_initial_pose=True):
    traj_a = R["traj_attacker"]
    traj_d = R["traj_defender"]
    time_arr = R["time"]
    N = 2  

    if align_to_initial_pose:
        r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)
    else:
        initial_conditions = np.array([
            [traj_a[0, 0], traj_d[0, 0]],
            [traj_a[0, 1], traj_d[0, 1]],
            [traj_a[0, 2], traj_d[0, 2]],
        ])
        r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

    ax = r.figure.axes[0]
    ax.set_aspect('equal')

    if not hasattr(env, "x_range"):
        env.x_range = [-1.6, 1.6]
    if not hasattr(env, "y_range"):
        env.y_range = [-1.0, 1.0]

    ax.set_xlim(env.x_range)
    ax.set_ylim(env.y_range)
    ax.set_title("Robotarium Playback (NumPy Proposed CompCBF)", fontsize=12)

    # ===== 绘制 UI 元素与障碍物模型 =====
    # 循环绘制 3 个静态障碍物
    for i in range(env._num_static_obs):
        static_center = env.static_centers[i]
        static_radius = env.static_radii[i]

        static_obs_patch = Circle(static_center, static_radius, color='gray', alpha=0.3, zorder=1)
        ax.add_patch(static_obs_patch)
        ax.text(static_center[0], static_center[1], f"Obs {i+1}", color="black", fontsize=9, ha="center", va="center", zorder=10)


    # 目标区域绘图
    target_patch = Circle(env.goal, env.goal_tol, edgecolor='green', facecolor='none', linestyle='--', linewidth=1.5, zorder=2)
    ax.add_patch(target_patch)
    ax.text(env.goal[0], env.goal[1], "Target", fontsize=10, color="green", ha="center", zorder=10)

    # 动态防守者捕捉半径绘图
    capture_patch = Circle((0.0, 0.0), env.capture_radius, edgecolor='blue', facecolor='none', linestyle='--', linewidth=1.5, zorder=2)
    ax.add_patch(capture_patch)

    # 实体标识说明
    attacker_label = ax.text(0.0, 0.0, "Attacker", color="red", fontsize=10, ha="center", zorder=10)
    defender_label = ax.text(0.0, 0.0, "Defender", color="blue", fontsize=10, ha="center", zorder=10)

    # 实例化位姿控制器和避障证书
    uni_pose_ctrl = create_clf_unicycle_pose_controller()      
    uni_pos_ctrl = create_clf_unicycle_position_controller()   
    uni_barrier_cert = create_unicycle_barrier_certificate_with_boundary(safety_radius=0.14)

    x = r.get_poses()
    r.step()

    # 阶段 1：预对齐 (Pre-alignment Phase)
    if align_to_initial_pose:
        print("开始预对齐阶段：引导机器人前往设定的初始位置及航向...")
        desired_init = np.array([
            [traj_a[0, 0], traj_d[0, 0]],
            [traj_a[0, 1], traj_d[0, 1]],
            [traj_a[0, 2], traj_d[0, 2]]
        ])

        while np.size(at_pose(x, desired_init, position_error=0.08, rotation_error=0.3)) != N:
            x = r.get_poses()
            dxu = uni_pose_ctrl(x, desired_init)
            dxu = uni_barrier_cert(dxu, x)
            
            dxu[0, :] = np.clip(dxu[0, :], -0.1, 0.1)   
            dxu[1, :] = np.clip(dxu[1, :], -1, 1)   
            
            r.set_velocities(np.arange(N), dxu)

            capture_patch.center = (x[0, 1], x[1, 1])
            attacker_label.set_position((x[0, 0], x[1, 0] - 0.08))
            defender_label.set_position((x[0, 1], x[1, 1] + 0.08))

            r.step()
            
        print("预对齐完成，开始部署轨迹回放...")

    # 阶段 2：轨迹回放
    T = len(time_arr)
    for k in range(T):
        x = r.get_poses()

        goals = np.array([
            [traj_a[k, 0], traj_d[k, 0]],
            [traj_a[k, 1], traj_d[k, 1]],
        ])

        dxu = uni_pos_ctrl(x, goals)
        dxu[0, :] = np.clip(dxu[0, :], -0.15, 0.15)
        dxu[1, :] = np.clip(dxu[1, :], -2, 2)
        
        r.set_velocities(np.arange(N), dxu)

        capture_patch.center = (x[0, 1], x[1, 1])
        attacker_label.set_position((x[0, 0], x[1, 0] - 0.08))
        defender_label.set_position((x[0, 1], x[1, 1] + 0.08))

        r.step()

    r.call_at_scripts_end()
    print("Robotarium 轨迹部署验证任务执行完毕。")


if __name__ == "__main__":
    print("正在初始化单车运动学交互环境...")
    env = UnicycleHOCBFEnv(T_max=50.0, safe_margin=0.0)
    
    print("正在加载基于纯 NumPy 的策略网络权重参数...")
    actor_np = NumPyActor(HARDCODED_WEIGHTS, act_limit=2.0)
    penalty_np = NumPyPenaltyNet(HARDCODED_WEIGHTS, num_obs=env.num_obs)
    
    print("开始执行纯 NumPy 离线轨迹生成运算...")
    R = rollout_one_episode(env, actor_np, penalty_np, max_steps=5000)
    print(f"离线运算完成，总计执行步数: {len(R['time'])}。")
    
    mat_filename = "experiment_results_compcbf.mat"
    save_data_to_mat(R, filename=mat_filename)
    
    print("准备将轨迹下发至 Robotarium 系统开展实际控制验证...")
    robotarium_playback(env, R, align_to_initial_pose=True)