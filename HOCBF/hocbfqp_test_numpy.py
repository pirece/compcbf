# hocbfqp_test_numpy.py

import os
import numpy as np
from matplotlib.patches import Circle
import scipy.io as sio

# 导入对比方法的 Robotarium 交互环境 (最新版 v2，含 3 个静态障碍物)
from hocbfqp_env_v2 import UnicycleHOCBFEnvRobotarium as UnicycleHOCBFEnv

# ==== Robotarium 原生可视化与底层控制相关模块 ====
import rps.robotarium as robotarium
from rps.utilities.controllers import create_clf_unicycle_position_controller, create_clf_unicycle_pose_controller
from rps.utilities.misc import at_pose
from rps.utilities.barrier_certificates import create_unicycle_barrier_certificate_with_boundary

# 从外部独立生成的参数文件中导入神经网络权重
try:
    from weights_dict_hocbf_v2 import HARDCODED_WEIGHTS
except ImportError:
    raise ImportError("无法导入 weights_dict_hocbf_v2 模块，请确保文件存在于同一目录下。")


# =====================================================================
# 1. 纯 NumPy 实现的神经网络前向推理
# =====================================================================

class NumPyActor:
    """基于纯 NumPy 矩阵运算的 Actor 策略网络前向推理类"""
    def __init__(self, weights_dict, act_limit=2.0):
        # 提取各个全连接层的权重矩阵与偏置向量（注意：PyTorch到NumPy的线性层权重需要转置）
        self.W1 = weights_dict['actor_net_0_weight'].T
        self.b1 = weights_dict['actor_net_0_bias']
        self.W2 = weights_dict['actor_net_2_weight'].T
        self.b2 = weights_dict['actor_net_2_bias']
        self.W_mu = weights_dict['actor_mu_layer_weight'].T
        self.b_mu = weights_dict['actor_mu_layer_bias']
        self.act_limit = act_limit

    def __call__(self, obs):
        # 第一层网络前向传播与 ReLU 激活
        h1 = np.maximum(0, obs @ self.W1 + self.b1)
        # 第二层网络前向传播与 ReLU 激活
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        # 输出层前向传播
        mu = h2 @ self.W_mu + self.b_mu
        # Tanh 激活并根据设定的动作限制进行幅值放缩
        u_nom = self.act_limit * np.tanh(mu)
        return u_nom.item()


class NumPyPenaltyNet:
    """基于纯 NumPy 矩阵运算的惩罚网络前向推理类 (针对标准 HOCBF)"""
    def __init__(self, weights_dict, num_obs):
        self.num_obs = num_obs
        # 提取网络层的权重矩阵与偏置向量
        self.W1 = weights_dict['penalty_backbone_0_weight'].T
        self.b1 = weights_dict['penalty_backbone_0_bias']
        self.W2 = weights_dict['penalty_backbone_2_weight'].T
        self.b2 = weights_dict['penalty_backbone_2_bias']
        self.W_mu = weights_dict['penalty_mu_layer_weight'].T
        self.b_mu = weights_dict['penalty_mu_layer_bias']

    def __call__(self, obs):
        # 隐藏层前向传播，使用 Tanh 作为激活函数
        h1 = np.tanh(obs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        # 输出层前向传播
        mu_p = h2 @ self.W_mu + self.b_mu
        
        # 采用 Softplus 函数保证输出变量均为正值
        # 为了提高数值计算的稳定性，对输入值进行截断处理
        bx = 0.5 * mu_p
        softplus_out = np.where(bx > 20, mu_p, 2.0 * np.log1p(np.exp(bx)))
        # 附加偏置值，避免除零或无穷小问题
        params = softplus_out + 1e-3
        
        # 拆分输出维度：对比方法标准 HOCBF 仅需要 k1 和 k2，不再需要 alpha_c
        k1_vec = params[:self.num_obs]
        k2_vec = params[self.num_obs:2 * self.num_obs]
        
        return k1_vec, k2_vec

# =====================================================================
# 2. 离线仿真 Rollout 模块 (在虚拟环境中验证并收集轨迹数据)
# =====================================================================

def rollout_one_episode(env, actor, penalty_net, max_steps=5000):
    """单次仿真的轨迹生成函数，记录系统状态与各项评价指标的数据"""
    obs = env.reset().astype(np.float64)

    # 初始化状态记录列表
    traj_attacker = [obs[:3].copy()]
    traj_defender = [obs[4:].copy()]
    omega_hist, omega_hat_hist, u_nom_hist = [], [], []
    H_hist, min_dist_hist, min_safe_hist = [], [], []
    h_geom_min_hist, psi2_min_hist = [], []
    phi_min_hist, phi_max_hist = [], []
    reward_hist = []
    k1_hist, k2_hist, relax_hist = [], [], []
    r_dist_hist, r_heading_hist, r_time_hist, r_near_goal_hist = [], [], [], []

    # 按最大步数限制执行控制迭代
    for t in range(max_steps):
        # 策略网络输出控制动作
        u_nom = actor(obs)
        # 惩罚网络输出参数 (修正点：只接收 k1 和 k2)
        k1_vec, k2_vec = penalty_net(obs)

        # 步进环境并获取下一状态及环境反馈信息 (修正点：只传入 3 个参数)
        next_obs, r, done, info = env.step(u_nom, k1_vec, k2_vec)
        next_obs = next_obs.astype(np.float64)

        # 追加数据至相应的记录列表
        traj_attacker.append(next_obs[:3].copy())
        traj_defender.append(next_obs[4:].copy())
        obs = next_obs

        # 使用 info.get() 确保字典中没有该键时不会报错，默认返回 0.0
        omega_hist.append(info.get("omega", 0.0))
        omega_hat_hist.append(info.get("omega_hat", 0.0))
        u_nom_hist.append(info.get("u_nom", 0.0))
        H_hist.append(info.get("H", 0.0))
        min_dist_hist.append(info.get("min_dist", 0.0))
        min_safe_hist.append(info.get("min_safe_radius", 0.0))
        h_geom_min_hist.append(info.get("h_geom_min", 0.0))
        psi2_min_hist.append(info.get("psi2_min", 0.0))
        phi_min_hist.append(info.get("phi_min", 0.0))
        phi_max_hist.append(info.get("phi_max", 0.0))
        reward_hist.append(info.get("reward", 0.0))

        # 记录 HOCBF 参数
        k1_hist.append(info.get("k1", np.zeros_like(k1_vec)))
        k2_hist.append(info.get("k2", np.zeros_like(k2_vec)))
        # 记录松弛量
        relax_hist.append(abs(info.get("omega_hat", 0.0) - info.get("u_nom", 0.0)))

        # 记录奖励函数分解
        r_dist_hist.append(info.get("r_dist", 0.0))
        r_heading_hist.append(info.get("r_heading", 0.0))
        r_time_hist.append(info.get("r_time", 0.0))
        r_near_goal_hist.append(info.get("r_near_goal", 0.0))

        if done:
            break

    # 计算全局时间序列向量
    time_arr = np.arange(len(omega_hist)) * env.dt
    
    # 将记录数据打包为字典格式返回，剔除了对比方法中不存在的 alpha_c
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
        relax=np.array(relax_hist),
        r_dist=np.array(r_dist_hist),
        r_heading=np.array(r_heading_hist),
        r_time=np.array(r_time_hist),
        r_near_goal=np.array(r_near_goal_hist),
    )

# =====================================================================
# 3. 数据保存接口
# =====================================================================

def save_data_to_mat(R, filename="experiment_results_hocbf.mat"):
    """将采集的数据集保存至 .mat 格式文件中，便于论文结果绘制"""
    print(f"正在保存实验数据至 {filename} ...")
    sio.savemat(filename, R)
    print("数据保存成功。")

# =====================================================================
# 4. Robotarium 物理系统轨迹回放与部署验证
# =====================================================================

def robotarium_playback(env, R, align_to_initial_pose=True):
    """基于离线计算出的轨迹序列在 Robotarium 物理平台上执行跟踪与回放"""
    traj_a = R["traj_attacker"]
    traj_d = R["traj_defender"]
    time_arr = R["time"]

    # 定义实验中使用的机器人总数 (一个攻击者，一个防守者)
    N = 2  

    # 初始化 Robotarium 对象，设定初始条件
    if align_to_initial_pose:
        r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)
    else:
        initial_conditions = np.array([
            [traj_a[0, 0], traj_d[0, 0]],
            [traj_a[0, 1], traj_d[0, 1]],
            [traj_a[0, 2], traj_d[0, 2]],
        ])
        r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

    # 配置 matplotlib 可视化界面的相关属性
    ax = r.figure.axes[0]
    ax.set_aspect('equal')

    if not hasattr(env, "x_range"):
        env.x_range = [-1.6, 1.6]
    if not hasattr(env, "y_range"):
        env.y_range = [-1.0, 1.0]

    ax.set_xlim(env.x_range)
    ax.set_ylim(env.y_range)
    ax.set_title("Robotarium Playback (Baseline HOCBF-QP)", fontsize=12)

    # ===== 绘制 UI 元素与障碍物模型 =====
    # 修复点：循环绘制 3 个静态障碍物
    for i in range(env._num_static_obs):
        static_center = env.static_centers[i]
        static_radius = env.static_radii[i]

        static_obs_patch = Circle(static_center, static_radius, color='gray', alpha=0.3, zorder=1)
        ax.add_patch(static_obs_patch)
        ax.text(static_center[0], static_center[1], f"Obs {i+1}", color="black", fontsize=9, ha="center", va="center", zorder=10)


    target_patch = Circle(env.goal, env.goal_tol, edgecolor='green', facecolor='none', linestyle='--', linewidth=1.5, zorder=2)
    ax.add_patch(target_patch)
    ax.text(env.goal[0], env.goal[1], "Target", fontsize=10, color="green", ha="center", zorder=10)

    capture_patch = Circle((0.0, 0.0), env.capture_radius, edgecolor='blue', facecolor='none', linestyle='--', linewidth=1.5, zorder=2)
    ax.add_patch(capture_patch)

    attacker_label = ax.text(0.0, 0.0, "Attacker", color="red", fontsize=10, ha="center", zorder=10)
    defender_label = ax.text(0.0, 0.0, "Defender", color="blue", fontsize=10, ha="center", zorder=10)

    # 实例化 Robotarium 底层位姿控制器和避障安全证书
    uni_pose_ctrl = create_clf_unicycle_pose_controller()      
    uni_pos_ctrl = create_clf_unicycle_position_controller()   
    uni_barrier_cert = create_unicycle_barrier_certificate_with_boundary(safety_radius=0.14)

    # 获取初始位姿状态并执行第一步更新
    x = r.get_poses()
    r.step()

    # 阶段 1：预对齐 (Pre-alignment Phase)
    # 将随机位置的机器人引导至轨迹序列指定的起始坐标和航向角
    if align_to_initial_pose:
        print("开始预对齐阶段：引导机器人前往设定的初始位置及航向...")
        desired_init = np.array([
            [traj_a[0, 0], traj_d[0, 0]],
            [traj_a[0, 1], traj_d[0, 1]],
            [traj_a[0, 2], traj_d[0, 2]]
        ])

        # 容差设置：位置误差小于 0.08m，航向误差小于 0.3rad
        while np.size(at_pose(x, desired_init, position_error=0.08, rotation_error=0.3)) != N:
            x = r.get_poses()
            dxu = uni_pose_ctrl(x, desired_init)
            # 经过底层 CBF 进行系统内部的安全修正
            dxu = uni_barrier_cert(dxu, x)
            
            # 对输出速度执行限幅以避免出现硬件报警信息
            dxu[0, :] = np.clip(dxu[0, :], -0.1, 0.1)   # 限制线速度在 [-0.1, 0.1]
            dxu[1, :] = np.clip(dxu[1, :], -1, 1)       # 限制角速度在 [-1, 1]
            
            r.set_velocities(np.arange(N), dxu)

            # 动态更新界面 UI 组件位置
            capture_patch.center = (x[0, 1], x[1, 1])
            attacker_label.set_position((x[0, 0], x[1, 0] - 0.08))
            defender_label.set_position((x[0, 1], x[1, 1] + 0.08))

            r.step()
            
        print("预对齐完成，开始部署轨迹回放...")

    # 阶段 2：轨迹回放 (Trajectory Playback Phase)
    # 控制机器人在每个时间步追寻已经规划完成的离线无碰撞轨迹
    T = len(time_arr)
    for k in range(T):
        x = r.get_poses()

        goals = np.array([
            [traj_a[k, 0], traj_d[k, 0]],
            [traj_a[k, 1], traj_d[k, 1]],
        ])

        dxu = uni_pos_ctrl(x, goals)
        
        # 施加硬件层级的安全限幅，保护电机
        dxu[0, :] = np.clip(dxu[0, :], -0.15, 0.15)
        dxu[1, :] = np.clip(dxu[1, :], -2, 2)
        
        r.set_velocities(np.arange(N), dxu)

        # 动态更新界面 UI 组件位置
        capture_patch.center = (x[0, 1], x[1, 1])
        attacker_label.set_position((x[0, 0], x[1, 0] - 0.08))
        defender_label.set_position((x[0, 1], x[1, 1] + 0.08))

        r.step()

    # 安全终止系统执行过程
    r.call_at_scripts_end()
    print("Robotarium 轨迹部署验证任务执行完毕。")


if __name__ == "__main__":
    print("正在初始化对比方法环境...")
    env = UnicycleHOCBFEnv(T_max=50.0, safe_margin=0.0)
    
    print("正在加载基于纯 NumPy 的策略网络权重参数...")
    actor_np = NumPyActor(HARDCODED_WEIGHTS, act_limit=2.0)
    penalty_np = NumPyPenaltyNet(HARDCODED_WEIGHTS, num_obs=env.num_obs)
    
    print("开始执行纯 NumPy 离线轨迹生成运算...")
    # 这里将自动按照对比方法的 step 定义执行
    R = rollout_one_episode(env, actor_np, penalty_np, max_steps=5000)
    print(f"离线运算完成，总计执行步数: {len(R['time'])}。")
    
    mat_filename = "experiment_results_hocbf.mat"
    save_data_to_mat(R, filename=mat_filename)
    
    print("准备将轨迹下发至 Robotarium 系统开展实际控制验证...")
    # 对齐初始位置后启动
    robotarium_playback(env, R, align_to_initial_pose=True)