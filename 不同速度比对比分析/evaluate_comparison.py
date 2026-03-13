import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. 导入你的最新环境 v2
# =====================================================================
from hocbfqp_env_v2 import UnicycleHOCBFEnvRobotarium as EnvHOCBF
from compcbf_env_v2 import UnicycleHOCBFEnvRobotarium as EnvCompCBF

# =====================================================================
# 2. 尝试导入你的权重文件 (如果找不到则使用随机权重以防报错)
# =====================================================================
try:
    from weights_dict_hocbf_v2 import HARDCODED_WEIGHTS as WEIGHTS_HOCBF
    from weights_dict_v2 import HARDCODED_WEIGHTS as WEIGHTS_COMPCBF
    WEIGHTS_LOADED = True
    print("成功加载 HOCBF 和 CompCBF 的外部权重字典。")
except ImportError:
    WEIGHTS_LOADED = False
    print("警告: 无法找到 weights_dict_hocbf_v2.py 或 weights_dict_v2.py！")
    print("将使用随机初始化的 NumPy 权重进行演示...")
    # 自动生成演示用的 Dummy 权重，保证代码能跑通并生成表格
    WEIGHTS_HOCBF = {
        'actor_net_0_weight': np.random.randn(128, 8), 'actor_net_0_bias': np.zeros(128),
        'actor_net_2_weight': np.random.randn(128, 128), 'actor_net_2_bias': np.zeros(128),
        'actor_mu_layer_weight': np.random.randn(1, 128), 'actor_mu_layer_bias': np.zeros(1),
        'penalty_backbone_0_weight': np.random.randn(64, 8), 'penalty_backbone_0_bias': np.zeros(64),
        'penalty_backbone_2_weight': np.random.randn(64, 64), 'penalty_backbone_2_bias': np.zeros(64),
        'penalty_mu_layer_weight': np.random.randn(8, 64), 'penalty_mu_layer_bias': np.zeros(8) # 2 * num_obs (4) = 8
    }
    WEIGHTS_COMPCBF = {
        'actor_W1': np.random.randn(8, 128), 'actor_b1': np.zeros(128),
        'actor_W2': np.random.randn(128, 128), 'actor_b2': np.zeros(128),
        'actor_W_mu': np.random.randn(128, 1), 'actor_b_mu': np.zeros(1),
        'penalty_W1': np.random.randn(8, 64), 'penalty_b1': np.zeros(64),
        'penalty_W2': np.random.randn(64, 64), 'penalty_b2': np.zeros(64),
        'penalty_W_mu': np.random.randn(64, 9), 'penalty_b_mu': np.zeros(9) # 2 * 4 + 1 = 9
    }


# =====================================================================
# 3. 纯 NumPy 网络定义 (提取自你的测试文件)
# =====================================================================
class NumPyActorHOCBF:
    def __init__(self, weights_dict, act_limit=2.0):
        self.W1 = weights_dict['actor_net_0_weight'].T
        self.b1 = weights_dict['actor_net_0_bias']
        self.W2 = weights_dict['actor_net_2_weight'].T
        self.b2 = weights_dict['actor_net_2_bias']
        self.W_mu = weights_dict['actor_mu_layer_weight'].T
        self.b_mu = weights_dict['actor_mu_layer_bias']
        self.act_limit = act_limit

    def __call__(self, obs):
        h1 = np.maximum(0, obs @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        mu = h2 @ self.W_mu + self.b_mu
        return (self.act_limit * np.tanh(mu)).item()

class NumPyPenaltyNetHOCBF:
    def __init__(self, weights_dict, num_obs):
        self.num_obs = num_obs
        self.W1 = weights_dict['penalty_backbone_0_weight'].T
        self.b1 = weights_dict['penalty_backbone_0_bias']
        self.W2 = weights_dict['penalty_backbone_2_weight'].T
        self.b2 = weights_dict['penalty_backbone_2_bias']
        self.W_mu = weights_dict['penalty_mu_layer_weight'].T
        self.b_mu = weights_dict['penalty_mu_layer_bias']

    def __call__(self, obs):
        h1 = np.tanh(obs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        mu_p = h2 @ self.W_mu + self.b_mu
        bx = 0.5 * mu_p
        params = np.where(bx > 20, mu_p, 2.0 * np.log1p(np.exp(bx))) + 1e-3
        return params[:self.num_obs], params[self.num_obs:2 * self.num_obs]

class NumPyActorCompCBF:
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
        return (self.act_limit * np.tanh(mu)).item()

class NumPyPenaltyNetCompCBF:
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
        params = np.where(bx > 20, mu_p, 2.0 * np.log1p(np.exp(bx))) + 1e-3
        return params[:self.num_obs], params[self.num_obs:2 * self.num_obs], params[-1]


# =====================================================================
# 4. 单回合 Rollout 函数 (增加初始位置的随机扰动)
# =====================================================================
def run_episode_hocbf(env, actor, penalty_net):
    obs = env.reset().astype(np.float64)
    # 添加 [-0.05, 0.05] 的随机扰动以测试鲁棒性并产生数据方差
    env.state_a[:2] += np.random.uniform(-0.05, 0.05, size=2)
    env.state_d[:2] += np.random.uniform(-0.05, 0.05, size=2)
    obs = env._get_obs().astype(np.float64)
    
    ep_reward = 0.0
    ep_infeasible_cnt = 0
    
    for _ in range(env.max_steps):
        u_nom = actor(obs)
        k1_vec, k2_vec = penalty_net(obs)
        next_obs, reward, done, info = env.step(u_nom, k1_vec, k2_vec)
        
        ep_reward += reward
        ep_infeasible_cnt += info.get("infeasible_cnt", 0)
        obs = next_obs.astype(np.float64)
        if done: break
            
    return ep_reward, ep_infeasible_cnt

def run_episode_compcbf(env, actor, penalty_net):
    obs = env.reset().astype(np.float64)
    # 相同幅度的随机扰动
    env.state_a[:2] += np.random.uniform(-0.05, 0.05, size=2)
    env.state_d[:2] += np.random.uniform(-0.05, 0.05, size=2)
    obs = env._get_obs().astype(np.float64)
    
    ep_reward = 0.0
    ep_infeasible_cnt = 0  # 提出方法是解析解，恒为0
    
    for _ in range(env.max_steps):
        u_nom = actor(obs)
        k1_vec, k2_vec, alpha_c = penalty_net(obs)
        next_obs, reward, done, info = env.step(u_nom, k1_vec, k2_vec, float(alpha_c))
        
        ep_reward += reward
        obs = next_obs.astype(np.float64)
        if done: break
            
    return ep_reward, ep_infeasible_cnt


# =====================================================================
# 5. 主评估与画表流程
# =====================================================================
def evaluate_and_compare():
    speed_ratios = [0.4, 0.6, 0.8]
    num_episodes = 50  # 每种设定测试 50 次以获取统计学均值和标准差
    
    table_data = []

    print(f"\n{'='*65}")
    print(f" 开始评估 (测试回合/配置: {num_episodes}) - 动态包含 4 个障碍物")
    print(f"{'='*65}")
    
    for ratio in speed_ratios:
        print(f"\n[ 测试中 ] 防守者/攻击者 速度比: {ratio}")
        
        # ------------------ Baseline HOCBF-QP ------------------
        env_hocbf = EnvHOCBF(defender_speed_ratio=ratio)
        actor_h = NumPyActorHOCBF(WEIGHTS_HOCBF)
        pn_h = NumPyPenaltyNetHOCBF(WEIGHTS_HOCBF, env_hocbf.num_obs)
        
        r_list_h, inf_list_h = [], []
        for _ in range(num_episodes):
            r, inf = run_episode_hocbf(env_hocbf, actor_h, pn_h)
            r_list_h.append(r)
            inf_list_h.append(inf)
            
        r_mean_h, r_std_h = np.mean(r_list_h), np.std(r_list_h)
        inf_mean_h = np.mean(inf_list_h)
        print(f"  > HOCBF-QP   | 奖励: {r_mean_h:6.2f} ± {r_std_h:5.2f} | 平均 QP 无解次数: {inf_mean_h:.1f}")
        table_data.append(["Baseline HOCBF-QP", f"{ratio}", f"{r_mean_h:.2f} ± {r_std_h:.2f}", f"{inf_mean_h:.1f}"])
        
        # ------------------ Proposed CompCBF -------------------
        env_comp = EnvCompCBF(defender_speed_ratio=ratio)
        actor_c = NumPyActorCompCBF(WEIGHTS_COMPCBF)
        pn_c = NumPyPenaltyNetCompCBF(WEIGHTS_COMPCBF, env_comp.num_obs)
        
        r_list_c, inf_list_c = [], []
        for _ in range(num_episodes):
            r, inf = run_episode_compcbf(env_comp, actor_c, pn_c)
            r_list_c.append(r)
            inf_list_c.append(inf)
            
        r_mean_c, r_std_c = np.mean(r_list_c), np.std(r_list_c)
        inf_mean_c = np.mean(inf_list_c)
        print(f"  > CompCBF    | 奖励: {r_mean_c:6.2f} ± {r_std_c:5.2f} | 平均 QP 无解次数: {inf_mean_c:.1f}")
        table_data.append(["Proposed CompCBF", f"{ratio}", f"{r_mean_c:.2f} ± {r_std_c:.2f}", f"{inf_mean_c:.1f}"])

    # =====================================================================
    # 6. 生成并保存对比表格图片
    # =====================================================================
    print("\n正在生成并保存性能对比表格...")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('tight')
    ax.axis('off')
    
    col_labels = ['Method', 'Defender Speed Ratio', 'Episode Reward\n(Mean ± Std)', 'Avg QP Infeasible\nSteps per Ep']
    
    table = ax.table(cellText=table_data, 
                     colLabels=col_labels, 
                     loc='center', 
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5) 
    
    # 样式美化：表头加粗，奇偶行换色
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=12)
            cell.set_facecolor('#2C3E50') # 深蓝色表头
        else:
            if "Proposed" in table_data[i-1][0]:
                cell.set_text_props(weight='bold', color='#006400') # 突出显示提出的方法
            
            if i % 2 == 0:
                cell.set_facecolor('#F8F9F9')
            else:
                cell.set_facecolor('#FFFFFF')

    plt.title(f"Quantitative Performance Comparison (Over {num_episodes} Randomized Episodes)", 
              fontweight="bold", fontsize=14, pad=20)
    plt.tight_layout()
    
    save_path = "performance_comparison_v2.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"表格已成功保存至当前目录: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_and_compare()