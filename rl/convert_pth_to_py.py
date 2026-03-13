# convert_pth_to_py.py

import torch
import torch.nn as nn
import numpy as np

# ==========================================================
# 1. 复刻 PyTorch 网络架构（仅用于加载 state_dict）
# ==========================================================
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

class GaussianPenaltyNet(nn.Module):
    def __init__(self, obs_dim, num_obs, hidden=64):
        super().__init__()
        self.num_obs = num_obs
        # 注意：这里是 CompCBF (提议方法) 的维度公式
        self.latent_dim = 2 * num_obs + 1

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(hidden, self.latent_dim)
        self.log_std_layer = nn.Linear(hidden, self.latent_dim)

# ==========================================================
# 2. 一键提取并写入 weights_dict.py
# ==========================================================
def convert_weights_directly():
    obs_dim = 8
    act_dim = 1
    act_limit = 2.0
    
    # 新环境：3 个静态障碍物 + 1 个动态防守者 = 4
    num_obs = 4  

    actor = ActorGaussian(obs_dim, act_dim, act_limit)
    pn = GaussianPenaltyNet(obs_dim, num_obs)

    print("正在加载 PyTorch 权重文件 (_v2)...")
    try:
        actor.load_state_dict(torch.load("actor_sac_robotarium_att_def_v2.pth", map_location="cpu"))
        pn.load_state_dict(torch.load("penalty_sac_robotarium_att_def_v2.pth", map_location="cpu"))
    except Exception as e:
        print(f"权重加载失败，请检查 .pth 文件是否存在以及 num_obs 的维度是否匹配: {e}")
        return

    weights = {}

    # 提取 Actor 权重 (转置以适配 NumPy 中的 x @ W + b)
    weights["actor_W1"] = actor.net[0].weight.detach().numpy().T
    weights["actor_b1"] = actor.net[0].bias.detach().numpy()
    weights["actor_W2"] = actor.net[2].weight.detach().numpy().T
    weights["actor_b2"] = actor.net[2].bias.detach().numpy()
    weights["actor_W_mu"] = actor.mu_layer.weight.detach().numpy().T
    weights["actor_b_mu"] = actor.mu_layer.bias.detach().numpy()

    # 提取 PenaltyNet 权重
    weights["penalty_W1"] = pn.backbone[0].weight.detach().numpy().T
    weights["penalty_b1"] = pn.backbone[0].bias.detach().numpy()
    weights["penalty_W2"] = pn.backbone[2].weight.detach().numpy().T
    weights["penalty_b2"] = pn.backbone[2].bias.detach().numpy()
    weights["penalty_W_mu"] = pn.mu_layer.weight.detach().numpy().T
    weights["penalty_b_mu"] = pn.mu_layer.bias.detach().numpy()

    output_py_file = "weights_dict_v2.py"
    print(f"正在生成 {output_py_file} ... (这可能需要几秒钟)")
    
    with open(output_py_file, 'w', encoding='utf-8') as f:
        f.write("# 自动生成的硬编码权重字典 (适用于纯 NumPy 推理)\n")
        f.write("import numpy as np\n\n")
        f.write("HARDCODED_WEIGHTS = {\n")
        
        for key, val in weights.items():
            # 写入为 NumPy 数组字符串
            f.write(f"    '{key}': np.array({val.tolist()}, dtype=np.float32),\n")
            
        f.write("}\n")

    print(f"转换成功！权重代码已保存至: {output_py_file}")

if __name__ == "__main__":
    convert_weights_directly()