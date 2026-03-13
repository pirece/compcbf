import torch
import numpy as np

def convert_pth_to_hardcoded_py(actor_pth_path, penalty_pth_path, output_py_path):
    actor_state = torch.load(actor_pth_path, map_location='cpu')
    penalty_state = torch.load(penalty_pth_path, map_location='cpu')

    with open(output_py_path, 'w', encoding='utf-8') as f:
        f.write("# 自动生成的网络权重文件，供 NumPy 前向推理使用\n")
        f.write("import numpy as np\n\n")
        f.write("HARDCODED_WEIGHTS = {\n")

        for key, tensor_val in actor_state.items():
            formatted_key = key.replace('.', '_')
            np_array = tensor_val.numpy()
            f.write(f"    'actor_{formatted_key}': np.array({np_array.tolist()}),\n")

        for key, tensor_val in penalty_state.items():
            formatted_key = key.replace('.', '_')
            np_array = tensor_val.numpy()
            f.write(f"    'penalty_{formatted_key}': np.array({np_array.tolist()}),\n")

        f.write("}\n")
        
    print(f"权重转换完成，已成功保存至: {output_py_path}")

if __name__ == "__main__":
    convert_pth_to_hardcoded_py(
        "actor_sac_hocbf_att_def_v2.pth", 
        "penaltynet_sac_hocbf_att_def_v2.pth", 
        "weights_dict_hocbf_v2.py"
    )