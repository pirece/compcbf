# robotarium_attacker_defender_env.py

import numpy as np

def wrap(a):
    """wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class UnicycleHOCBFEnvRobotarium:
    """
    攻击者-防守者场景的 Unicycle + 输入升阶 HOCBF 环境。

    特点：
      - 只对攻击者做 RL + LSE-HOCBF-QP
      - 防守者速度为攻击者的一定比例，使用视线角(LOS)控制器拦截
      - CBF 包含：
          1) 对静态圆障碍物：圆心 static_center，半径 static_radius
          2) 对防守者：距离 > 捕捉半径 capture_radius（动态障碍，考虑 defender 速度对 h_dot 的影响）
          3) 输入约束 CBF（对攻击者角速度 omega 的上下界）
    状态: [x_a, y_a, th_a, om_a,  x_d, y_d, th_d, om_d]
    动作为: 攻击者标称角速度 u_nom (scalar)
    """

    def __init__(self,
                 dt=0.05,
                 T_max=50.0,
                 v_attacker=0.08,
                 defender_speed_ratio=0.4,
                 a=2.0,
                 goal=None,
                 kappa=100.0,
                 omega_min=-1.0,
                 omega_max=1.0,
                 goal_tol=0.2,
                 safe_margin=0.0,
                 los_k=2.0):
        # 仿真参数
        self.dt = dt
        self.max_steps = int(T_max / dt)

        # 运动学参数
        self.v_a = v_attacker
        self.v_d = defender_speed_ratio * v_attacker
        self.a = a                         # 角速度一阶系统系数
        self.safe_margin = safe_margin

        # 静态圆障碍物（几何意义上的“静态威胁”）
        # 注：这里的 static_radius 直接用于 CBF 几何函数 h = ||p - c||^2 - R^2
        self.static_center = np.array([0.5, 0.3], dtype=np.float64)
        self.static_radius = 0.35

        # 防守者捕捉半径（动态圆障碍）
        self.capture_radius = 0.1

        # “障碍物”数量（1 个静态 + 1 个动态防守者）
        self._num_static_obs = 1
        self._num_obs = self._num_static_obs + 1

        # 目标点（攻击者）
        if goal is None:
            self.goal = np.array([1.2, 0.6], dtype=np.float64)
        else:
            self.goal = np.asarray(goal, dtype=np.float64)

        self.kappa = kappa
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.goal_tol = goal_tol

        # LOS 控制增益（防守者）
        self.los_k = los_k

        # 状态: 攻击者 + 防守者
        self.state_a = None
        self.state_d = None
        self.step_count = 0

        # 奖励相关缓存（只对攻击者-目标距离）
        self.prev_dist = None
        self.prev_theta_err = None

        # 奖励权重（和原环境一致）
        self.w_dist = 8.0
        self.w_time = 0.01
        self.k_th = 0.01  # 目前 r_heading 置零

        # 可视化 / Robotarium 区域
        self.x_range = [-1.6, 1.6]
        self.y_range = [-1.0, 1.0]

        # 初始状态（与 reset 中保持一致）
        self.state_a = np.array(
            [-1.2, -0.8, 2.5 * np.pi / 15.0, 0.0],
            dtype=np.float64
        )

        # 防守者初始状态：放在中部偏左下，朝向略指向攻击者
        self.state_d = np.array(
            [0.0, 0.0, -5 * np.pi / 6.0, 0.0],
            dtype=np.float64
        )

    # ---------- 属性 ----------
    @property
    def obs_dim(self):
        # [x_a, y_a, th_a, om_a,  x_d, y_d, th_d, om_d]
        return 8

    @property
    def act_dim(self):
        # 标称角速度 u_nom
        return 1

    @property
    def num_obs(self):
        # CBF 原始障碍物数量（静态 + 动态防守者）
        return self._num_obs

    @property
    def dt_step(self):
        return self.dt

    @property
    def dt_sim(self):
        # 为兼容旧代码中 env.dt 的用法
        return self.dt

    # ---------- 内部工具 ----------
    def _get_obs(self):
        return np.concatenate([self.state_a, self.state_d], axis=0)

    # ---------- reset ----------
    def reset(self):
        # 攻击者初始状态：沿用原 UnicycleHOCBFEnv
        self.state_a = np.array(
            [-1.2, -0.8, 2.5 * np.pi / 15.0, 0.0],
            dtype=np.float64
        )

        # 防守者初始状态
        self.state_d = np.array(
            [0.0, 0.0, -5 * np.pi / 6.0, 0.0],
            dtype=np.float64
        )

        self.step_count = 0

        # 攻击者-目标距离
        self.prev_dist = np.linalg.norm(self.state_a[:2] - self.goal)

        e = self.goal - self.state_a[:2]
        theta_des = np.arctan2(e[1], e[0])
        self.prev_theta_err = wrap(theta_des - self.state_a[2])

        return self._get_obs().copy()

    # ---------- 防守者 LOS 控制 ----------
    def _defender_control(self, p_a, p_d, theta_d):
        """
        视线角 LOS 控制器：
            u_d = k * wrap(psi - theta_d)
        其中 psi 为从防守者指向攻击者的方位角。
        """
        delta = p_a - p_d
        psi = np.arctan2(delta[1], delta[0])
        e_psi = wrap(psi - theta_d)
        u_d = self.los_k * e_psi
        # 饱和在相同的角速度限制内
        u_d = np.clip(u_d, self.omega_min, self.omega_max)
        return u_d

    # ---------- step ----------
    def step(self, u_nom, k1_vec, k2_vec, alpha_c):
        """
        输入:
            u_nom:   攻击者标称角速度 (scalar)
            k1_vec, k2_vec: shape = (num_obs,)
            alpha_c: 标量，LSE 组合 CBF 的增益
        输出:
            obs_next, reward, done, info
        """
        # 解包攻击者、防守者状态
        x_a, y_a, th_a, om_a = self.state_a
        x_d, y_d, th_d, om_d = self.state_d

        p_a = np.array([x_a, y_a], dtype=np.float64)
        p_d = np.array([x_d, y_d], dtype=np.float64)

        v_a = self.v_a
        v_d = self.v_d
        a = self.a

        # --- 防守者控制提前算好（用于动态障碍中心速度） ---
        # 注意：该控制不会影响当前时刻 c_d(t) 的位置，
        # 但影响 c_d 的加速度等漂移项，可用于更严格推导。
        u_d = self._defender_control(p_a, p_d, th_d)

        # 防守者当前线速度 & 角速度导数（用于动态障碍的中心速度近似）
        c_d_dot = v_d * np.array([np.cos(th_d), np.sin(th_d)], dtype=np.float64)
        # 如需更严格的三阶推导，可继续引入 c_d_ddot、om_d_dot 等项

        # --- CBF 准备 ---
        omega_hat_nom = float(u_nom)

        h_geom_list = []    # 各个“障碍物”的几何 h
        psi2_obs_list = []  # 对应 psi2
        h_all_list = []     # 用于 LSE 的原始 CBF 构造量（这里用 psi2/phi）
        psi2_all_list = []
        Lf_list = []        # 对应 CBF（psi2 或 phi）的 L_f
        Lg_list = []        # 对应 CBF（psi2 或 phi）的 L_g（只对攻击者控制）

        cos_t = np.cos(th_a)
        sin_t = np.sin(th_a)

        min_geom_dist = np.inf

        # 遍历“障碍物”：i=0 -> 静态圆, i=1 -> 动态防守者捕捉圆
        for i in range(self._num_obs):
            if i == 0:
                # --- 静态圆障碍 ---
                c = self.static_center
                R = self.static_radius
                d = p_a - c
                dist = np.linalg.norm(d)
                min_geom_dist = min(min_geom_dist, dist)

                # 在攻击者车体坐标系下的投影
                q = d @ np.array([cos_t, sin_t])
                r = d @ np.array([-sin_t, cos_t])

                # 原始几何 CBF
                h = d @ d - R ** 2
                h_geom_list.append(h)

                # 一阶、二阶导数（静态障碍推导）
                h_dot = 2 * v_a * q
                h_ddot = 2 * v_a ** 2 + 2 * v_a * r * om_a

            else:
                # --- 动态防守者圆障碍 ---
                # 将防守者视作动态圆障碍：中心 c = p_d(t)，
                # 并在 h_dot 中显式考虑中心速度 c_d_dot 造成的漂移项。
                c = p_d
                R = self.capture_radius
                d = p_a - c
                dist = np.linalg.norm(d)
                min_geom_dist = min(min_geom_dist, dist)

                q = d @ np.array([cos_t, sin_t])
                r = d @ np.array([-sin_t, cos_t])

                # 几何 CBF
                h = d @ d - R ** 2
                h_geom_list.append(h)

                # ---- 动态障碍的一阶导数（考虑 c_d_dot）----
                # \dot h = 2 d^T (p_dot - c_dot)
                # p_dot = v_a [cos th_a, sin th_a], c_dot = c_d_dot
                p_dot_a = v_a * np.array([cos_t, sin_t], dtype=np.float64)
                h_dot = 2.0 * d @ (p_dot_a - c_d_dot)

                # 二阶导数为了保持与原解析结构一致，仍采用静态推导形式：
                # h_ddot ≈ 2 v_a^2 + 2 v_a r om_a
                # 动态障碍对更高阶的漂移项统一吸收进 L_f 中（通过 h_dot, h 等参与 psi2）。
                h_ddot = 2 * v_a ** 2 + 2 * v_a * r * om_a

            # 取对应障碍的 HOCBF 参数
            k1 = float(k1_vec[i])
            k2 = float(k2_vec[i])

            # 二阶 HOCBF 中的 psi2
            psi2 = h_ddot + (k1 + k2) * h_dot + k1 * k2 * h
            psi2_obs_list.append(psi2)

            # 三阶导 involving 控制输入的部分（静态推导结构）
            # h^{(3)} 中与 \hat\omega 相关的项给出 L_g psi2 = 2 a v_a r
            # 其余项统一吸收进 L_f。
            Phi = -2 * v_a * (om_a ** 2) * q - 2 * a * v_a * r * om_a
            Lf_psi = Phi + (k1 + k2) * h_ddot + k1 * k2 * h_dot
            Lg_psi = 2 * a * v_a * r

            h_all_list.append(psi2)
            psi2_all_list.append(psi2)
            Lf_list.append(Lf_psi)
            Lg_list.append(Lg_psi)

        # ---- 输入约束 CBF（针对攻击者角速度）----
        phi_min = om_a - self.omega_min
        phi_max = self.omega_max - om_a

        h_all_list.append(phi_min)
        h_all_list.append(phi_max)
        psi2_all_list.append(phi_min)
        psi2_all_list.append(phi_max)

        # \dot phi = L_f phi + L_g phi * \hat\omega
        Lf_list.append(-a * om_a)  # varphi_1: omega - omega_min
        Lf_list.append(a * om_a)   # varphi_2: omega_max - omega
        Lg_list.append(a)
        Lg_list.append(-a)

        h_all = np.array(h_all_list, dtype=np.float64)
        psi2_all = np.array(psi2_all_list, dtype=np.float64)
        Lf_arr = np.array(Lf_list, dtype=np.float64)
        Lg_arr = np.array(Lg_list, dtype=np.float64)

        # ---- LSE 组合 CBF H ----
        exp_terms = np.exp(-self.kappa * psi2_all)
        S = np.sum(exp_terms)
        lambdas = exp_terms / (S + 1e-12)
        H = -(1.0 / self.kappa) * np.log(S + 1e-12)

        Lf_H = np.sum(lambdas * Lf_arr)
        Lg_H = np.sum(lambdas * Lg_arr)

        # ---- 单约束 QP 解析解（H(\hat x,t) >= 0）----
        A = float(Lg_H)
        b = -Lf_H - float(alpha_c) * H

        if A * omega_hat_nom >= b:
            omega_hat_a = omega_hat_nom
        else:
            gamma = (b - A * omega_hat_nom) / (A * A + 1e-12)
            omega_hat_a = omega_hat_nom + gamma * A

        # ---- 动力学更新 ----
        # 攻击者
        om_dot_a = -a * om_a + a * omega_hat_a
        om_a = om_a + om_dot_a * self.dt
        om_a = np.clip(om_a, self.omega_min, self.omega_max)

        th_a = th_a + om_a * self.dt
        x_a = x_a + v_a * np.cos(th_a) * self.dt
        y_a = y_a + v_a * np.sin(th_a) * self.dt

        # 防守者：LOS 控制
        om_dot_d = -a * om_d + a * u_d
        om_d = om_d + om_dot_d * self.dt
        om_d = np.clip(om_d, self.omega_min, self.omega_max)

        th_d = th_d + om_d * self.dt
        x_d = x_d + v_d * np.cos(th_d) * self.dt
        y_d = y_d + v_d * np.sin(th_d) * self.dt

        # 写回状态
        self.state_a = np.array([x_a, y_a, th_a, om_a], dtype=np.float64)
        self.state_d = np.array([x_d, y_d, th_d, om_d], dtype=np.float64)
        self.step_count += 1

        # ---- 奖励 & 终止条件 ----
        d_goal = np.linalg.norm(self.state_a[:2] - self.goal)
        # 距离项（与原版本一致）
        r_dist = self.w_dist * (self.prev_dist - d_goal)
        # 时间惩罚
        r_time = -self.w_time
        # 航向项暂不使用
        r_heading = 0.0

        reward = r_dist + r_heading + r_time
        r_near_goal = 0.0

        done = False
        success = False
        captured = False

        # 目标成功
        if d_goal < self.goal_tol:
            reward += 10.0 + 0.06 * (-self.step_count + self.max_steps)
            done = True
            success = True

        # 被捕捉：攻击者与防守者之间距离小于捕捉半径
        dist_ad = np.linalg.norm(self.state_a[:2] - self.state_d[:2])
        if dist_ad <= self.capture_radius:
            # 强烈惩罚
            #reward -= 20.0
            done = True
            captured = True

        # 时间耗尽
        if self.step_count >= self.max_steps:
            done = True

        self.prev_dist = d_goal

        # 安全半径集合（静态障碍 + 捕捉圆）
        radii_all = np.array([self.static_radius, self.capture_radius], dtype=np.float64)
        min_safe_radius = float(np.min(radii_all + self.safe_margin))

        # 障碍物部分的最小 h/psi2
        h_geom_arr = np.array(h_geom_list, dtype=np.float64)
        psi2_obs_arr = np.array(psi2_obs_list, dtype=np.float64)
        h_geom_min = float(np.min(h_geom_arr)) if h_geom_arr.size > 0 else 0.0
        psi2_min = float(np.min(psi2_obs_arr)) if psi2_obs_arr.size > 0 else 0.0

        info = {
            # 控制 & CBF
            "omega": om_a,
            "omega_hat": omega_hat_a,
            "H": H,
            "min_dist": float(min_geom_dist),
            "min_safe_radius": min_safe_radius,
            "h_geom_min": h_geom_min,
            "psi2_min": psi2_min,
            "phi_min": float(phi_min),
            "phi_max": float(phi_max),
            "alpha_c": float(alpha_c),
            "k1": np.array(k1_vec, copy=True),
            "k2": np.array(k2_vec, copy=True),
            "reward": reward,
            "u_nom": float(u_nom),

            # 奖励分解
            "r_dist": float(r_dist),
            "r_time": float(r_time),
            "r_heading": float(r_heading),
            "r_near_goal": float(r_near_goal),

            # 场景相关
            "attacker_pos": self.state_a[:2].copy(),
            "defender_pos": self.state_d[:2].copy(),
            "dist_ad": float(dist_ad),
            "success": success,
            "captured": captured,
        }

        return self._get_obs().copy(), reward, done, info
