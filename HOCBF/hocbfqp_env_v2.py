# robotarium_attacker_defender_env.py
# Final version matching updated train/test (no alpha_c, no omega_hat, no H, no psi2_min)
# Removed cvxpy dependency. Using highly efficient Analytical 1D QP Solver.

import numpy as np

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class UnicycleHOCBFEnvRobotarium:
    """
    Unicycle attacker-defender environment using HOCBF-QP.
    (No input lift, no log-sum-exp, no alpha_c, no omega_hat.)
    step(u_nom, k1_vec, k2_vec)
    """

    def __init__(self,
                 dt=0.05,
                 T_max=50.0,
                 v_attacker=0.08,
                 defender_speed_ratio=0.4,
                 omega_min=-1.0,
                 omega_max=1.0,
                 goal=None,
                 goal_tol=0.2,
                 safe_margin=0.1):

        self.dt = dt
        self.max_steps = int(T_max / dt)

        self.v_a = v_attacker
        self.v_d = defender_speed_ratio * v_attacker

        # Static obstacles
        self.static_centers = np.array([
            [0.0, 0.45],
            [0.0, -0.45],
            [0.6, 0.0]
        ])
        self.static_radii = np.array([0.3, 0.3, 0.3])
        self._num_static_obs = 3

        self.capture_radius = 0.1
        self._num_obs = self._num_static_obs + 1  # 4 total

        if goal is None:
            self.goal = np.array([1.2, 0.6])
        else:
            self.goal = np.asarray(goal)

        self.omega_min = omega_min
        self.omega_max = omega_max
        self.goal_tol = goal_tol
        self.safe_margin = safe_margin

        # Attacker state: [x, y, theta, omega]
        self.state_a = None
        # Defender     : [x, y, theta, omega]
        self.state_d = None

        self.prev_dist = None
        self.step_count = 0

        # reward weights
        self.w_dist = 8.0
        self.w_time = 0.01

        self.reset()

    # ---------------------------------------------------------------------
    @property
    def obs_dim(self):
        return 8

    @property
    def act_dim(self):
        return 1

    @property
    def num_obs(self):
        return self._num_obs

    # ---------------------------------------------------------------------
    def reset(self):
        # 优化：攻击者和防守者的初始位置，制造“防守者从后方追击攻击者进入通道”的极限场景
        self.state_a = np.array([-1.2, -0.8, 2.5*np.pi/15, 0.0], float)
        self.state_d = np.array([0.0, 0.0, -5 * np.pi / 6.0, 0.0], float) 
        self.step_count = 0

        self.prev_dist = np.linalg.norm(self.state_a[:2] - self.goal)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.state_a, self.state_d])

    # ---------------------------------------------------------------------
    def _defender_control(self, pa, pd, thd):
        delta = pa - pd
        psi = np.arctan2(delta[1], delta[0])
        e = wrap(psi - thd)
        u = 2.0 * e
        return np.clip(u, self.omega_min, self.omega_max)

    @staticmethod
    def _solve_qp_1d(u_nom, A_list, b_list):
        """
        基于纯 Numpy 解析法求解 1D QP：
            min 0.5 (u - u_nom)^2
            s.t. A_i * u >= b_i
        """
        low = -np.inf
        high = np.inf
        tol = 1e-7

        for A, b in zip(A_list, b_list):
            if A > tol:
                low = max(low, b / A)
            elif A < -tol:
                high = min(high, b / A)
            else:
                if b > tol:
                    return float(u_nom), 1

        if low > high + tol:
            return float(u_nom), 1

        u_star = np.clip(u_nom, low, high)
        return float(u_star), 0

    # ---------------------------------------------------------------------
    def step(self, u_nom, k1_vec, k2_vec):
        x_a, y_a, th_a, om_a_prev = self.state_a
        x_d, y_d, th_d, om_d_prev = self.state_d

        pa = self.state_a[:2]
        pd = self.state_d[:2]

        v_a = self.v_a
        v_d = self.v_d

        # Defender control
        u_d = self._defender_control(pa, pd, th_d)

        # Defender motion 
        cdot = v_d * np.array([np.cos(th_d), np.sin(th_d)])
        om_d = u_d
        cddot = v_d * np.array([-np.sin(th_d)*om_d, np.cos(th_d)*om_d])

        # Attackers motion basis
        cos_t = np.cos(th_a)
        sin_t = np.sin(th_a)
        pa_dot = v_a * np.array([cos_t, sin_t])

        # ------------------ HOCBF constraints ------------------
        A_list = []
        B_list = []

        h_geom_min = np.inf
        min_dist = np.inf

        for j in range(self._num_obs):
            if j < self._num_static_obs:
                c = self.static_centers[j]
                R = self.static_radii[j]
                c_dot = np.zeros(2)
                c_ddot = np.zeros(2)
            else:
                c = pd
                R = self.capture_radius
                c_dot = cdot
                c_ddot = cddot

            d = pa - c
            dist = np.linalg.norm(d)
            min_dist = min(min_dist, dist)

            h = d @ d - R*R
            h_geom_min = min(h_geom_min, h)

            e1 = np.array([cos_t, sin_t])
            e2 = np.array([-sin_t, cos_t])

            q = d @ e1
            r = d @ e2

            d_dot = pa_dot - c_dot
            h_dot = 2 * d @ d_dot

            xi = -2 * d @ c_dot
            xi_dot = -2 * (d_dot @ c_dot + d @ c_ddot)

            Gamma = -2*v_a*(c_dot @ e1) + xi_dot

            alpha0 = float(k1_vec[j])
            alpha1 = float(k2_vec[j])

            A_j = 2 * v_a * r
            C_j = 2*v_a*v_a + Gamma + alpha1*h_dot + alpha0*h
            b_j = -C_j

            A_list.append(A_j)
            B_list.append(b_j)

        # input bounds
        A_list.append(1.0)
        B_list.append(self.omega_min)
        A_list.append(-1.0)
        B_list.append(-self.omega_max)
        
        # Solve 1D QP explicitly
        u_star, infeasible_cnt = self._solve_qp_1d(u_nom, A_list, B_list)

        # ------------------ Apply dynamics ------------------
        om_a = u_star
        th_a2 = th_a + om_a * self.dt
        x_a2 = x_a + v_a*np.cos(th_a2)*self.dt
        y_a2 = y_a + v_a*np.sin(th_a2)*self.dt

        th_d2 = th_d + u_d * self.dt
        x_d2 = x_d + v_d*np.cos(th_d2)*self.dt
        y_d2 = y_d + v_d*np.sin(th_d2)*self.dt

        self.state_a = np.array([x_a2, y_a2, th_a2, om_a])
        self.state_d = np.array([x_d2, y_d2, th_d2, u_d])
        self.step_count += 1

        # ------------------ Reward ------------------
        d_goal = np.linalg.norm(self.state_a[:2] - self.goal)
        r_dist = self.w_dist*(self.prev_dist - d_goal)
        r_time = -self.w_time
        reward = r_dist + r_time
        self.prev_dist = d_goal

        done = False
        success = False
        captured = False

        if d_goal < self.goal_tol:
            reward += 10.0 + 0.06 * (-self.step_count + self.max_steps)
            done = True
            success = True

        dist_ad = np.linalg.norm(self.state_a[:2] - self.state_d[:2])
        if dist_ad <= self.capture_radius:
            done = True
            captured = True

        if self.step_count >= self.max_steps:
            done = True

        info = {
            "omega": float(om_a),
            "min_dist": float(min_dist),
            # 修复：使用 np.append 将静态半径数组和动态捕捉半径标量拼接后再求 min
            "min_safe_radius": float(np.min(np.append(self.static_radii, self.capture_radius))),
            "h_geom_min": float(h_geom_min),
            "phi_min": float(om_a - self.omega_min),
            "phi_max": float(self.omega_max - om_a),

            "k1": np.array(k1_vec),
            "k2": np.array(k2_vec),

            "reward": float(reward),
            "attacker_pos": self.state_a[:2].copy(),
            "defender_pos": self.state_d[:2].copy(),
            "dist_ad": float(dist_ad),
            "success": success,
            "captured": captured,
            "infeasible_cnt": infeasible_cnt
        }

        info["r_dist"] = r_dist
        info["r_time"] = r_time

        return self._get_obs(), reward, done, info