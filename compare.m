% =========================================================================
% Script Name: plot_v2_comparisons.m
% Description: Reads data from V2 environments and generates a 2x2 comparison.
%              Adapted for 3 static obstacles (narrow passage scenario).
% =========================================================================

clear; close all; clc;

%% 1. Configuration & Formatting
% Set default text interpreter to LaTeX for publication-quality rendering
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

% Define standard colors
color_prop = [0, 0.4470, 0.7410];      % Blue (Proposed)
color_base = [0.8500, 0.3250, 0.0980]; % Orange (Baseline)
color_h    = [0.4660, 0.6740, 0.1880]; % Green (for Comp H)

% Load Data (Updated file names for V2)
file_prop = 'experiment_results_compcbf.mat';
file_base = 'experiment_results_hocbf.mat';

if ~isfile(file_prop) || ~isfile(file_base)
    error('Data files not found. Please ensure both .mat files are in the current directory.');
end

prop = load(file_prop);
base = load(file_base);

% Extract common Time vectors (Ensure they are column vectors)
t_p = prop.time(:);
t_b = base.time(:);

%% 2. Create Figure Layout (2x2 Grid)
fig = figure('Name', 'Core Comparisons V2', 'Position', [100, 100, 1000, 800], 'Color', 'w');
t = tiledlayout(2, 2, 'TileSpacing', 'normal', 'Padding', 'compact');

%% ------------------------------------------------------------------------
% (a) Trajectory Comparison (轨迹对比 - 适配 3 个障碍物)
% ------------------------------------------------------------------------
nexttile; hold on; grid on; box on; axis equal;

% 1. Draw 3 Static Obstacles (Narrow Passage)
obs_centers = [0.0, 0.45; 
               0.0, -0.45; 
               0.6, 0.0];
obs_radii = [0.3, 0.3, 0.3];
th = linspace(0, 2*pi, 100);

for i = 1:3
    obs_x = obs_centers(i, 1) + obs_radii(i) * cos(th); 
    obs_y = obs_centers(i, 2) + obs_radii(i) * sin(th);
    % 使用 fill 填充灰色半透明，让障碍物区域更清晰
    h_obs = fill(obs_x, obs_y, [0.8, 0.8, 0.8], 'EdgeColor', 'r', 'LineStyle', '--', 'FaceAlpha', 0.3);
    text(obs_centers(i,1), obs_centers(i,2), sprintf('Obs %d', i), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Interpreter', 'latex');
end

% 2. Draw Target Goal
goal_x = 1.2 + 0.2*cos(th); 
goal_y = 0.6 + 0.2*sin(th);
h_goal = plot(goal_x, goal_y, 'g--', 'LineWidth', 1.5);
text(1.2, 0.6, 'Target', 'HorizontalAlignment', 'center', 'Interpreter', 'latex');

% 3. Plot Attacker Trajectories
p_prop = plot(prop.traj_attacker(:,1), prop.traj_attacker(:,2), 'LineWidth', 1.8, 'Color', color_prop);
p_base = plot(base.traj_attacker(:,1), base.traj_attacker(:,2), 'LineWidth', 1.8, 'Color', color_base);

% 4. Plot Defender Trajectories (Dotted lines)
plot(prop.traj_defender(:,1), prop.traj_defender(:,2), ':', 'LineWidth', 1.5, 'Color', color_prop);
plot(base.traj_defender(:,1), base.traj_defender(:,2), ':', 'LineWidth', 1.5, 'Color', color_base);

% Format axes
xlabel('$X$ (m)', 'FontSize', 12); 
ylabel('$Y$ (m)', 'FontSize', 12);
xlim([-1.5, 1.6]); ylim([-1.2, 1.2]);

% Use specific handles for legend to avoid multiple "Obstacle" entries
legend([h_obs, h_goal, p_prop, p_base], {'Obstacle', 'Target', 'Proposed', 'HOCBF-QP'}, ...
    'Location', 'northwest', 'FontSize', 10);
title('(a) Test Trajectories', 'FontSize', 12);

%% ------------------------------------------------------------------------
% (b) Control Input Comparison (控制量对比)
% ------------------------------------------------------------------------
nexttile; hold on; grid on; box on;

% Plot Actual Control (omega) for both methods
plot(t_p, prop.omega(:), '-', 'LineWidth', 1.5, 'Color', color_prop);
plot(t_b, base.omega(:), '-', 'LineWidth', 1.5, 'Color', color_base);

% Plot Nominal Control of Proposed method as reference
plot(t_p, prop.u_nom(:), 'k:', 'LineWidth', 1.2, 'Color', [0.5 0.5 0.5]);

xlabel('Time $t$ (s)', 'FontSize', 12); 
ylabel('Angular Velocity $\omega$ (rad/s)', 'FontSize', 12);
legend({'Proposed $\omega$', 'HOCBF-QP $\omega$', 'Proposed $u_{nom}$'}, 'Location', 'best', 'FontSize', 10);
title('(b) Control Inputs', 'FontSize', 12);

%% ------------------------------------------------------------------------
% (c) CBF Curves Comparison (CBF曲线对比: h_min 和 组合 H)
% ------------------------------------------------------------------------
nexttile; hold on; grid on; box on;

% Plot Minimum Geometric CBF value (h_geom_min) for both methods
plot(t_p, prop.h_geom_min(:), '-', 'LineWidth', 1.5, 'Color', color_prop);
plot(t_b, base.h_geom_min(:), '-', 'LineWidth', 1.5, 'Color', color_base);

% Plot Composite CBF (H) for Proposed method 
plot(t_p, prop.H(:), '--', 'LineWidth', 1.5, 'Color', color_h);

% Draw Safe Threshold (Zero Line)
yline(0, 'k--', 'LineWidth', 1.0);

xlabel('Time $t$ (s)', 'FontSize', 12); 
ylabel('CBF Values', 'FontSize', 12);
legend({'Proposed $h_{\min}$', 'HOCBF-QP $h_{\min}$', 'Proposed Comp. $H$'}, 'Location', 'best', 'FontSize', 10);
title('(c) CBF Values', 'FontSize', 12);

%% ------------------------------------------------------------------------
% (d) Reward Function Comparison (累积奖励对比)
% ------------------------------------------------------------------------
nexttile; hold on; grid on; box on;

% 使用 cumsum 计算累积奖励
plot(t_p, cumsum(prop.reward(:)), '-', 'LineWidth', 1.5, 'Color', color_prop);
plot(t_b, cumsum(base.reward(:)), '-', 'LineWidth', 1.5, 'Color', color_base);

xlabel('Time $t$ (s)', 'FontSize', 12); 
ylabel('Cumulative Reward', 'FontSize', 12); 
legend({'Proposed', 'HOCBF-QP'}, 'Location', 'best', 'FontSize', 10);
title('(d) Cumulative Rewards during Test', 'FontSize', 12);

%% 3. Final Adjustments
% Add an overall title to the figure
title(t, 'Comparison between Proposed Method and Baseline (Narrow Passage)', ...
    'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');