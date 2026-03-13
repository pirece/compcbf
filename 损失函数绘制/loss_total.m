% =========================================================================
% 论文对比图绘制：总体 Loss (Total Loss) 曲线
% CompCBF (Ours) vs HOCBF-QP (Baseline)
% =========================================================================
clc; clear; close all;

%% 1. 文件名定义
file_ours = 'train_log_sac_robotarium_att_def_v2.mat'; % 所提方法 CompCBF
file_base = 'train_log_sac_hocbf_att_def_v2.mat';     % 对比方法 HOCBF-QP

% 检查文件是否存在
if ~isfile(file_ours) || ~isfile(file_base)
    error('未找到 .mat 文件，请检查当前工作路径下是否包含这两个日志文件！');
end

%% 2. 加载数据
data_ours = load(file_ours);
data_base = load(file_base);

%% 3. 计算总体 Loss
% SAC 算法中没有单一的总体 Loss，这里将四个主要 Loss 进行代数求和
% Total Loss = Q1_loss + Q2_loss + Pi_loss + PN_loss
total_loss_ours = data_ours.q1_loss + data_ours.q2_loss + data_ours.pi_loss + data_ours.pn_loss;
total_loss_base = data_base.q1_loss + data_base.q2_loss + data_base.pi_loss + data_base.pn_loss;

% 获取 Episode 数量 (假设两者训练回合数相同)
episodes = 1:length(total_loss_ours);

%% 4. 绘图参数设置
window_size = 15; % 滑动平均窗口大小 (数值越大越平滑)

% 颜色定义：蓝 (Ours) 和 红 (Baseline)
color_ours = [0.0000, 0.4470, 0.7410]; 
color_base = [0.8500, 0.3250, 0.0980]; 

% 浅色 (用于绘制含有噪声的原始曲线)
alpha_val = 0.25; 
color_ours_light = color_ours * alpha_val + [1,1,1] * (1 - alpha_val);
color_base_light = color_base * alpha_val + [1,1,1] * (1 - alpha_val);

% 线宽与字体
line_width_smooth = 2.5;
line_width_raw = 1.0;
font_size = 14;

%% 5. 绘制总体 Loss 曲线
figure('Name', 'Total Loss Comparison', 'Position', [200, 200, 800, 550], 'Color', 'w');
hold on; grid on; box on;

% 5.1 绘制原始数据的浅色折线
plot(episodes, total_loss_base, 'Color', color_base_light, 'LineWidth', line_width_raw);
plot(episodes, total_loss_ours, 'Color', color_ours_light, 'LineWidth', line_width_raw);

% 5.2 计算并绘制平滑后的深色曲线
smooth_base = smoothdata(total_loss_base, 'movmean', window_size);
smooth_ours = smoothdata(total_loss_ours, 'movmean', window_size);

h1 = plot(episodes, smooth_base, 'Color', color_base, 'LineWidth', line_width_smooth);
h2 = plot(episodes, smooth_ours, 'Color', color_ours, 'LineWidth', line_width_smooth);

% 5.3 图表装饰
title('Total Training Loss Comparison', 'FontSize', font_size + 2, 'FontWeight', 'bold');
xlabel('Episode', 'FontSize', font_size, 'FontWeight', 'bold');
ylabel('Total Loss (Q_1 + Q_2 + \pi + PN)', 'FontSize', font_size, 'FontWeight', 'bold');

% 设置坐标轴属性
set(gca, 'FontSize', 12, 'LineWidth', 1.2);
xlim([0, length(episodes)]);

% 如果初期 Loss 爆炸导致图形被压缩，可以取消下面这一行的注释来限制 Y 轴显示范围
ylim([-50, 150]); % 具体数值请根据你实际运行出来的图进行微调

% 5.4 图例设置 (只显示平滑曲线的图例，使得图面整洁)
legend([h1, h2], {'HOCBF-QP (Baseline)', 'CompCBF (Ours)'}, ...
    'Location', 'best', 'FontSize', 12, 'EdgeColor', [0.8 0.8 0.8]);

% 优化图像输出边距
set(gcf, 'PaperPositionMode', 'auto');

disp('总体 Loss 对比图绘制完成！');