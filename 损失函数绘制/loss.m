% =========================================================================
% 论文对比图绘制：CompCBF (Ours) vs HOCBF-QP (Baseline)
% 绘制内容：Returns, Q-Loss, Pi-Loss, PenaltyNet-Loss
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

%% 3. 绘图参数设置
window_size = 15; % 滑动平均的窗口大小（可根据震荡程度自行调大或调小）

% 定义颜色 (使用 MATLAB 经典的清晰配色)
color_ours = [0.0000, 0.4470, 0.7410]; % 蓝色 (Ours)
color_base = [0.8500, 0.3250, 0.0980]; % 橙红色 (Baseline)

% 浅色用于绘制原始带有噪音的背景曲线
alpha_val = 0.3; % 透明度模拟 (通过颜色混合)
color_ours_light = color_ours * alpha_val + [1,1,1] * (1 - alpha_val);
color_base_light = color_base * alpha_val + [1,1,1] * (1 - alpha_val);

% 线宽与字体大小
line_width_smooth = 2.0;
line_width_raw = 1.0;
font_size = 12;

%% 4. 创建图形窗口
figure('Name', 'Training Curves Comparison', 'Position', [100, 100, 1100, 750], 'Color', 'w');

%% ------ 子图 1: Episode Returns (总回报) ------
subplot(2, 2, 1);
hold on; grid on; box on;
% 绘制原始曲线 (浅色)
plot(data_base.returns, 'Color', color_base_light, 'LineWidth', line_width_raw);
plot(data_ours.returns, 'Color', color_ours_light, 'LineWidth', line_width_raw);
% 绘制平滑曲线 (深色)
plot(smoothdata(data_base.returns, 'movmean', window_size), 'Color', color_base, 'LineWidth', line_width_smooth);
plot(smoothdata(data_ours.returns, 'movmean', window_size), 'Color', color_ours, 'LineWidth', line_width_smooth);

title('Episode Returns', 'FontSize', font_size, 'FontWeight', 'bold');
xlabel('Episode', 'FontSize', font_size);
ylabel('Return', 'FontSize', font_size);
% 图例只标注平滑线
h1 = plot(nan, nan, 'Color', color_base, 'LineWidth', line_width_smooth);
h2 = plot(nan, nan, 'Color', color_ours, 'LineWidth', line_width_smooth);
legend([h1, h2], {'HOCBF-QP', 'CompCBF (Ours)'}, 'Location', 'southeast', 'FontSize', 10);

%% ------ 子图 2: Actor Loss (Pi Loss) ------
subplot(2, 2, 2);
hold on; grid on; box on;
plot(data_base.pi_loss, 'Color', color_base_light, 'LineWidth', line_width_raw);
plot(data_ours.pi_loss, 'Color', color_ours_light, 'LineWidth', line_width_raw);
plot(smoothdata(data_base.pi_loss, 'movmean', window_size), 'Color', color_base, 'LineWidth', line_width_smooth);
plot(smoothdata(data_ours.pi_loss, 'movmean', window_size), 'Color', color_ours, 'LineWidth', line_width_smooth);

title('Actor Loss (\pi_{loss})', 'FontSize', font_size, 'FontWeight', 'bold');
xlabel('Episode', 'FontSize', font_size);
ylabel('Loss', 'FontSize', font_size);

%% ------ 子图 3: Critic Loss (Q1 Loss) ------
% (通常 Q1 和 Q2 趋势一致，展示一个即可)
subplot(2, 2, 3);
hold on; grid on; box on;
plot(data_base.q1_loss, 'Color', color_base_light, 'LineWidth', line_width_raw);
plot(data_ours.q1_loss, 'Color', color_ours_light, 'LineWidth', line_width_raw);
plot(smoothdata(data_base.q1_loss, 'movmean', window_size), 'Color', color_base, 'LineWidth', line_width_smooth);
plot(smoothdata(data_ours.q1_loss, 'movmean', window_size), 'Color', color_ours, 'LineWidth', line_width_smooth);

title('Critic Loss (Q1_{loss})', 'FontSize', font_size, 'FontWeight', 'bold');
xlabel('Episode', 'FontSize', font_size);
ylabel('Loss', 'FontSize', font_size);
% 若早期 Loss 爆炸导致纵坐标跨度太大，可取消下面这行的注释来限制 Y 轴范围：
% ylim([0, 10]); 

%% ------ 子图 4: PenaltyNet Loss (PN Loss) ------
subplot(2, 2, 4);
hold on; grid on; box on;
plot(data_base.pn_loss, 'Color', color_base_light, 'LineWidth', line_width_raw);
plot(data_ours.pn_loss, 'Color', color_ours_light, 'LineWidth', line_width_raw);
plot(smoothdata(data_base.pn_loss, 'movmean', window_size), 'Color', color_base, 'LineWidth', line_width_smooth);
plot(smoothdata(data_ours.pn_loss, 'movmean', window_size), 'Color', color_ours, 'LineWidth', line_width_smooth);

title('PenaltyNet Loss', 'FontSize', font_size, 'FontWeight', 'bold');
xlabel('Episode', 'FontSize', font_size);
ylabel('Loss', 'FontSize', font_size);

% 调整整体布局
set(gcf, 'PaperPositionMode', 'auto');
sgtitle('Training Process Comparison: CompCBF vs HOCBF-QP', 'FontSize', 14, 'FontWeight', 'bold');

disp('绘图完成！可以利用 MATLAB 图窗上方菜单栏：文件 -> 导出设置，将其导出为高清 PDF 或 PNG 供论文使用。');