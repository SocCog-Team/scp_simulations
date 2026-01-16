function scpsim_cpr_optimal_wagering_cursor
% SCPSIM_CPR_OPTIMAL_WAGERING_CURSOR - Find optimal wagering strategy
%
% Finds the optimal wagering function (tilt as function of accuracy) that
% maximizes cumulative reward for given psychometric function parameters.
%
% Supports batch processing over families of psychometric functions.
%
% OUTPUTS:
%   - Optimal wagering function for each psychometric function
%   - Performance metrics (cumulative reward, hit rate, etc.)
%   - Visualization of optimal strategies
%
% Created with Cursor AI for Igor Kagan, DPZ Göttingen

%% ========================================================================
%  CONFIGURATION SECTION
%  ========================================================================

% Set random seed for reproducibility
rng(42, 'twister');

% --- Task Parameters (same as main simulation) ---
CONFIG.coherence_levels = [0 36 59 98];
CONFIG.n_trials_per_coherence = 500;
CONFIG.min_arc_deg = 10;
CONFIG.max_arc_deg = 180;
CONFIG.reward_min_ml = 0.05;
CONFIG.reward_max_ml = 0.5;

% --- Psychometric Function(s) to Test ---
% Can specify single function or array of functions for batch processing
PSYCHOMETRIC_FUNCTIONS = struct();

% Current psychometric function
PSYCHOMETRIC_FUNCTIONS(1).baseline_acc = 0.5;
PSYCHOMETRIC_FUNCTIONS(1).max_acc = 0.85;
PSYCHOMETRIC_FUNCTIONS(1).slope = 0.08;
PSYCHOMETRIC_FUNCTIONS(1).threshold = 50;
PSYCHOMETRIC_FUNCTIONS(1).variability = 0.1;
PSYCHOMETRIC_FUNCTIONS(1).name = 'Current';

% Example: Add more psychometric functions for batch processing
% PSYCHOMETRIC_FUNCTIONS(2).baseline_acc = 0.5;
% PSYCHOMETRIC_FUNCTIONS(2).max_acc = 0.90;
% PSYCHOMETRIC_FUNCTIONS(2).slope = 0.10;
% PSYCHOMETRIC_FUNCTIONS(2).threshold = 50;
% PSYCHOMETRIC_FUNCTIONS(2).variability = 0.15;
% PSYCHOMETRIC_FUNCTIONS(2).name = 'High Variability';

% --- Wagering Function Parameterization ---
% Wagering function: tilt = f(accuracy)
% Parameterized as: tilt = tilt_min + (tilt_max - tilt_min) * g(accuracy)
% where g(accuracy) maps [acc_min, acc_max] -> [0, 1]
% 
% Options for g():
%   'linear': g(acc) = (acc - acc_min) / (acc_max - acc_min)
%   'power': g(acc) = ((acc - acc_min) / (acc_max - acc_min))^power_exp
%   'sigmoid': g(acc) = sigmoid((acc - acc_center) / acc_scale)
%
CONFIG.wagering.tilt_min = 0.01;  % Minimum tilt
CONFIG.wagering.tilt_max = 0.7;   % Maximum tilt
CONFIG.wagering.function_type = 'linear';  % 'linear', 'power', 'sigmoid'
CONFIG.wagering.power_exp = 1.0;  % For power function
CONFIG.wagering.acc_center = 0.65;  % For sigmoid function
CONFIG.wagering.acc_scale = 0.1;   % For sigmoid function
CONFIG.wagering.noise_std = 0.05;  % Gaussian noise on tilt

% --- Optimization Parameters ---
CONFIG.optimization.method = 'grid_search';  % 'grid_search', 'fmincon', 'genetic'
CONFIG.optimization.tilt_min_range = [0, 1];  % Search range for tilt_min (full range)
CONFIG.optimization.tilt_max_range = [0, 1];   % Search range for tilt_max (full range)
CONFIG.optimization.n_grid_points = 20;  % For grid search
% Note: Cumulative reward is calculated over all trials (CONFIG.n_trials_per_coherence × number of coherences)
CONFIG.optimization.n_fixed_tilt_points = 50;  % Number of fixed tilt values to test (0 to 1)

% --- Output Options ---
CONFIG.show_figures = true;
CONFIG.save_results = false;
CONFIG.results_filename = 'optimal_wagering_results.mat';

%% ========================================================================
%  MAIN OPTIMIZATION
%  ========================================================================

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  Optimal Wagering Strategy Finder                              ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

n_psychometric = length(PSYCHOMETRIC_FUNCTIONS);
optimal_results = struct();

for p = 1:n_psychometric
    psychometric = PSYCHOMETRIC_FUNCTIONS(p);
    fprintf('Processing psychometric function %d/%d: %s\n', p, n_psychometric, psychometric.name);
    fprintf('─────────────────────────────────────────────────────────────────\n');
    
    % Generate accuracy data for this psychometric function
    [accuracy_data, psychometric_stats] = generate_psychometric_data_batch(...
        CONFIG, psychometric);
    
    % Find optimal wagering parameters
    optimal_params = find_optimal_wagering(CONFIG, accuracy_data, psychometric_stats);
    
    % Evaluate optimal strategy
    optimal_performance = evaluate_wagering_strategy(...
        CONFIG, accuracy_data, optimal_params, psychometric);
    
    % Test fixed tilt strategies from 0 to 1 for comparison
    fixed_tilt_results = test_fixed_tilt_strategies(CONFIG, accuracy_data);
    
    % Store results
    optimal_results(p).psychometric = psychometric;
    optimal_results(p).optimal_params = optimal_params;
    optimal_results(p).performance = optimal_performance;
    optimal_results(p).psychometric_stats = psychometric_stats;
    optimal_results(p).psychometric_stats.accuracy_data = accuracy_data;  % Store for visualization
    optimal_results(p).fixed_tilt_results = fixed_tilt_results;  % Store fixed tilt comparison
    
    fprintf('  Optimal tilt_min: %.3f\n', optimal_params.tilt_min);
    fprintf('  Optimal tilt_max: %.3f\n', optimal_params.tilt_max);
    fprintf('  Cumulative reward: %.4f ml\n', optimal_performance.cumulative_reward);
    fprintf('  Hit rate: %.3f\n', optimal_performance.hit_rate);
    fprintf('\n');
end

%% ========================================================================
%  VISUALIZATION AND SAVING
%  ========================================================================

if CONFIG.show_figures
    create_optimal_wagering_figures(CONFIG, optimal_results);
end

if CONFIG.save_results
    save(CONFIG.results_filename, 'optimal_results', 'CONFIG', 'PSYCHOMETRIC_FUNCTIONS');
    fprintf('Results saved to: %s\n', CONFIG.results_filename);
end

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  Optimization Complete!                                        ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

end

%% ========================================================================
%  CORE FUNCTIONS
%  ========================================================================

function [accuracy_data, stats] = generate_psychometric_data_batch(CONFIG, psychometric)
% Generate accuracy distributions for given psychometric function

n_coherence = length(CONFIG.coherence_levels);
accuracy_data = cell(n_coherence, 1);

stats.mean_acc = zeros(n_coherence, 1);
stats.std_acc = zeros(n_coherence, 1);
stats.mean_error_deg = zeros(n_coherence, 1);

for i = 1:n_coherence
    coh = CONFIG.coherence_levels(i);
    
    [accuracy, mean_acc, std_acc] = generate_accuracy_distribution_batch(...
        coh, CONFIG.n_trials_per_coherence, psychometric);
    
    accuracy_data{i} = accuracy;
    stats.mean_acc(i) = mean_acc;
    stats.std_acc(i) = std_acc;
    stats.mean_error_deg(i) = (1 - mean_acc) * 180;
end

end

function [accuracy, mean_acc, std_acc] = generate_accuracy_distribution_batch(...
    coherence, n_trials, psychometric_params)
% Generate accuracy distribution for given coherence and psychometric parameters

if coherence == 0
    mean_acc = psychometric_params.baseline_acc;
else
    mean_acc = psychometric_params.baseline_acc + ...
        (psychometric_params.max_acc - psychometric_params.baseline_acc) / ...
        (1 + exp(-psychometric_params.slope * (coherence - psychometric_params.threshold)));
end

std_acc = psychometric_params.variability * sqrt(1 - coherence/100);
std_acc = max(std_acc, 0.02);

accuracy = generate_normal_samples_clipped(mean_acc, std_acc, n_trials);

end

function samples = generate_normal_samples_clipped(mu, sigma, n)
% Generate normally-distributed samples, clipped to [0,1]

samples = normrnd(mu, sigma, n, 1);
samples = max(0, min(1, samples));

if abs(mu - 0.5) < 0.01
    actual_mean = mean(samples);
    if abs(actual_mean - 0.5) > 0.001
        samples = samples + (0.5 - actual_mean);
        samples = max(0, min(1, samples));
    end
end

end

function optimal_params = find_optimal_wagering(CONFIG, accuracy_data, psychometric_stats)
% Find optimal wagering parameters using specified optimization method

switch CONFIG.optimization.method
    case 'grid_search'
        optimal_params = optimize_grid_search(CONFIG, accuracy_data, psychometric_stats);
    case 'fmincon'
        optimal_params = optimize_fmincon(CONFIG, accuracy_data, psychometric_stats);
    otherwise
        error('Unknown optimization method: %s', CONFIG.optimization.method);
end

end

function optimal_params = optimize_grid_search(CONFIG, accuracy_data, psychometric_stats)
% Grid search for optimal tilt_min and tilt_max
% Includes fixed tilt strategies (tilt_min == tilt_max) in the search
% Stores all tested combinations for optimality proof visualization

% Compute global accuracy range
all_accuracies = [];
for c = 1:length(accuracy_data)
    all_accuracies = [all_accuracies; accuracy_data{c}];
end
acc_min = min(all_accuracies);
acc_max = max(all_accuracies);

% Create grid
tilt_min_grid = linspace(CONFIG.optimization.tilt_min_range(1), ...
    CONFIG.optimization.tilt_min_range(2), CONFIG.optimization.n_grid_points);
tilt_max_grid = linspace(CONFIG.optimization.tilt_max_range(1), ...
    CONFIG.optimization.tilt_max_range(2), CONFIG.optimization.n_grid_points);

best_reward = -inf;
best_params = struct();

% Store all tested combinations for visualization
tested_params = [];
tested_rewards = [];

fprintf('  Grid search: %d x %d = %d evaluations\n', ...
    length(tilt_min_grid), length(tilt_max_grid), ...
    length(tilt_min_grid) * length(tilt_max_grid));

for i = 1:length(tilt_min_grid)
    for j = 1:length(tilt_max_grid)
        % Allow tilt_min <= tilt_max (includes fixed tilt when equal)
        if tilt_min_grid(i) > tilt_max_grid(j)
            continue;  % Skip invalid combinations (tilt_min > tilt_max)
        end
        
        params.tilt_min = tilt_min_grid(i);
        params.tilt_max = tilt_max_grid(j);
        params.acc_min = acc_min;
        params.acc_max = acc_max;
        params.function_type = CONFIG.wagering.function_type;
        params.noise_std = CONFIG.wagering.noise_std;
        
        performance = evaluate_wagering_strategy(...
            CONFIG, accuracy_data, params, struct());
        
        % Store tested combination
        tested_params(end+1, :) = [tilt_min_grid(i), tilt_max_grid(j)];
        tested_rewards(end+1) = performance.cumulative_reward;
        
        if performance.cumulative_reward > best_reward
            best_reward = performance.cumulative_reward;
            best_params = params;
        end
    end
    
    if mod(i, 5) == 0
        fprintf('    Progress: %d/%d\n', i, length(tilt_min_grid));
    end
end

optimal_params = best_params;
optimal_params.tested_params = tested_params;
optimal_params.tested_rewards = tested_rewards;
optimal_params.tilt_min_grid = tilt_min_grid;
optimal_params.tilt_max_grid = tilt_max_grid;

end

function optimal_params = optimize_fmincon(CONFIG, accuracy_data, psychometric_stats)
% Use fmincon for optimization (faster but may get stuck in local minima)

% Compute global accuracy range
all_accuracies = [];
for c = 1:length(accuracy_data)
    all_accuracies = [all_accuracies; accuracy_data{c}];
end
acc_min = min(all_accuracies);
acc_max = max(all_accuracies);

% Objective function: negative cumulative reward (minimize negative = maximize reward)
objective = @(x) -evaluate_wagering_params(CONFIG, accuracy_data, ...
    x(1), x(2), acc_min, acc_max);

% Constraints: tilt_min < tilt_max, both in [0, 1]
A = [1, -1];  % tilt_min - tilt_max < 0
b = 0;
lb = [CONFIG.optimization.tilt_min_range(1), CONFIG.optimization.tilt_max_range(1)];
ub = [CONFIG.optimization.tilt_min_range(2), CONFIG.optimization.tilt_max_range(2)];

    % Initial guess
    x0 = [mean([lb(1), ub(1)]), mean([lb(2), ub(2)])];

options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 50);
x_opt = fmincon(objective, x0, A, b, [], [], lb, ub, [], options);

optimal_params.tilt_min = x_opt(1);
optimal_params.tilt_max = x_opt(2);
optimal_params.acc_min = acc_min;
optimal_params.acc_max = acc_max;
optimal_params.function_type = CONFIG.wagering.function_type;
optimal_params.noise_std = CONFIG.wagering.noise_std;

end

function reward = evaluate_wagering_params(CONFIG, accuracy_data, ...
    tilt_min, tilt_max, acc_min, acc_max)
% Quick evaluation of wagering parameters (returns cumulative reward only)

params.tilt_min = tilt_min;
params.tilt_max = tilt_max;
params.acc_min = acc_min;
params.acc_max = acc_max;
params.function_type = CONFIG.wagering.function_type;
params.noise_std = CONFIG.wagering.noise_std;

performance = evaluate_wagering_strategy(CONFIG, accuracy_data, params, struct());
reward = performance.cumulative_reward;

end

function performance = evaluate_wagering_strategy(CONFIG, accuracy_data, params, psychometric)
% Evaluate a wagering strategy and return performance metrics

n_coherence = length(CONFIG.coherence_levels);
total_reward = 0;
total_hits = 0;
total_trials = 0;

for c = 1:n_coherence
    trial_accuracies = accuracy_data{c};
    n_trials = length(trial_accuracies);
    
    % Compute tilts using wagering function
    trial_tilts = compute_wagering_tilt(trial_accuracies, params);
    
    % Calculate arc widths
    arc_widths = CONFIG.max_arc_deg - ...
        (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts;
    
    % Determine hits
    angular_errors = (1 - trial_accuracies) * 180;
    hits = angular_errors < (arc_widths / 2);
    
    % Calculate rewards
    rewards = hits .* scale_reward(trial_tilts, CONFIG);
    
    total_reward = total_reward + sum(rewards);
    total_hits = total_hits + sum(hits);
    total_trials = total_trials + n_trials;
end

performance.cumulative_reward = total_reward;
performance.hit_rate = total_hits / total_trials;
performance.n_hits = total_hits;
performance.n_trials = total_trials;

end

function tilt = compute_wagering_tilt(accuracy, params)
% Compute tilt from accuracy using wagering function parameters

% Map accuracy from [acc_min, acc_max] to [0, 1]
acc_range = params.acc_max - params.acc_min;
if acc_range > 0
    normalized_acc = (accuracy - params.acc_min) / acc_range;
else
    normalized_acc = ones(size(accuracy)) * 0.5;
end

% Apply function type
switch params.function_type
    case 'linear'
        g_acc = normalized_acc;
    case 'power'
        g_acc = normalized_acc .^ params.power_exp;
    case 'sigmoid'
        g_acc = 1 ./ (1 + exp(-(normalized_acc - params.acc_center) / params.acc_scale));
    otherwise
        error('Unknown function type: %s', params.function_type);
end

% Map to tilt range
tilt = params.tilt_min + (params.tilt_max - params.tilt_min) * g_acc;

% Add noise
noise = randn(size(accuracy)) * params.noise_std;
tilt = tilt + noise;

% Clip to valid range
tilt = max(0, min(1, tilt));

end

function reward_ml = scale_reward(normalized_reward, CONFIG)
% Scale normalized reward [0,1] to actual reward amount in ml

reward_ml = CONFIG.reward_min_ml + ...
    (CONFIG.reward_max_ml - CONFIG.reward_min_ml) * normalized_reward;

end

function fixed_tilt_results = test_fixed_tilt_strategies(CONFIG, accuracy_data)
% Test fixed tilt strategies from 0 to 1 for comparison
% Returns struct with tilt values and their cumulative rewards

% Test fixed tilts from 0 to 1
fixed_tilts = linspace(0, 1, CONFIG.optimization.n_fixed_tilt_points);
n_fixed = length(fixed_tilts);
fixed_rewards = zeros(n_fixed, 1);
fixed_hit_rates = zeros(n_fixed, 1);

fprintf('  Testing %d fixed tilt strategies (0 to 1) for comparison...\n', n_fixed);

% Get accuracy range
all_accuracies = [];
for c = 1:length(accuracy_data)
    all_accuracies = [all_accuracies; accuracy_data{c}];
end
acc_min = min(all_accuracies);
acc_max = max(all_accuracies);

for i = 1:n_fixed
    tilt_val = fixed_tilts(i);
    
    % Create params for fixed tilt (tilt_min = tilt_max = tilt_val)
    params.tilt_min = tilt_val;
    params.tilt_max = tilt_val;
    params.acc_min = acc_min;
    params.acc_max = acc_max;
    params.function_type = 'linear';
    params.noise_std = CONFIG.wagering.noise_std;
    
    % Evaluate fixed tilt strategy
    performance = evaluate_wagering_strategy(CONFIG, accuracy_data, params, struct());
    
    fixed_rewards(i) = performance.cumulative_reward;
    fixed_hit_rates(i) = performance.hit_rate;
end

fixed_tilt_results.tilts = fixed_tilts;
fixed_tilt_results.rewards = fixed_rewards;
fixed_tilt_results.hit_rates = fixed_hit_rates;
[best_fixed_reward, best_fixed_idx] = max(fixed_rewards);
fixed_tilt_results.best_tilt = fixed_tilts(best_fixed_idx);
fixed_tilt_results.best_reward = best_fixed_reward;

fprintf('    Best fixed tilt: %.3f (reward: %.4f ml)\n', ...
    fixed_tilt_results.best_tilt, fixed_tilt_results.best_reward);

end

function create_optimal_wagering_figures(CONFIG, optimal_results)
% Create visualization of optimal wagering strategies

n_results = length(optimal_results);

for p = 1:n_results
    result = optimal_results(p);
    
    fig = figure('Name', sprintf('Optimal Wagering: %s', result.psychometric.name), ...
        'Position', [100 100 1400 1200]);
    sgtitle(sprintf('Optimal Wagering Strategy: %s', result.psychometric.name), ...
        'FontSize', 18, 'FontWeight', 'bold');
    
    % Subplot 1: Optimal wagering function (tilt vs accuracy)
    subplot(3, 3, 1);
    hold on;
    acc_range = linspace(result.optimal_params.acc_min, ...
        result.optimal_params.acc_max, 100);
    
    % Compute theoretical tilt without noise for smooth plot
    acc_range_norm = (acc_range - result.optimal_params.acc_min) / ...
        (result.optimal_params.acc_max - result.optimal_params.acc_min);
    tilt_theoretical = result.optimal_params.tilt_min + ...
        (result.optimal_params.tilt_max - result.optimal_params.tilt_min) * acc_range_norm;
    
    plot(acc_range, tilt_theoretical, 'b-', 'LineWidth', 3);
    xlabel('Accuracy', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Optimal Tilt', 'FontSize', 13, 'FontWeight', 'bold');
    title('Optimal Wagering Function (Theoretical)', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0 1]);
    ylim([0 1]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    hold off;
    
    % Subplot 2: Performance summary
    subplot(3, 3, 2);
    axis off;
    text_x = 0.1;
    text_y = 0.9;
    line_height = 0.1;
    
    text(text_x, text_y, 'OPTIMAL PARAMETERS', 'FontSize', 14, 'FontWeight', 'bold');
    text_y = text_y - line_height * 1.5;
    text(text_x, text_y, sprintf('Tilt min: %.3f', result.optimal_params.tilt_min), ...
        'FontSize', 12);
    text_y = text_y - line_height;
    text(text_x, text_y, sprintf('Tilt max: %.3f', result.optimal_params.tilt_max), ...
        'FontSize', 12);
    text_y = text_y - line_height * 1.5;
    text(text_x, text_y, 'PERFORMANCE', 'FontSize', 14, 'FontWeight', 'bold');
    text_y = text_y - line_height * 1.5;
    text(text_x, text_y, sprintf('Cumulative reward: %.4f ml', ...
        result.performance.cumulative_reward), 'FontSize', 12);
    text_y = text_y - line_height;
    text(text_x, text_y, sprintf('Hit rate: %.3f', result.performance.hit_rate), ...
        'FontSize', 12);
    text_y = text_y - line_height;
    text(text_x, text_y, sprintf('Total hits: %d / %d', ...
        result.performance.n_hits, result.performance.n_trials), 'FontSize', 12);
    
    % Subplot 3: Psychometric function
    subplot(3, 3, 3);
    plot(CONFIG.coherence_levels, result.psychometric_stats.mean_acc, '-o', ...
        'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0 0]);
    hold on;
    errorbar(CONFIG.coherence_levels, result.psychometric_stats.mean_acc, ...
        result.psychometric_stats.std_acc, 'LineStyle', 'none', 'LineWidth', 2);
    xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Mean Accuracy', 'FontSize', 13, 'FontWeight', 'bold');
    title('Psychometric Function', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    ylim([0.35 1.05]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    hold off;
    
    % Subplot 4: Tilt vs Accuracy scatter (from actual trials, colored by coherence)
    subplot(3, 3, 4);
    hold on;
    
    % Create color map for coherences (same as other simulation file)
    n_coherence = length(CONFIG.coherence_levels);
    colors = cool(n_coherence);
    
    for c = 1:n_coherence
        trial_acc = result.psychometric_stats.accuracy_data{c};
        trial_tilt = compute_wagering_tilt(trial_acc, result.optimal_params);
        scatter(trial_acc, trial_tilt, 20, colors(c, :), 'filled', ...
            'MarkerFaceAlpha', 0.4, 'DisplayName', sprintf('%d%%', CONFIG.coherence_levels(c)));
    end
    
    xlabel('Accuracy', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Tilt', 'FontSize', 13, 'FontWeight', 'bold');
    title('Tilt vs Accuracy (Colored by Coherence)', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    xlim([0 1]);
    ylim([0 1]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    hold off;
    
    % Subplot 5: Hit rate by coherence
    subplot(3, 3, 5);
    hit_rates = zeros(length(CONFIG.coherence_levels), 1);
    for c = 1:length(CONFIG.coherence_levels)
        trial_acc = result.psychometric_stats.accuracy_data{c};
        trial_tilt = compute_wagering_tilt(trial_acc, result.optimal_params);
        arc_widths = CONFIG.max_arc_deg - ...
            (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilt;
        angular_errors = (1 - trial_acc) * 180;
        hits = angular_errors < (arc_widths / 2);
        hit_rates(c) = mean(hits);
    end
    plot(CONFIG.coherence_levels, hit_rates, '-o', 'LineWidth', 3, ...
        'MarkerSize', 14, 'Color', [0 0.6 0]);
    xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Hit Rate', 'FontSize', 13, 'FontWeight', 'bold');
    title('Hit Rate vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    ylim([0 1.1]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    
    % Subplot 6: Reward by coherence
    subplot(3, 3, 6);
    rewards = zeros(length(CONFIG.coherence_levels), 1);
    for c = 1:length(CONFIG.coherence_levels)
        trial_acc = result.psychometric_stats.accuracy_data{c};
        trial_tilt = compute_wagering_tilt(trial_acc, result.optimal_params);
        arc_widths = CONFIG.max_arc_deg - ...
            (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilt;
        angular_errors = (1 - trial_acc) * 180;
        hits = angular_errors < (arc_widths / 2);
        rewards(c) = sum(hits .* scale_reward(trial_tilt, CONFIG));
    end
    plot(CONFIG.coherence_levels, rewards, '-o', 'LineWidth', 3, ...
        'MarkerSize', 14, 'Color', [0.8 0.2 0.2]);
    xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Cumulative Reward (ml)', 'FontSize', 13, 'FontWeight', 'bold');
    title('Reward vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    
    % Row 3: Optimality proof plots
    if isfield(result.optimal_params, 'tested_params') && ...
       ~isempty(result.optimal_params.tested_params)
        
        % Subplot 7: 2D intensity map of reward vs tilt_min and tilt_max
        subplot(3, 3, 7);
        hold on;
        
        tilt_min_grid = result.optimal_params.tilt_min_grid;
        tilt_max_grid = result.optimal_params.tilt_max_grid;
        
        % Create reward matrix
        Reward = nan(length(tilt_max_grid), length(tilt_min_grid));
        
        % Fill in reward values
        for k = 1:size(result.optimal_params.tested_params, 1)
            t_min = result.optimal_params.tested_params(k, 1);
            t_max = result.optimal_params.tested_params(k, 2);
            reward = result.optimal_params.tested_rewards(k);
            
            [~, i] = min(abs(tilt_min_grid - t_min));
            [~, j] = min(abs(tilt_max_grid - t_max));
            
            if i <= size(Reward, 2) && j <= size(Reward, 1)
                Reward(j, i) = reward;
            end
        end
        
        % Plot 2D intensity map
        imagesc(tilt_min_grid, tilt_max_grid, Reward);
        colormap(jet);
        colorbar;
        axis xy;  % Flip y-axis so tilt_max increases upward
        hold on;
        
        % Mark optimal point (small black cross)
        plot(result.optimal_params.tilt_min, result.optimal_params.tilt_max, ...
            'k+', 'MarkerSize', 10, 'LineWidth', 2);
        
        % Overlay fixed tilt strategies (diagonal line where tilt_min == tilt_max)
        % Extract fixed tilt rewards
        fixed_tilts = [];
        fixed_rewards = [];
        for k = 1:size(result.optimal_params.tested_params, 1)
            if abs(result.optimal_params.tested_params(k, 1) - result.optimal_params.tested_params(k, 2)) < 0.001
                fixed_tilts(end+1) = result.optimal_params.tested_params(k, 1);
                fixed_rewards(end+1) = result.optimal_params.tested_rewards(k);
            end
        end
        if ~isempty(fixed_tilts)
            [fixed_sorted, idx] = sort(fixed_tilts);
            fixed_rewards_sorted = fixed_rewards(idx);
            plot(fixed_sorted, fixed_sorted, 'w-', 'LineWidth', 2, 'LineStyle', '--');
            [best_fixed_reward, best_fixed_idx] = max(fixed_rewards_sorted);
            best_fixed_tilt = fixed_sorted(best_fixed_idx);
            plot(best_fixed_tilt, best_fixed_tilt, 'mo', 'MarkerSize', 12, ...
                'MarkerFaceColor', 'm', 'LineWidth', 2);
        end
        
        xlabel('Tilt Min', 'FontSize', 13, 'FontWeight', 'bold');
        ylabel('Tilt Max', 'FontSize', 13, 'FontWeight', 'bold');
        title('Reward Intensity Map (Optimal Marked)', 'FontSize', 14, 'FontWeight', 'bold');
        set(gca, 'FontSize', 12, 'LineWidth', 1.5);
        box on;
        hold off;
        
        % Subplot 8: Cumulative reward vs fixed tilt (0 to 1)
        subplot(3, 3, 8);
        hold on;
        
        if isfield(result, 'fixed_tilt_results')
            fixed_tilts = result.fixed_tilt_results.tilts;
            fixed_rewards = result.fixed_tilt_results.rewards;
            plot(fixed_tilts, fixed_rewards, '-', 'LineWidth', 2, ...
                'Color', [0.3 0.3 0.8], 'DisplayName', 'Fixed Tilt');
            
            % Mark best fixed tilt
            [best_fixed_reward, best_fixed_idx] = max(fixed_rewards);
            best_fixed_tilt = fixed_tilts(best_fixed_idx);
            plot(best_fixed_tilt, best_fixed_reward, 'mo', 'MarkerSize', 10, ...
                'MarkerFaceColor', 'm', 'LineWidth', 2, 'DisplayName', 'Best Fixed');
            
            % Mark optimal linear mapping
            xlim_current = xlim;
            plot(xlim_current, [result.performance.cumulative_reward, result.performance.cumulative_reward], ...
                'r--', 'LineWidth', 2, 'DisplayName', ...
                sprintf('Optimal Linear: %.4f ml', result.performance.cumulative_reward));
        end
        
        xlabel('Fixed Tilt Value', 'FontSize', 13, 'FontWeight', 'bold');
        ylabel('Cumulative Reward (ml)', 'FontSize', 13, 'FontWeight', 'bold');
        title('Reward vs Fixed Tilt (0 to 1)', 'FontSize', 14, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 10);
        grid on;
        set(gca, 'FontSize', 12, 'LineWidth', 1.5);
        box on;
        hold off;
        
        % Subplot 9: Reward vs tilt_max (for optimal tilt_min)
        subplot(3, 3, 9);
        hold on;
        optimal_t_min = result.optimal_params.tilt_min;
        
        % Extract rewards for optimal tilt_min
        rewards_at_optimal_tmin = [];
        tilt_max_at_optimal_tmin = [];
        for k = 1:size(result.optimal_params.tested_params, 1)
            if abs(result.optimal_params.tested_params(k, 1) - optimal_t_min) < 0.01
                tilt_max_at_optimal_tmin(end+1) = result.optimal_params.tested_params(k, 2);
                rewards_at_optimal_tmin(end+1) = result.optimal_params.tested_rewards(k);
            end
        end
        
        if ~isempty(tilt_max_at_optimal_tmin)
            [tilt_max_sorted, idx] = sort(tilt_max_at_optimal_tmin);
            rewards_sorted = rewards_at_optimal_tmin(idx);
            plot(tilt_max_sorted, rewards_sorted, '-o', 'LineWidth', 2, ...
                'MarkerSize', 8, 'Color', [0.3 0.3 0.8]);
            plot(result.optimal_params.tilt_max, result.performance.cumulative_reward, ...
                'k+', 'MarkerSize', 10, 'LineWidth', 2);
        end
        xlabel('Tilt Max', 'FontSize', 13, 'FontWeight', 'bold');
        ylabel('Cumulative Reward (ml)', 'FontSize', 13, 'FontWeight', 'bold');
        title(sprintf('Reward vs Tilt Max (Tilt Min=%.3f)', optimal_t_min), ...
            'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        set(gca, 'FontSize', 12, 'LineWidth', 1.5);
        box on;
        hold off;
    else
        % If grid search data not available, show message
        subplot(3, 3, 7);
        axis off;
        text(0.5, 0.5, 'Grid search data not available', ...
            'HorizontalAlignment', 'center', 'FontSize', 12);
        
        subplot(3, 3, 8);
        axis off;
        text(0.5, 0.5, 'Grid search data not available', ...
            'HorizontalAlignment', 'center', 'FontSize', 12);
        
        subplot(3, 3, 9);
        axis off;
        text(0.5, 0.5, 'Grid search data not available', ...
            'HorizontalAlignment', 'center', 'FontSize', 12);
    end
end

end

