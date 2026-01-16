function scpsim_cpr_optimal_wagering_strategy_cursor
% SCPSIM_CPR_OPTIMAL_WAGERING_STRATEGY_CURSOR - Find absolutely optimal wagering strategy
%
% Comprehensive search over all possible wagering strategies to maximize cumulative reward.
% Searches over:
%   - Coherence-dependent baseline tilts
%   - Accuracy scaling parameters
%   - Hit/miss prediction strategies
%   - Any combination of the above
%
% OUTPUTS:
%   - Optimal wagering strategy (potentially complex, multi-parameter)
%   - Performance metrics (cumulative reward, hit rate, etc.)
%   - Visualization of optimal strategy and comparison with simpler strategies
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
PSYCHOMETRIC_FUNCTIONS = struct();

% Current psychometric function
PSYCHOMETRIC_FUNCTIONS(1).baseline_acc = 0.5;
PSYCHOMETRIC_FUNCTIONS(1).max_acc = 0.85;
PSYCHOMETRIC_FUNCTIONS(1).slope = 0.08;
PSYCHOMETRIC_FUNCTIONS(1).threshold = 50;
PSYCHOMETRIC_FUNCTIONS(1).variability = 0.1;
PSYCHOMETRIC_FUNCTIONS(1).name = 'Current';

% --- Strategy Parameter Ranges for Optimization ---
% Strategy 1: Coherence-dependent baseline only (no accuracy scaling)
% tilt_min_coherence = tilt value at 0% coherence
% tilt_max_coherence = tilt value at 100% coherence
% Tilt at any coherence: tilt = tilt_min_coherence + (tilt_max_coherence - tilt_min_coherence) * (coherence / 100)
CONFIG.strategy1.tilt_min_coherence_range = [0, 0.3];  % Tilt at 0% coherence
CONFIG.strategy1.tilt_max_coherence_range = [0.3, 0.9];  % Tilt at 100% coherence
CONFIG.strategy1.noise_std = 0.1;

% Strategy 2: Coherence-dependent baseline + accuracy scaling
CONFIG.strategy2.tilt_min_coherence_range = [0, 0.3];
CONFIG.strategy2.tilt_max_coherence_range = [0.3, 0.9];
CONFIG.strategy2.accuracy_scaling_range = [0, 1];  % 0 = no scaling, 1 = full scaling
CONFIG.strategy2.tilt_range_for_scaling = [0.01, 0.9];  % Range for accuracy scaling
CONFIG.strategy2.noise_std = 0.1;

% Strategy 3: Hit/miss prediction aware (uses predicted hit probability)
CONFIG.strategy3.tilt_min_coherence_range = [0, 0.3];
CONFIG.strategy3.tilt_max_coherence_range = [0.3, 0.9];
CONFIG.strategy3.hit_prediction_weight = [0, 1];  % Weight for hit prediction in tilt calculation
CONFIG.strategy3.noise_std = 0.1;

% Strategy 4: Full combination (coherence + accuracy + hit prediction)
CONFIG.strategy4.tilt_min_coherence_range = [0, 0.3];
CONFIG.strategy4.tilt_max_coherence_range = [0.3, 0.9];
CONFIG.strategy4.accuracy_scaling_range = [0, 1];
CONFIG.strategy4.tilt_range_for_scaling = [0.01, 0.9];
CONFIG.strategy4.hit_prediction_weight = [0, 1];
CONFIG.strategy4.noise_std = 0.1;

% --- Optimization Parameters ---
CONFIG.optimization.method = 'comprehensive_search';  % Comprehensive multi-strategy search
CONFIG.optimization.n_grid_points = 15;  % Grid points per parameter (reduced for speed)
CONFIG.optimization.strategies_to_test = [1, 2, 3, 4];  % Which strategy types to test

% --- Output Options ---
CONFIG.show_figures = true;
CONFIG.save_results = false;
CONFIG.results_filename = 'optimal_wagering_strategy_results.mat';

%% ========================================================================
%  MAIN OPTIMIZATION
%  ========================================================================

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  Comprehensive Optimal Wagering Strategy Finder                ║\n');
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
    
    % Find optimal wagering strategy across all strategy types
    optimal_strategy = find_optimal_wagering_strategy(CONFIG, accuracy_data, psychometric_stats);
    
    % Evaluate optimal strategy
    optimal_performance = evaluate_wagering_strategy_comprehensive(...
        CONFIG, accuracy_data, optimal_strategy, psychometric_stats);
    
    % Also test fixed tilt strategies for comparison
    fixed_tilt_results = test_fixed_tilt_strategies(CONFIG, accuracy_data);
    
    % Store results
    optimal_results(p).psychometric = psychometric;
    optimal_results(p).optimal_strategy = optimal_strategy;
    optimal_results(p).performance = optimal_performance;
    optimal_results(p).psychometric_stats = psychometric_stats;
    optimal_results(p).psychometric_stats.accuracy_data = accuracy_data;
    optimal_results(p).fixed_tilt_results = fixed_tilt_results;
    
    % Also evaluate best strategy from each type for comparison
    all_strategy_performances = struct();
    if isfield(optimal_strategy, 'all_strategies')
        % Evaluate best from each strategy type
        for st = CONFIG.optimization.strategies_to_test
            best_from_type = get_best_strategy_from_type(optimal_strategy.all_strategies, st, ...
                CONFIG, accuracy_data, psychometric_stats);
            if ~isempty(best_from_type)
                all_strategy_performances.(sprintf('strategy%d', st)) = best_from_type;
            end
        end
    end
    optimal_results(p).all_strategy_performances = all_strategy_performances;
    
    fprintf('  Optimal strategy type: %d\n', optimal_strategy.strategy_type);
    fprintf('  Cumulative reward: %.4f ml\n', optimal_performance.cumulative_reward);
    fprintf('  Hit rate: %.3f\n', optimal_performance.hit_rate);
    fprintf('  Best fixed tilt reward: %.4f ml\n', fixed_tilt_results.best_reward);
    fprintf('  Improvement over fixed: %.2f%%\n', ...
        ((optimal_performance.cumulative_reward - fixed_tilt_results.best_reward) / ...
        fixed_tilt_results.best_reward) * 100);
    fprintf('\n');
end

%% ========================================================================
%  VISUALIZATION AND SAVING
%  ========================================================================

if CONFIG.show_figures
    create_optimal_strategy_figures(CONFIG, optimal_results);
end

if CONFIG.save_results
    save(CONFIG.results_filename, 'optimal_results', 'CONFIG');
    fprintf('Results saved to: %s\n', CONFIG.results_filename);
end

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  Optimization Complete!                                       ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

end

%% ========================================================================
%  CORE FUNCTIONS
%  ========================================================================

function [accuracy_data, stats] = generate_psychometric_data_batch(CONFIG, psychometric)
% Generate accuracy distributions for each coherence level

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

function optimal_strategy = find_optimal_wagering_strategy(CONFIG, accuracy_data, psychometric_stats)
% Comprehensive search over all strategy types
% Returns optimal strategy and stores all tested strategies for comparison

fprintf('  Starting comprehensive strategy search...\n');

% Compute global accuracy range
all_accuracies = [];
for c = 1:length(accuracy_data)
    all_accuracies = [all_accuracies; accuracy_data{c}];
end
acc_min = min(all_accuracies);
acc_max = max(all_accuracies);

best_reward = -inf;
best_strategy = struct();
all_strategies = struct();  % Store all tested strategies

% Test each strategy type
for strategy_type = CONFIG.optimization.strategies_to_test
    fprintf('    Testing strategy type %d...\n', strategy_type);
    
    switch strategy_type
        case 1
            % Strategy 1: Coherence-dependent baseline only
            [strategy_params, all_params_1] = optimize_strategy1(CONFIG, accuracy_data, acc_min, acc_max);
            all_strategies.strategy1 = all_params_1;
        case 2
            % Strategy 2: Coherence-dependent + accuracy scaling
            [strategy_params, all_params_2] = optimize_strategy2(CONFIG, accuracy_data, acc_min, acc_max);
            all_strategies.strategy2 = all_params_2;
        case 3
            % Strategy 3: Coherence-dependent + hit prediction
            [strategy_params, all_params_3] = optimize_strategy3(CONFIG, accuracy_data, acc_min, acc_max);
            all_strategies.strategy3 = all_params_3;
        case 4
            % Strategy 4: Full combination
            [strategy_params, all_params_4] = optimize_strategy4(CONFIG, accuracy_data, acc_min, acc_max);
            all_strategies.strategy4 = all_params_4;
        otherwise
            continue;
    end
    
    strategy_params.strategy_type = strategy_type;
    strategy_params.acc_min = acc_min;
    strategy_params.acc_max = acc_max;
    
    performance = evaluate_wagering_strategy_comprehensive(...
        CONFIG, accuracy_data, strategy_params, psychometric_stats);
    
    if performance.cumulative_reward > best_reward
        best_reward = performance.cumulative_reward;
        best_strategy = strategy_params;
    end
end

optimal_strategy = best_strategy;
optimal_strategy.all_strategies = all_strategies;  % Store all for visualization
fprintf('  Best strategy found: Type %d, Reward: %.4f ml\n', ...
    optimal_strategy.strategy_type, best_reward);

end

function [best_params, all_params] = optimize_strategy1(CONFIG, accuracy_data, acc_min, acc_max)
% Optimize Strategy 1: Coherence-dependent baseline only
% tilt_min_coherence = tilt at 0% coherence
% tilt_max_coherence = tilt at 100% coherence
% Tilt at any coherence: tilt = tilt_min_coherence + (tilt_max_coherence - tilt_min_coherence) * (coherence / 100)

tilt_min_grid = linspace(CONFIG.strategy1.tilt_min_coherence_range(1), ...
    CONFIG.strategy1.tilt_min_coherence_range(2), CONFIG.optimization.n_grid_points);
tilt_max_grid = linspace(CONFIG.strategy1.tilt_max_coherence_range(1), ...
    CONFIG.strategy1.tilt_max_coherence_range(2), CONFIG.optimization.n_grid_points);

best_reward = -inf;
best_params = struct();
all_params.params = [];
all_params.rewards = [];

for i = 1:length(tilt_min_grid)
    for j = 1:length(tilt_max_grid)
        if tilt_min_grid(i) > tilt_max_grid(j)
            continue;
        end
        
        params.tilt_min_coherence = tilt_min_grid(i);
        params.tilt_max_coherence = tilt_max_grid(j);
        params.noise_std = CONFIG.strategy1.noise_std;
        
        performance = evaluate_strategy1(CONFIG, accuracy_data, params);
        
        % Store all tested parameters
        all_params.params(end+1, :) = [tilt_min_grid(i), tilt_max_grid(j)];
        all_params.rewards(end+1) = performance.cumulative_reward;
        
        if performance.cumulative_reward > best_reward
            best_reward = performance.cumulative_reward;
            best_params = params;
        end
    end
end

end

function [best_params, all_params] = optimize_strategy2(CONFIG, accuracy_data, acc_min, acc_max)
% Optimize Strategy 2: Coherence-dependent + accuracy scaling
% tilt_min_coherence = tilt at 0% coherence
% tilt_max_coherence = tilt at 100% coherence
% accuracy_scaling = weight for accuracy-based modulation (0 = no scaling, 1 = full scaling)

tilt_min_grid = linspace(CONFIG.strategy2.tilt_min_coherence_range(1), ...
    CONFIG.strategy2.tilt_min_coherence_range(2), CONFIG.optimization.n_grid_points);
tilt_max_grid = linspace(CONFIG.strategy2.tilt_max_coherence_range(1), ...
    CONFIG.strategy2.tilt_max_coherence_range(2), CONFIG.optimization.n_grid_points);
acc_scaling_grid = linspace(CONFIG.strategy2.accuracy_scaling_range(1), ...
    CONFIG.strategy2.accuracy_scaling_range(2), CONFIG.optimization.n_grid_points);

best_reward = -inf;
best_params = struct();
all_params.params = [];
all_params.rewards = [];

for i = 1:length(tilt_min_grid)
    for j = 1:length(tilt_max_grid)
        if tilt_min_grid(i) > tilt_max_grid(j)
            continue;
        end
        
        for k = 1:length(acc_scaling_grid)
            params.tilt_min_coherence = tilt_min_grid(i);
            params.tilt_max_coherence = tilt_max_grid(j);
            params.accuracy_scaling = acc_scaling_grid(k);
            params.tilt_min_scaling = CONFIG.strategy2.tilt_range_for_scaling(1);
            params.tilt_max_scaling = CONFIG.strategy2.tilt_range_for_scaling(2);
            params.noise_std = CONFIG.strategy2.noise_std;
            
            performance = evaluate_strategy2(CONFIG, accuracy_data, params);
            
            % Store all tested parameters (sample to avoid memory issues)
            if rand < 0.1  % Store 10% of evaluations
                all_params.params(end+1, :) = [tilt_min_grid(i), tilt_max_grid(j), acc_scaling_grid(k)];
                all_params.rewards(end+1) = performance.cumulative_reward;
            end
            
            if performance.cumulative_reward > best_reward
                best_reward = performance.cumulative_reward;
                best_params = params;
            end
        end
    end
end

end

function [best_params, all_params] = optimize_strategy3(CONFIG, accuracy_data, acc_min, acc_max)
% Optimize Strategy 3: Coherence-dependent + hit prediction
% tilt_min_coherence = tilt at 0% coherence
% tilt_max_coherence = tilt at 100% coherence
% hit_prediction_weight = weight for hit prediction (0 = coherence only, 1 = hit prediction only)

tilt_min_grid = linspace(CONFIG.strategy3.tilt_min_coherence_range(1), ...
    CONFIG.strategy3.tilt_min_coherence_range(2), CONFIG.optimization.n_grid_points);
tilt_max_grid = linspace(CONFIG.strategy3.tilt_max_coherence_range(1), ...
    CONFIG.strategy3.tilt_max_coherence_range(2), CONFIG.optimization.n_grid_points);
hit_weight_grid = linspace(CONFIG.strategy3.hit_prediction_weight(1), ...
    CONFIG.strategy3.hit_prediction_weight(2), CONFIG.optimization.n_grid_points);

best_reward = -inf;
best_params = struct();
all_params.params = [];
all_params.rewards = [];

for i = 1:length(tilt_min_grid)
    for j = 1:length(tilt_max_grid)
        if tilt_min_grid(i) > tilt_max_grid(j)
            continue;
        end
        
        for k = 1:length(hit_weight_grid)
            params.tilt_min_coherence = tilt_min_grid(i);
            params.tilt_max_coherence = tilt_max_grid(j);
            params.hit_prediction_weight = hit_weight_grid(k);
            params.noise_std = CONFIG.strategy3.noise_std;
            
            performance = evaluate_strategy3(CONFIG, accuracy_data, params);
            
            % Store all tested parameters (sample to avoid memory issues)
            if rand < 0.1  % Store 10% of evaluations
                all_params.params(end+1, :) = [tilt_min_grid(i), tilt_max_grid(j), hit_weight_grid(k)];
                all_params.rewards(end+1) = performance.cumulative_reward;
            end
            
            if performance.cumulative_reward > best_reward
                best_reward = performance.cumulative_reward;
                best_params = params;
            end
        end
    end
end

end

function [best_params, all_params] = optimize_strategy4(CONFIG, accuracy_data, acc_min, acc_max)
% Optimize Strategy 4: Full combination (coherence + accuracy + hit prediction)
% This is expensive, so use coarser grid
% tilt_min_coherence = tilt at 0% coherence
% tilt_max_coherence = tilt at 100% coherence
% accuracy_scaling = weight for accuracy-based modulation
% hit_prediction_weight = weight for hit prediction

n_points = max(8, round(CONFIG.optimization.n_grid_points / 2));  % Coarser grid

tilt_min_grid = linspace(CONFIG.strategy4.tilt_min_coherence_range(1), ...
    CONFIG.strategy4.tilt_min_coherence_range(2), n_points);
tilt_max_grid = linspace(CONFIG.strategy4.tilt_max_coherence_range(1), ...
    CONFIG.strategy4.tilt_max_coherence_range(2), n_points);
acc_scaling_grid = linspace(CONFIG.strategy4.accuracy_scaling_range(1), ...
    CONFIG.strategy4.accuracy_scaling_range(2), n_points);
hit_weight_grid = linspace(CONFIG.strategy4.hit_prediction_weight(1), ...
    CONFIG.strategy4.hit_prediction_weight(2), n_points);

best_reward = -inf;
best_params = struct();
all_params.params = [];
all_params.rewards = [];
total_evals = length(tilt_min_grid) * length(tilt_max_grid) * ...
    length(acc_scaling_grid) * length(hit_weight_grid);
eval_count = 0;

for i = 1:length(tilt_min_grid)
    for j = 1:length(tilt_max_grid)
        if tilt_min_grid(i) > tilt_max_grid(j)
            continue;
        end
        
        for k = 1:length(acc_scaling_grid)
            for l = 1:length(hit_weight_grid)
                params.tilt_min_coherence = tilt_min_grid(i);
                params.tilt_max_coherence = tilt_max_grid(j);
                params.accuracy_scaling = acc_scaling_grid(k);
                params.tilt_min_scaling = CONFIG.strategy4.tilt_range_for_scaling(1);
                params.tilt_max_scaling = CONFIG.strategy4.tilt_range_for_scaling(2);
                params.hit_prediction_weight = hit_weight_grid(l);
                params.noise_std = CONFIG.strategy4.noise_std;
                
                performance = evaluate_strategy4(CONFIG, accuracy_data, params);
                
                % Store all tested parameters (sample to avoid memory issues)
                if rand < 0.05  % Store 5% of evaluations
                    all_params.params(end+1, :) = [tilt_min_grid(i), tilt_max_grid(j), ...
                        acc_scaling_grid(k), hit_weight_grid(l)];
                    all_params.rewards(end+1) = performance.cumulative_reward;
                end
                
                if performance.cumulative_reward > best_reward
                    best_reward = performance.cumulative_reward;
                    best_params = params;
                end
                
                eval_count = eval_count + 1;
                if mod(eval_count, 100) == 0
                    fprintf('      Progress: %d/%d (%.1f%%)\n', ...
                        eval_count, total_evals, 100*eval_count/total_evals);
                end
            end
        end
    end
end

end

function performance = evaluate_strategy1(CONFIG, accuracy_data, params)
% Evaluate Strategy 1: Coherence-dependent baseline only

n_coherence = length(CONFIG.coherence_levels);
total_reward = 0;
total_hits = 0;
total_trials = 0;

for c = 1:n_coherence
    coh = CONFIG.coherence_levels(c);
    trial_accuracies = accuracy_data{c};
    n_trials = length(trial_accuracies);
    
    % Coherence-dependent baseline tilt
    tilt_base = params.tilt_min_coherence + ...
        (params.tilt_max_coherence - params.tilt_min_coherence) * (coh / 100);
    
    % Add noise
    trial_tilts = tilt_base + randn(n_trials, 1) * params.noise_std;
    trial_tilts = max(0, min(1, trial_tilts));
    
    % Calculate rewards
    arc_widths = CONFIG.max_arc_deg - ...
        (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts;
    angular_errors = (1 - trial_accuracies) * 180;
    hits = angular_errors < (arc_widths / 2);
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

function performance = evaluate_strategy2(CONFIG, accuracy_data, params)
% Evaluate Strategy 2: Coherence-dependent + accuracy scaling

n_coherence = length(CONFIG.coherence_levels);
total_reward = 0;
total_hits = 0;
total_trials = 0;

% Get global accuracy range
all_accuracies = [];
for c = 1:length(accuracy_data)
    all_accuracies = [all_accuracies; accuracy_data{c}];
end
acc_min = min(all_accuracies);
acc_max = max(all_accuracies);

for c = 1:n_coherence
    coh = CONFIG.coherence_levels(c);
    trial_accuracies = accuracy_data{c};
    n_trials = length(trial_accuracies);
    
    % Coherence-dependent baseline tilt
    tilt_base = params.tilt_min_coherence + ...
        (params.tilt_max_coherence - params.tilt_min_coherence) * (coh / 100);
    
    % Accuracy scaling component
    if params.accuracy_scaling > 0 && (acc_max - acc_min) > 0
        normalized_acc = (trial_accuracies - acc_min) / (acc_max - acc_min);
        tilt_from_acc = params.tilt_min_scaling + ...
            (params.tilt_max_scaling - params.tilt_min_scaling) * normalized_acc;
        tilt_modulation = (tilt_from_acc - mean(tilt_from_acc)) * params.accuracy_scaling;
        trial_tilts = tilt_base + tilt_modulation;
    else
        trial_tilts = ones(n_trials, 1) * tilt_base;
    end
    
    % Add noise
    trial_tilts = trial_tilts + randn(n_trials, 1) * params.noise_std;
    trial_tilts = max(0, min(1, trial_tilts));
    
    % Calculate rewards
    arc_widths = CONFIG.max_arc_deg - ...
        (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts;
    angular_errors = (1 - trial_accuracies) * 180;
    hits = angular_errors < (arc_widths / 2);
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

function performance = evaluate_strategy3(CONFIG, accuracy_data, params)
% Evaluate Strategy 3: Coherence-dependent + hit prediction

n_coherence = length(CONFIG.coherence_levels);
total_reward = 0;
total_hits = 0;
total_trials = 0;

for c = 1:n_coherence
    coh = CONFIG.coherence_levels(c);
    trial_accuracies = accuracy_data{c};
    n_trials = length(trial_accuracies);
    
    % Coherence-dependent baseline tilt
    tilt_base = params.tilt_min_coherence + ...
        (params.tilt_max_coherence - params.tilt_min_coherence) * (coh / 100);
    
    % Predict hit probability for each trial
    % For a given accuracy, what's the optimal tilt that maximizes expected reward?
    % Expected reward = P(hit) * reward(tilt)
    % P(hit) = P(angular_error < arc_width/2)
    % We can estimate this by trying different tilts and seeing which gives best expected reward
    
    trial_tilts = zeros(n_trials, 1);
    for t = 1:n_trials
        acc = trial_accuracies(t);
        angular_error = (1 - acc) * 180;
        
        % For each possible tilt, calculate expected reward
        test_tilts = linspace(0, 1, 50);
        expected_rewards = zeros(size(test_tilts));
        
        for ti = 1:length(test_tilts)
            test_tilt = test_tilts(ti);
            arc_width = CONFIG.max_arc_deg - ...
                (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * test_tilt;
            p_hit = (angular_error < (arc_width / 2));
            reward_if_hit = scale_reward(test_tilt, CONFIG);
            expected_rewards(ti) = p_hit * reward_if_hit;
        end
        
        % Find tilt that maximizes expected reward
        [~, best_idx] = max(expected_rewards);
        optimal_tilt = test_tilts(best_idx);
        
        % Blend with baseline based on hit_prediction_weight
        trial_tilts(t) = (1 - params.hit_prediction_weight) * tilt_base + ...
            params.hit_prediction_weight * optimal_tilt;
    end
    
    % Add noise
    trial_tilts = trial_tilts + randn(n_trials, 1) * params.noise_std;
    trial_tilts = max(0, min(1, trial_tilts));
    
    % Calculate rewards
    arc_widths = CONFIG.max_arc_deg - ...
        (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts;
    angular_errors = (1 - trial_accuracies) * 180;
    hits = angular_errors < (arc_widths / 2);
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

function performance = evaluate_strategy4(CONFIG, accuracy_data, params)
% Evaluate Strategy 4: Full combination (coherence + accuracy + hit prediction)

n_coherence = length(CONFIG.coherence_levels);
total_reward = 0;
total_hits = 0;
total_trials = 0;

% Get global accuracy range
all_accuracies = [];
for c = 1:length(accuracy_data)
    all_accuracies = [all_accuracies; accuracy_data{c}];
end
acc_min = min(all_accuracies);
acc_max = max(all_accuracies);

for c = 1:n_coherence
    coh = CONFIG.coherence_levels(c);
    trial_accuracies = accuracy_data{c};
    n_trials = length(trial_accuracies);
    
    % Coherence-dependent baseline tilt
    tilt_base = params.tilt_min_coherence + ...
        (params.tilt_max_coherence - params.tilt_min_coherence) * (coh / 100);
    
    % Accuracy scaling component
    if params.accuracy_scaling > 0 && (acc_max - acc_min) > 0
        normalized_acc = (trial_accuracies - acc_min) / (acc_max - acc_min);
        tilt_from_acc = params.tilt_min_scaling + ...
            (params.tilt_max_scaling - params.tilt_min_scaling) * normalized_acc;
        tilt_modulation = (tilt_from_acc - mean(tilt_from_acc)) * params.accuracy_scaling;
        tilt_with_acc = tilt_base + tilt_modulation;
    else
        tilt_with_acc = ones(n_trials, 1) * tilt_base;
    end
    
    % Hit prediction component
    trial_tilts = zeros(n_trials, 1);
    for t = 1:n_trials
        acc = trial_accuracies(t);
        angular_error = (1 - acc) * 180;
        
        % Find optimal tilt for this accuracy
        test_tilts = linspace(0, 1, 30);  % Coarser for speed
        expected_rewards = zeros(size(test_tilts));
        
        for ti = 1:length(test_tilts)
            test_tilt = test_tilts(ti);
            arc_width = CONFIG.max_arc_deg - ...
                (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * test_tilt;
            p_hit = (angular_error < (arc_width / 2));
            reward_if_hit = scale_reward(test_tilt, CONFIG);
            expected_rewards(ti) = p_hit * reward_if_hit;
        end
        
        [~, best_idx] = max(expected_rewards);
        optimal_tilt = test_tilts(best_idx);
        
        % Blend all three components
        trial_tilts(t) = (1 - params.hit_prediction_weight) * tilt_with_acc(t) + ...
            params.hit_prediction_weight * optimal_tilt;
    end
    
    % Add noise
    trial_tilts = trial_tilts + randn(n_trials, 1) * params.noise_std;
    trial_tilts = max(0, min(1, trial_tilts));
    
    % Calculate rewards
    arc_widths = CONFIG.max_arc_deg - ...
        (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts;
    angular_errors = (1 - trial_accuracies) * 180;
    hits = angular_errors < (arc_widths / 2);
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

function performance = evaluate_wagering_strategy_comprehensive(CONFIG, accuracy_data, strategy_params, psychometric_stats)
% Evaluate a comprehensive wagering strategy

switch strategy_params.strategy_type
    case 1
        performance = evaluate_strategy1(CONFIG, accuracy_data, strategy_params);
    case 2
        performance = evaluate_strategy2(CONFIG, accuracy_data, strategy_params);
    case 3
        performance = evaluate_strategy3(CONFIG, accuracy_data, strategy_params);
    case 4
        performance = evaluate_strategy4(CONFIG, accuracy_data, strategy_params);
    otherwise
        error('Unknown strategy type: %d', strategy_params.strategy_type);
end

end

function fixed_tilt_results = test_fixed_tilt_strategies(CONFIG, accuracy_data)
% Test fixed tilt strategies from 0 to 1 for comparison

fixed_tilts = linspace(0, 1, 50);
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
    
    % Create params for fixed tilt
    params.tilt_min_coherence = tilt_val;
    params.tilt_max_coherence = tilt_val;
    params.noise_std = 0.1;
    
    performance = evaluate_strategy1(CONFIG, accuracy_data, params);
    
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

function reward_ml = scale_reward(normalized_reward, CONFIG)
% Scale normalized reward [0,1] to actual reward amount in ml

reward_ml = CONFIG.reward_min_ml + ...
    (CONFIG.reward_max_ml - CONFIG.reward_min_ml) * normalized_reward;

end

function create_optimal_strategy_figures(CONFIG, optimal_results)
% Create visualization of optimal wagering strategies

% Strategy colors and names (used across multiple subplots)
strategy_colors = {[0 0.4 0.8], [0.8 0.2 0.2], [0 0.6 0], [0.9 0.6 0]};
strategy_names = {'Strategy 1: Coherence', 'Strategy 2: Coherence+Acc', ...
    'Strategy 3: Coherence+Hit', 'Strategy 4: Full'};

n_results = length(optimal_results);

for p = 1:n_results
    result = optimal_results(p);
    
    fig = figure('Name', sprintf('Optimal Strategy: %s', result.psychometric.name), ...
        'Position', [100 100 1600 1000]);
    sgtitle(sprintf('Comprehensive Optimal Wagering Strategy: %s', result.psychometric.name), ...
        'FontSize', 18, 'FontWeight', 'bold');
    
    % Subplot 1: Strategy parameters summary
    subplot(2, 3, 1);
    axis off;
    text_x = 0.1;
    text_y = 0.9;
    line_height = 0.08;
    
    text(text_x, text_y, 'OPTIMAL STRATEGY', 'FontSize', 14, 'FontWeight', 'bold');
    text_y = text_y - line_height * 1.5;
    text(text_x, text_y, sprintf('Strategy Type: %d', result.optimal_strategy.strategy_type), ...
        'FontSize', 12);
    text_y = text_y - line_height;
    
    switch result.optimal_strategy.strategy_type
        case 1
            text(text_x, text_y, 'Coherence-dependent baseline only', 'FontSize', 11);
            text_y = text_y - line_height;
            text(text_x, text_y, sprintf('Tilt min (0%%): %.3f', ...
                result.optimal_strategy.tilt_min_coherence), 'FontSize', 11);
            text_y = text_y - line_height;
            text(text_x, text_y, sprintf('Tilt max (100%%): %.3f', ...
                result.optimal_strategy.tilt_max_coherence), 'FontSize', 11);
        case 2
            text(text_x, text_y, 'Coherence + Accuracy scaling', 'FontSize', 11);
            text_y = text_y - line_height;
            text(text_x, text_y, sprintf('Tilt min/max coherence: %.3f / %.3f', ...
                result.optimal_strategy.tilt_min_coherence, ...
                result.optimal_strategy.tilt_max_coherence), 'FontSize', 11);
            text_y = text_y - line_height;
            text(text_x, text_y, sprintf('Accuracy scaling: %.3f', ...
                result.optimal_strategy.accuracy_scaling), 'FontSize', 11);
        case 3
            text(text_x, text_y, 'Coherence + Hit prediction', 'FontSize', 11);
            text_y = text_y - line_height;
            text(text_x, text_y, sprintf('Tilt min/max coherence: %.3f / %.3f', ...
                result.optimal_strategy.tilt_min_coherence, ...
                result.optimal_strategy.tilt_max_coherence), 'FontSize', 11);
            text_y = text_y - line_height;
            text(text_x, text_y, sprintf('Hit prediction weight: %.3f', ...
                result.optimal_strategy.hit_prediction_weight), 'FontSize', 11);
        case 4
            text(text_x, text_y, 'Full combination', 'FontSize', 11);
            text_y = text_y - line_height;
            text(text_x, text_y, sprintf('Tilt min/max coherence: %.3f / %.3f', ...
                result.optimal_strategy.tilt_min_coherence, ...
                result.optimal_strategy.tilt_max_coherence), 'FontSize', 11);
            text_y = text_y - line_height;
            text(text_x, text_y, sprintf('Accuracy scaling: %.3f', ...
                result.optimal_strategy.accuracy_scaling), 'FontSize', 11);
            text_y = text_y - line_height;
            text(text_x, text_y, sprintf('Hit prediction weight: %.3f', ...
                result.optimal_strategy.hit_prediction_weight), 'FontSize', 11);
    end
    
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
    text_y = text_y - line_height;
    text(text_x, text_y, sprintf('Best fixed tilt: %.4f ml', ...
        result.fixed_tilt_results.best_reward), 'FontSize', 12);
    text_y = text_y - line_height;
    improvement = ((result.performance.cumulative_reward - result.fixed_tilt_results.best_reward) / ...
        result.fixed_tilt_results.best_reward) * 100;
    text(text_x, text_y, sprintf('Improvement: %.2f%%', improvement), ...
        'FontSize', 12, 'FontWeight', 'bold');
    
    % Subplot 2: Psychometric function
    subplot(2, 3, 2);
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
    
    % Subplot 3: Tilt vs Accuracy scatter (colored by coherence)
    subplot(2, 3, 3);
    hold on;
    n_coherence = length(CONFIG.coherence_levels);
    colors = cool(n_coherence);
    
    for c = 1:n_coherence
        trial_acc = result.psychometric_stats.accuracy_data{c};
        trial_tilts = compute_optimal_tilt(CONFIG, trial_acc, ...
            CONFIG.coherence_levels(c), result.optimal_strategy);
        scatter(trial_acc, trial_tilts, 20, colors(c, :), 'filled', ...
            'MarkerFaceAlpha', 0.4, 'DisplayName', sprintf('%d%%', CONFIG.coherence_levels(c)));
    end
    
    xlabel('Accuracy', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Tilt', 'FontSize', 13, 'FontWeight', 'bold');
    title('Tilt vs Accuracy (Optimal Strategy)', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    xlim([0 1]);
    ylim([0 1]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    hold off;
    
    % Subplot 4: Hit rate by coherence (all strategies)
    subplot(2, 3, 4);
    hold on;
    
    % Plot all strategies
    if isfield(result, 'all_strategy_performances')
        for st = CONFIG.optimization.strategies_to_test
            field_name = sprintf('strategy%d', st);
            if isfield(result.all_strategy_performances, field_name)
                strategy_params = result.all_strategy_performances.(field_name);
                hit_rates = zeros(length(CONFIG.coherence_levels), 1);
                for c = 1:length(CONFIG.coherence_levels)
                    trial_acc = result.psychometric_stats.accuracy_data{c};
                    trial_tilts = compute_optimal_tilt(CONFIG, trial_acc, ...
                        CONFIG.coherence_levels(c), strategy_params);
                    arc_widths = CONFIG.max_arc_deg - ...
                        (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts;
                    angular_errors = (1 - trial_acc) * 180;
                    hits = angular_errors < (arc_widths / 2);
                    hit_rates(c) = mean(hits);
                end
                plot(CONFIG.coherence_levels, hit_rates, '-o', 'LineWidth', 2.5, ...
                    'MarkerSize', 10, 'Color', strategy_colors{st}, ...
                    'DisplayName', strategy_names{st});
            end
        end
    end
    
    % Plot optimal strategy
    hit_rates_optimal = zeros(length(CONFIG.coherence_levels), 1);
    for c = 1:length(CONFIG.coherence_levels)
        trial_acc = result.psychometric_stats.accuracy_data{c};
        trial_tilts = compute_optimal_tilt(CONFIG, trial_acc, ...
            CONFIG.coherence_levels(c), result.optimal_strategy);
        arc_widths = CONFIG.max_arc_deg - ...
            (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts;
        angular_errors = (1 - trial_acc) * 180;
        hits = angular_errors < (arc_widths / 2);
        hit_rates_optimal(c) = mean(hits);
    end
    plot(CONFIG.coherence_levels, hit_rates_optimal, '-o', 'LineWidth', 3, ...
        'MarkerSize', 12, 'Color', [1 0 0], 'MarkerFaceColor', [1 0 0], ...
        'DisplayName', sprintf('OPTIMAL (Type %d)', result.optimal_strategy.strategy_type));
    
    xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Hit Rate', 'FontSize', 13, 'FontWeight', 'bold');
    title('Hit Rate vs Coherence (All Strategies)', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    ylim([0 1.1]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    hold off;
    
    % Subplot 5: Reward by coherence (all strategies)
    subplot(2, 3, 5);
    hold on;
    
    % Plot all strategies
    if isfield(result, 'all_strategy_performances')
        for st = CONFIG.optimization.strategies_to_test
            field_name = sprintf('strategy%d', st);
            if isfield(result.all_strategy_performances, field_name)
                strategy_params = result.all_strategy_performances.(field_name);
                rewards_by_coh = zeros(length(CONFIG.coherence_levels), 1);
                for c = 1:length(CONFIG.coherence_levels)
                    trial_acc = result.psychometric_stats.accuracy_data{c};
                    trial_tilts = compute_optimal_tilt(CONFIG, trial_acc, ...
                        CONFIG.coherence_levels(c), strategy_params);
                    arc_widths = CONFIG.max_arc_deg - ...
                        (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts;
                    angular_errors = (1 - trial_acc) * 180;
                    hits = angular_errors < (arc_widths / 2);
                    rewards_by_coh(c) = sum(hits .* scale_reward(trial_tilts, CONFIG));
                end
                plot(CONFIG.coherence_levels, rewards_by_coh, '-o', ...
                    'LineWidth', 2.5, 'MarkerSize', 10, 'Color', strategy_colors{st}, ...
                    'DisplayName', strategy_names{st});
            end
        end
    end
    
    % Plot optimal strategy
    rewards_optimal = zeros(length(CONFIG.coherence_levels), 1);
    for c = 1:length(CONFIG.coherence_levels)
        trial_acc = result.psychometric_stats.accuracy_data{c};
        trial_tilts = compute_optimal_tilt(CONFIG, trial_acc, ...
            CONFIG.coherence_levels(c), result.optimal_strategy);
        arc_widths = CONFIG.max_arc_deg - ...
            (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts;
        angular_errors = (1 - trial_acc) * 180;
        hits = angular_errors < (arc_widths / 2);
        rewards_optimal(c) = sum(hits .* scale_reward(trial_tilts, CONFIG));
    end
    plot(CONFIG.coherence_levels, rewards_optimal, '-o', 'LineWidth', 3, ...
        'MarkerSize', 12, 'Color', [1 0 0], 'MarkerFaceColor', [1 0 0], ...
        'DisplayName', sprintf('OPTIMAL (Type %d)', result.optimal_strategy.strategy_type));
    
    xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Cumulative Reward (ml)', 'FontSize', 13, 'FontWeight', 'bold');
    title('Reward vs Coherence (All Strategies)', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    hold off;
    
    % Subplot 6: Cumulative reward vs fixed tilt (with other strategies as dashed lines)
    subplot(2, 3, 6);
    hold on;
    
    % Plot fixed tilt curve
    plot(result.fixed_tilt_results.tilts, result.fixed_tilt_results.rewards, ...
        '-', 'LineWidth', 2.5, 'Color', [0.3 0.3 0.3], ...
        'DisplayName', 'Fixed Tilt');
    
    % Mark best fixed tilt
    [best_fixed_reward, best_fixed_idx] = max(result.fixed_tilt_results.rewards);
    best_fixed_tilt = result.fixed_tilt_results.tilts(best_fixed_idx);
    plot(best_fixed_tilt, best_fixed_reward, 'mo', 'MarkerSize', 10, ...
        'MarkerFaceColor', 'm', 'LineWidth', 2, 'DisplayName', ...
        sprintf('Best Fixed: %.4f ml', best_fixed_reward));
    
    % Plot all other strategies as horizontal dashed lines
    if isfield(result, 'all_strategy_performances')
        for st = CONFIG.optimization.strategies_to_test
            field_name = sprintf('strategy%d', st);
            if isfield(result.all_strategy_performances, field_name)
                strategy_params = result.all_strategy_performances.(field_name);
                xlim_current = xlim;
                plot(xlim_current, [strategy_params.cumulative_reward, strategy_params.cumulative_reward], ...
                    '--', 'LineWidth', 2, 'Color', strategy_colors{st}, ...
                    'DisplayName', sprintf('%s: %.4f ml', strategy_names{st}, ...
                    strategy_params.cumulative_reward));
            end
        end
    end
    
    % Plot optimal strategy as horizontal dashed line
    xlim_current = xlim;
    plot(xlim_current, [result.performance.cumulative_reward, result.performance.cumulative_reward], ...
        '--', 'LineWidth', 2.5, 'Color', [1 0 0], ...
        'DisplayName', sprintf('OPTIMAL (Type %d): %.4f ml', ...
        result.optimal_strategy.strategy_type, result.performance.cumulative_reward));
    
    xlabel('Fixed Tilt Value', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Cumulative Reward (ml)', 'FontSize', 13, 'FontWeight', 'bold');
    title('Reward vs Fixed Tilt (All Strategies)', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9);
    grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    hold off;
end
end

function best_strategy = get_best_strategy_from_type(all_strategies, strategy_type, ...
    CONFIG, accuracy_data, psychometric_stats)
% Get the best strategy from a given type for comparison

best_strategy = struct();
best_reward = -inf;

field_name = sprintf('strategy%d', strategy_type);
if ~isfield(all_strategies, field_name)
    return;
end

strategy_data = all_strategies.(field_name);
if isempty(strategy_data.params)
    return;
end

% Find best parameters
[best_reward, best_idx] = max(strategy_data.rewards);
best_params_vec = strategy_data.params(best_idx, :);

% Reconstruct strategy params
switch strategy_type
    case 1
        best_strategy.strategy_type = 1;
        best_strategy.tilt_min_coherence = best_params_vec(1);
        best_strategy.tilt_max_coherence = best_params_vec(2);
        best_strategy.noise_std = CONFIG.strategy1.noise_std;
    case 2
        best_strategy.strategy_type = 2;
        best_strategy.tilt_min_coherence = best_params_vec(1);
        best_strategy.tilt_max_coherence = best_params_vec(2);
        best_strategy.accuracy_scaling = best_params_vec(3);
        best_strategy.tilt_min_scaling = CONFIG.strategy2.tilt_range_for_scaling(1);
        best_strategy.tilt_max_scaling = CONFIG.strategy2.tilt_range_for_scaling(2);
        best_strategy.noise_std = CONFIG.strategy2.noise_std;
    case 3
        best_strategy.strategy_type = 3;
        best_strategy.tilt_min_coherence = best_params_vec(1);
        best_strategy.tilt_max_coherence = best_params_vec(2);
        best_strategy.hit_prediction_weight = best_params_vec(3);
        best_strategy.noise_std = CONFIG.strategy3.noise_std;
    case 4
        best_strategy.strategy_type = 4;
        best_strategy.tilt_min_coherence = best_params_vec(1);
        best_strategy.tilt_max_coherence = best_params_vec(2);
        best_strategy.accuracy_scaling = best_params_vec(3);
        best_strategy.tilt_min_scaling = CONFIG.strategy4.tilt_range_for_scaling(1);
        best_strategy.tilt_max_scaling = CONFIG.strategy4.tilt_range_for_scaling(2);
        best_strategy.hit_prediction_weight = best_params_vec(4);
        best_strategy.noise_std = CONFIG.strategy4.noise_std;
end

% Get accuracy range
all_accuracies = [];
for c = 1:length(accuracy_data)
    all_accuracies = [all_accuracies; accuracy_data{c}];
end
best_strategy.acc_min = min(all_accuracies);
best_strategy.acc_max = max(all_accuracies);

% Evaluate performance
performance = evaluate_wagering_strategy_comprehensive(...
    CONFIG, accuracy_data, best_strategy, psychometric_stats);
best_strategy.performance = performance;
best_strategy.cumulative_reward = performance.cumulative_reward;

end

function tilt = compute_optimal_tilt(CONFIG, accuracy, coherence, strategy_params)
% Compute tilt using optimal strategy parameters (for visualization)

n_trials = length(accuracy);

switch strategy_params.strategy_type
    case 1
        % Strategy 1: Coherence-dependent baseline only
        tilt_base = strategy_params.tilt_min_coherence + ...
            (strategy_params.tilt_max_coherence - strategy_params.tilt_min_coherence) * (coherence / 100);
        tilt = ones(n_trials, 1) * tilt_base;
        
    case 2
        % Strategy 2: Coherence-dependent + accuracy scaling
        tilt_base = strategy_params.tilt_min_coherence + ...
            (strategy_params.tilt_max_coherence - strategy_params.tilt_min_coherence) * (coherence / 100);
        
        if strategy_params.accuracy_scaling > 0
            normalized_acc = (accuracy - strategy_params.acc_min) / ...
                (strategy_params.acc_max - strategy_params.acc_min);
            tilt_from_acc = strategy_params.tilt_min_scaling + ...
                (strategy_params.tilt_max_scaling - strategy_params.tilt_min_scaling) * normalized_acc;
            tilt_modulation = (tilt_from_acc - mean(tilt_from_acc)) * strategy_params.accuracy_scaling;
            tilt = tilt_base + tilt_modulation;
        else
            tilt = ones(n_trials, 1) * tilt_base;
        end
        
    case 3
        % Strategy 3: Coherence-dependent + hit prediction
        tilt_base = strategy_params.tilt_min_coherence + ...
            (strategy_params.tilt_max_coherence - strategy_params.tilt_min_coherence) * (coherence / 100);
        
        tilt = zeros(n_trials, 1);
        for t = 1:n_trials
            acc = accuracy(t);
            angular_error = (1 - acc) * 180;
            
            test_tilts = linspace(0, 1, 50);
            expected_rewards = zeros(size(test_tilts));
            
            for ti = 1:length(test_tilts)
                test_tilt = test_tilts(ti);
                arc_width = CONFIG.max_arc_deg - ...
                    (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * test_tilt;
                p_hit = (angular_error < (arc_width / 2));
                reward_if_hit = scale_reward(test_tilt, CONFIG);
                expected_rewards(ti) = p_hit * reward_if_hit;
            end
            
            [~, best_idx] = max(expected_rewards);
            optimal_tilt = test_tilts(best_idx);
            
            tilt(t) = (1 - strategy_params.hit_prediction_weight) * tilt_base + ...
                strategy_params.hit_prediction_weight * optimal_tilt;
        end
        
    case 4
        % Strategy 4: Full combination
        tilt_base = strategy_params.tilt_min_coherence + ...
            (strategy_params.tilt_max_coherence - strategy_params.tilt_min_coherence) * (coherence / 100);
        
        if strategy_params.accuracy_scaling > 0
            normalized_acc = (accuracy - strategy_params.acc_min) / ...
                (strategy_params.acc_max - strategy_params.acc_min);
            tilt_from_acc = strategy_params.tilt_min_scaling + ...
                (strategy_params.tilt_max_scaling - strategy_params.tilt_min_scaling) * normalized_acc;
            tilt_modulation = (tilt_from_acc - mean(tilt_from_acc)) * strategy_params.accuracy_scaling;
            tilt_with_acc = tilt_base + tilt_modulation;
        else
            tilt_with_acc = ones(n_trials, 1) * tilt_base;
        end
        
        tilt = zeros(n_trials, 1);
        for t = 1:n_trials
            acc = accuracy(t);
            angular_error = (1 - acc) * 180;
            
            test_tilts = linspace(0, 1, 30);
            expected_rewards = zeros(size(test_tilts));
            
            for ti = 1:length(test_tilts)
                test_tilt = test_tilts(ti);
                arc_width = CONFIG.max_arc_deg - ...
                    (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * test_tilt;
                p_hit = (angular_error < (arc_width / 2));
                reward_if_hit = scale_reward(test_tilt, CONFIG);
                expected_rewards(ti) = p_hit * reward_if_hit;
            end
            
            [~, best_idx] = max(expected_rewards);
            optimal_tilt = test_tilts(best_idx);
            
            tilt(t) = (1 - strategy_params.hit_prediction_weight) * tilt_with_acc(t) + ...
                strategy_params.hit_prediction_weight * optimal_tilt;
        end
        
    otherwise
        error('Unknown strategy type: %d', strategy_params.strategy_type);
end

% Add noise (for visualization, use same noise as in evaluation)
noise = randn(size(tilt)) * strategy_params.noise_std;
tilt = tilt + noise;
tilt = max(0, min(1, tilt));

end
