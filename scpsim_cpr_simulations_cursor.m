function scpsim_cpr_simulations_cursor
% SCPSIM_CPR_SIMULATIONS_CURSOR - Comprehensive CPR task wagering simulation
%
% Simulates monkey performance in Continuous Perceptual Report (CPR) task
% to assess optimal reward formulations for eliciting coherence-dependent
% and accuracy-dependent wagering behavior.
%
% KEY QUESTION:
%   Is fixed tilt as effective as variable tilt under linear reward structure?
%   (i.e., does "large reward rarely" = "small reward often"?)
%
% TASK STRUCTURE:
%   - Random dot pattern (RDP) at different coherence levels
%   - Subject uses joystick: angle = direction, tilt = confidence
%   - Arc width: tilt=1 -> 10 deg, tilt=0 -> 180 deg
%   - Reward: function of tilt, only when response arc hits target
%   - Target appears in veridical RDP direction
%
% OUTPUTS:
%   - Console summary of results
%   - Figure 1: Psychometric function and accuracy distributions
%   - Figure 2: Coherence-dependent vs Adaptive Wagering Strategies
%   - Figure 3: Accuracy vs Tilt Scatter Plots by Coherence
%   - Text file: 'cpr_simulation_results.txt'
%
% Based on: Schneider et al. (2024) eLife 101021
% https://elifesciences.org/reviewed-preprints/101021
%
% Created with Cursor AI for Igor Kagan, DPZ Göttingen

%% ========================================================================
%  CONFIGURATION SECTION - Modify parameters here
%  ========================================================================

% --- Coherence Levels ---
CONFIG.coherence_levels = [0 36 59 98];

% --- Task Parameters ---
CONFIG.n_trials_per_coherence = 200;
CONFIG.targets_per_second = 0.5;

% --- Arc Width Mapping ---
CONFIG.min_arc_deg = 10;                   % Arc at maximum tilt (1.0)
CONFIG.max_arc_deg = 180;                  % Arc at minimum tilt (0.0)

% --- Reward Amounts ---
CONFIG.reward_min_ml = 0.05;               % Reward for tilt=0 (minimal confidence)
CONFIG.reward_max_ml = 0.5;                % Reward for tilt=1 (maximal confidence)

% --- Psychometric Function Parameters ---
CONFIG.psychometric.baseline_acc = 0.5;    % Chance level (random responding, ~90° error)
CONFIG.psychometric.max_acc = 0.85;        % Ceiling accuracy (asymptote)
CONFIG.psychometric.slope = 0.08;          % Steepness of psychometric curve
CONFIG.psychometric.threshold = 50;        % Coherence at midpoint between chance and max accuracy
CONFIG.psychometric.variability = 0.1;     % Trial-by-trial variability of accuracy

% --- Strategy Parameters ---
% Coherence-dependent tilt: maps coherence to tilt range
CONFIG.strategy.coherence_dependent.tilt_min = 0.1;  % Tilt at 0% coherence
CONFIG.strategy.coherence_dependent.tilt_max = 0.6;  % Tilt at 100% coherence
CONFIG.strategy.coherence_dependent.noise_std = 0.1;  % Gaussian noise standard deviation

% Adaptive wagering: scales tilt with trial-by-trial accuracy
CONFIG.strategy.adaptive_wagering.noise_std = 0.1;   % Gaussian noise standard deviation
CONFIG.strategy.adaptive_wagering.accuracy_scaling = 1.0;  % Scaling factor for tilt based on accuracy 

% Fixed tilt strategies to test
CONFIG.strategy.fixed_tilts = [0.2, 0.4, 0.6, 0.8];

% --- Strategies to Test ---
% Each strategy is a function: tilt = f(coherence, accuracy, mean_accuracy, CONFIG)
% Fixed tilt strategies are generated automatically from CONFIG.strategy.fixed_tilts
CONFIG.strategies = {};

% Generate fixed tilt strategies from CONFIG.strategy.fixed_tilts
for i = 1:length(CONFIG.strategy.fixed_tilts)
    tilt_val = CONFIG.strategy.fixed_tilts(i);
    strategy_name = sprintf('Fixed (%.1f)', tilt_val);
    CONFIG.strategies{end+1, 1} = strategy_name;
    CONFIG.strategies{end, 2} = @(coh, acc, mean_acc, cfg) tilt_val;
end

% Add variable strategies
CONFIG.strategies{end+1, 1} = 'Coherence Scaled';
CONFIG.strategies{end, 2} = @(coh, acc, mean_acc, cfg) compute_coherence_tilt(coh, acc, mean_acc, cfg);

CONFIG.strategies{end+1, 1} = 'Adaptive Wagering';
CONFIG.strategies{end, 2} = @(coh, acc, mean_acc, cfg) compute_adaptive_tilt(coh, acc, mean_acc, cfg);

% --- Reward Formulations to Test ---
% Each formulation: reward = f(tilt, hit, accuracy, CONFIG)
CONFIG.reward_formulations = {
    'Linear',               @(tilt, hit, acc, cfg) scale_reward(hit .* tilt, cfg);
    % To add more formulations, uncomment and add below:
    % 'Quadratic',            @(tilt, hit, acc, cfg) scale_reward(hit .* (tilt.^2), cfg);
    % 'Square Root',          @(tilt, hit, acc, cfg) scale_reward(hit .* sqrt(tilt), cfg);
};

% --- Visualization Options ---
CONFIG.show_figures = true;
CONFIG.save_results_to_file = true;
CONFIG.results_filename = 'cpr_simulation_results.txt';

%% ========================================================================
%  MAIN SIMULATION
%  ========================================================================

% Set random seed for reproducibility
% rng(42, 'twister');

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  CPR Task Wagering Simulation - Cursor Edition                ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

if CONFIG.save_results_to_file
    diary(CONFIG.results_filename);
    diary on;
end

%% STEP 1: Generate Psychometric Data
fprintf('STEP 1: Generating Psychometric Data\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

[accuracy_data, psychometric_stats] = generate_psychometric_data(CONFIG);

fprintf('  Coherence | Mean Accuracy | Std Accuracy | Angular Error (deg)\n');
fprintf('  ----------|---------------|--------------|---------------------\n');
for i = 1:length(CONFIG.coherence_levels)
    coh = CONFIG.coherence_levels(i);
    fprintf('  %8d%% | %13.3f | %12.3f | %18.1f\n', ...
        coh, psychometric_stats.mean_acc(i), ...
        psychometric_stats.std_acc(i), ...
        psychometric_stats.mean_error_deg(i));
end

%% STEP 2: Run All Simulations
fprintf('\n\nSTEP 2: Running All Simulations\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

simulation_results = run_all_simulations(CONFIG, accuracy_data);

fprintf('  Completed %d strategy × %d formulation × %d coherence scenarios\n', ...
    length(CONFIG.strategies), length(CONFIG.reward_formulations), length(CONFIG.coherence_levels));

% Print diagnostic table: Number of hits by strategy and coherence
fprintf('\n  Diagnostic: Number of Hits by Strategy and Coherence\n');
fprintf('  ─────────────────────────────────────────────────────────────────\n');
formulation_idx = 1;  % Use first formulation (Linear)
fprintf('  %-25s', 'Strategy');
for c = 1:length(CONFIG.coherence_levels)
    fprintf(' | %8d%%', CONFIG.coherence_levels(c));
end
fprintf(' | %10s\n', 'Total');
fprintf('  %s', repmat('-', 1, 25));
for c = 1:length(CONFIG.coherence_levels)
    fprintf(' | %s', repmat('-', 1, 8));
end
fprintf(' | %s\n', repmat('-', 1, 10));

for s = 1:length(simulation_results.strategy_names)
    fprintf('  %-25s', simulation_results.strategy_names{s});
    total_hits = 0;
    for c = 1:length(CONFIG.coherence_levels)
        n_hits = simulation_results.n_hits(s, formulation_idx, c);
        n_trials = simulation_results.n_trials(s, formulation_idx, c);
        total_hits = total_hits + n_hits;
        fprintf(' | %6d/%d', n_hits, n_trials);
    end
    fprintf(' | %10d\n', total_hits);
end
fprintf('\n');

%% STEP 3: Generate Visualizations
if CONFIG.show_figures
    fprintf('\n\nSTEP 3: Generating Visualizations\n');
    fprintf('─────────────────────────────────────────────────────────────────\n');
    
    create_figure1_psychometric(CONFIG, accuracy_data, psychometric_stats, simulation_results);
    fprintf('  ✓ Figure 1: Accuracy and Hit Rate vs Coherence\n');
    
    create_figure2_strategy_comparison(CONFIG, accuracy_data, psychometric_stats, simulation_results);
    fprintf('  ✓ Figure 2: Coherence-Dependent vs Adaptive Wagering Strategies\n');
    
    create_figure3_accuracy_tilt_scatter(CONFIG, accuracy_data, simulation_results);
    fprintf('  ✓ Figure 3: Accuracy vs Tilt Scatter Plots by Coherence\n');
end

%% Finalize
if CONFIG.save_results_to_file
    diary off;
    fprintf('\n\n✓ Results saved to: %s\n', CONFIG.results_filename);
end

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  Simulation Complete!                                          ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

end

%% ========================================================================
%  CORE SIMULATION FUNCTIONS
%  ========================================================================

function [accuracy_data, stats] = generate_psychometric_data(CONFIG)
% Generate accuracy distributions for each coherence level

n_coherence = length(CONFIG.coherence_levels);
accuracy_data = cell(n_coherence, 1);

stats.mean_acc = zeros(n_coherence, 1);
stats.std_acc = zeros(n_coherence, 1);
stats.mean_error_deg = zeros(n_coherence, 1);

for i = 1:n_coherence
    coh = CONFIG.coherence_levels(i);
    
    [accuracy, mean_acc, std_acc] = generate_accuracy_distribution(...
        coh, CONFIG.n_trials_per_coherence, CONFIG.psychometric);
    
    accuracy_data{i} = accuracy;
    stats.mean_acc(i) = mean_acc;
    stats.std_acc(i) = std_acc;
    stats.mean_error_deg(i) = (1 - mean_acc) * 180;
end

end

function simulation_results = run_all_simulations(CONFIG, accuracy_data)
% Run simulations for all strategy × reward formulation × coherence combinations
% Returns structured results for plotting

n_strategies = size(CONFIG.strategies, 1);
n_formulations = size(CONFIG.reward_formulations, 1);
n_coherence = length(CONFIG.coherence_levels);

% Pre-compute mean accuracies for each coherence (needed for adaptive wagering)
mean_accuracies = zeros(n_coherence, 1);
for c = 1:n_coherence
    mean_accuracies(c) = mean(accuracy_data{c});
end

% Initialize results structure
simulation_results = struct();
simulation_results.strategy_names = CONFIG.strategies(:, 1);
simulation_results.formulation_names = CONFIG.reward_formulations(:, 1);
simulation_results.coherence_levels = CONFIG.coherence_levels;

% Pre-allocate result arrays
simulation_results.tilt_data = cell(n_strategies, n_formulations, n_coherence);
simulation_results.hit_rate = zeros(n_strategies, n_formulations, n_coherence);
simulation_results.n_hits = zeros(n_strategies, n_formulations, n_coherence);
simulation_results.n_trials = zeros(n_strategies, n_formulations, n_coherence);
simulation_results.reward_per_trial = zeros(n_strategies, n_formulations, n_coherence);
simulation_results.cumulative_reward = zeros(n_strategies, n_formulations, n_coherence);
simulation_results.mean_tilt = zeros(n_strategies, n_formulations, n_coherence);
simulation_results.std_tilt = zeros(n_strategies, n_formulations, n_coherence);
simulation_results.mean_arc_width = zeros(n_strategies, n_formulations, n_coherence);
simulation_results.mean_reward_per_hit = zeros(n_strategies, n_formulations, n_coherence);

% Run simulations
for s = 1:n_strategies
    strategy_func = CONFIG.strategies{s, 2};
    
    for f = 1:n_formulations
        reward_func = CONFIG.reward_formulations{f, 2};
        
        for c = 1:n_coherence
            coh = CONFIG.coherence_levels(c);
            trial_accuracies = accuracy_data{c};
            mean_acc = mean_accuracies(c);
            n_trials = length(trial_accuracies);
            
            % Calculate tilt for all trials (vectorized)
            trial_tilts = strategy_func(coh, trial_accuracies, mean_acc, CONFIG);
            trial_tilts = max(0, min(1, trial_tilts));  % Clip to [0, 1]
            
            % Calculate arc widths
            arc_widths = compute_arc_width(trial_tilts, CONFIG);
            
            % Determine hits
            angular_errors = (1 - trial_accuracies) * 180;
            hits = angular_errors < (arc_widths / 2);
            
            % Calculate rewards
            rewards = reward_func(trial_tilts, hits, trial_accuracies, CONFIG);
            
            % Store results
            simulation_results.tilt_data{s, f, c} = trial_tilts;
            simulation_results.hit_rate(s, f, c) = mean(hits);
            simulation_results.n_hits(s, f, c) = sum(hits);
            simulation_results.n_trials(s, f, c) = n_trials;
            simulation_results.reward_per_trial(s, f, c) = mean(rewards);
            simulation_results.cumulative_reward(s, f, c) = sum(rewards);
            simulation_results.mean_tilt(s, f, c) = mean(trial_tilts);
            simulation_results.std_tilt(s, f, c) = std(trial_tilts);
            simulation_results.mean_arc_width(s, f, c) = mean(arc_widths);
            
            % Mean reward per hit (only for trials that hit)
            if sum(hits) > 0
                simulation_results.mean_reward_per_hit(s, f, c) = mean(rewards(hits));
            else
                simulation_results.mean_reward_per_hit(s, f, c) = 0;
            end
        end
    end
end

end

function tilt = compute_coherence_tilt(coherence, accuracy, mean_accuracy, CONFIG)
% Compute coherence-dependent tilt with Gaussian noise
% Handles both scalar and vector inputs for accuracy

% Base tilt from coherence-dependent strategy
tilt = CONFIG.strategy.coherence_dependent.tilt_min + ...
    (CONFIG.strategy.coherence_dependent.tilt_max - CONFIG.strategy.coherence_dependent.tilt_min) * (coherence / 100);

% Add noise (vectorized if accuracy is a vector)
noise = randn(size(accuracy)) * CONFIG.strategy.coherence_dependent.noise_std;
tilt = tilt + noise;

% Clip to valid range
tilt = max(0, min(1, tilt));

end

function tilt = compute_adaptive_tilt(coherence, accuracy, mean_accuracy, CONFIG)
% Compute adaptive tilt: centered around coherence-dependent tilt, scaled with accuracy
% Tilt SD matches coherence-based strategy SD
% Handles both scalar and vector inputs for accuracy

% Base tilt from coherence-dependent strategy
tilt_base = CONFIG.strategy.coherence_dependent.tilt_min + ...
    (CONFIG.strategy.coherence_dependent.tilt_max - CONFIG.strategy.coherence_dependent.tilt_min) * (coherence / 100);

% Scale with accuracy deviation from mean
accuracy_deviation = accuracy - mean_accuracy;
accuracy_modulation = CONFIG.strategy.adaptive_wagering.accuracy_scaling * accuracy_deviation;

% Add noise (vectorized if accuracy is a vector)
noise = randn(size(accuracy)) * CONFIG.strategy.adaptive_wagering.noise_std;

% Combine: tilt = tilt_base + accuracy_modulation + noise
tilt = tilt_base + accuracy_modulation + noise;

% Normalize tilt distribution to match target SD
% First, ensure mean is at tilt_base (accuracy modulation should average to 0, but noise might shift it)
target_sd = CONFIG.strategy.coherence_dependent.noise_std;
current_mean = mean(tilt);
current_sd = std(tilt);

% Adjust mean to tilt_base and scale SD to target
if current_sd > 0
    tilt = tilt_base + (tilt - current_mean) * (target_sd / current_sd);
else
    tilt = tilt_base + (tilt - current_mean);
end

% Clip to valid range
tilt = max(0, min(1, tilt));

end

function arc_widths = compute_arc_width(tilts, CONFIG)
% Compute arc widths from tilt values (handles both scalars and vectors)

arc_widths = CONFIG.max_arc_deg - ...
    (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * tilts;

end

function [accuracy, mean_acc, std_acc] = generate_accuracy_distribution(...
    coherence, n_trials, psychometric_params)
% Generate accuracy distribution for given coherence using psychometric function

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
% Generate normally-distributed random samples, clipped to [0,1]

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

function reward_ml = scale_reward(normalized_reward, CONFIG)
% Scale normalized reward [0,1] to actual reward amount in ml

reward_ml = CONFIG.reward_min_ml + ...
    (CONFIG.reward_max_ml - CONFIG.reward_min_ml) * normalized_reward;

end

%% ========================================================================
%  VISUALIZATION FUNCTIONS
%  ========================================================================

function create_figure1_psychometric(CONFIG, accuracy_data, stats, simulation_results)
% Figure 1: Accuracy and hit rate as function of coherence

fig = figure('Name', 'Accuracy and Hit Rate vs Coherence', 'Position', [100 100 1400 800]);
sgtitle('CPR Task: Accuracy and Hit Rate as Function of Coherence', ...
    'FontSize', 18, 'FontWeight', 'bold', 'Color', [0 0.3 0.6]);

n_coherence = length(CONFIG.coherence_levels);

% Subplot 1: Accuracy vs Coherence
subplot(2, 3, 1);
plot(CONFIG.coherence_levels, stats.mean_acc, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0]);
hold on;
errorbar(CONFIG.coherence_levels, stats.mean_acc, stats.std_acc, ...
    'LineStyle', 'none', 'LineWidth', 2.5, 'Color', [0 0 0], 'CapSize', 12);
yline(0.5, 'k--', 'LineWidth', 2.5, 'Label', 'Chance (0.5)', ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Accuracy (0 to 1)', 'FontSize', 13, 'FontWeight', 'bold');
title('Accuracy vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
ylim([0.35 1.05]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 2: Angular Error vs Coherence
subplot(2, 3, 2);
plot(CONFIG.coherence_levels, stats.mean_error_deg, '-s', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0]);
hold on;
yline(90, 'k--', 'LineWidth', 2.5, 'Label', 'Chance (90°)', ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Mean Angular Error (degrees)', 'FontSize', 13, 'FontWeight', 'bold');
title('Angular Error vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
ylim([0 180]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 3: Accuracy distributions
subplot(2, 3, 3);
hold on;
colors = cool(n_coherence);
for i = 1:n_coherence
    [counts, edges] = histcounts(accuracy_data{i}, 30);
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    bin_width = edges(2) - edges(1);
    probability = counts / (sum(counts) * bin_width);
    plot(centers, probability, 'LineWidth', 2.5, ...
        'Color', colors(i, :), ...
        'DisplayName', sprintf('%d%%', CONFIG.coherence_levels(i)));
end
xlabel('Accuracy', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Probability Density', 'FontSize', 13, 'FontWeight', 'bold');
title('Accuracy Distributions', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10, 'Box', 'off');
grid on;
xlim([0.2 1]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 4: Hit Rate vs Tilt
subplot(2, 3, 4);
hold on;
tilt_range = 0:0.01:1;
for i = 1:n_coherence
    trial_accuracies = accuracy_data{i};
    angular_errors = (1 - trial_accuracies) * 180;
    hit_prob = zeros(size(tilt_range));
    
    for t_idx = 1:length(tilt_range)
        arc_width = compute_arc_width(tilt_range(t_idx), CONFIG);
        hits = angular_errors < (arc_width / 2);
        hit_prob(t_idx) = mean(hits);
    end
    
    plot(tilt_range, hit_prob, 'LineWidth', 3, 'Color', colors(i, :), ...
        'DisplayName', sprintf('%d%% (err=%.0f°)', CONFIG.coherence_levels(i), stats.mean_error_deg(i)));
end
xlabel('Tilt (Confidence)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Hit Probability', 'FontSize', 13, 'FontWeight', 'bold');
title('Hit Rate vs Tilt', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 9, 'Box', 'off');
grid on;
ylim([0 1.1]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 5: Hit Rate vs Coherence (for different fixed tilts)
subplot(2, 3, 5);
hold on;
% Find fixed tilt strategy indices (strategies that start with "Fixed")
fixed_tilt_values = CONFIG.strategy.fixed_tilts;
fixed_strategy_indices = [];
for s = 1:length(simulation_results.strategy_names)
    if startsWith(simulation_results.strategy_names{s}, 'Fixed')
        fixed_strategy_indices(end+1) = s;
    end
end
tilt_colors = parula(length(fixed_strategy_indices));
formulation_idx = 1;  % Use first formulation (Linear)

for t_idx = 1:length(fixed_strategy_indices)
    s_idx = fixed_strategy_indices(t_idx);
    tilt_val = fixed_tilt_values(t_idx);
    arc_width_val = compute_arc_width(tilt_val, CONFIG);
    
    hit_rates = squeeze(simulation_results.hit_rate(s_idx, formulation_idx, :));
    
    plot(CONFIG.coherence_levels, hit_rates, '-o', 'LineWidth', 3, ...
        'MarkerSize', 12, 'Color', tilt_colors(t_idx, :), ...
        'MarkerFaceColor', tilt_colors(t_idx, :), ...
        'DisplayName', sprintf('Tilt=%.1f (arc=%.0f°)', tilt_val, arc_width_val));
end
xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Hit Rate (0 to 1)', 'FontSize', 13, 'FontWeight', 'bold');
title('Hit Rate vs Coherence (by Tilt)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 10, 'Box', 'off');
grid on;
ylim([0 1.1]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 6: Cumulative Reward vs Coherence (by Tilt)
subplot(2, 3, 6);
hold on;
% Use same fixed strategy indices as subplot 5
tilt_colors = parula(length(fixed_strategy_indices));

for t_idx = 1:length(fixed_strategy_indices)
    s_idx = fixed_strategy_indices(t_idx);
    tilt_val = fixed_tilt_values(t_idx);
    
    cumulative_reward = squeeze(simulation_results.cumulative_reward(s_idx, formulation_idx, :));
    total_reward = sum(cumulative_reward);
    
    plot(CONFIG.coherence_levels, cumulative_reward, '-o', 'LineWidth', 3, ...
        'MarkerSize', 12, 'Color', tilt_colors(t_idx, :), ...
        'MarkerFaceColor', tilt_colors(t_idx, :));
    
    xlim_current = xlim;
    plot(xlim_current, [total_reward total_reward], '--', ...
        'LineWidth', 2, 'Color', tilt_colors(t_idx, :));
end

xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Cumulative Reward (ml)', 'FontSize', 13, 'FontWeight', 'bold');
title('Cumulative Reward vs Coherence (by Tilt)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

end

function create_figure2_strategy_comparison(CONFIG, accuracy_data, stats, simulation_results)
% Figure 2: Comparison of coherence-dependent and adaptive wagering strategies

fig = figure('Name', 'Strategy Comparison', 'Position', [100 100 1400 800]);
sgtitle('CPR Task: Coherence-Dependent vs Adaptive Wagering Strategies', ...
    'FontSize', 18, 'FontWeight', 'bold', 'Color', [0 0.3 0.6]);

n_coherence = length(CONFIG.coherence_levels);

% Find strategy indices
coherence_strategy_idx = find(strcmp(simulation_results.strategy_names, 'Coherence Scaled'));
adaptive_strategy_idx = find(strcmp(simulation_results.strategy_names, 'Adaptive Wagering'));
formulation_idx = 1;  % Use first formulation (Linear)

% Subplot 1: Accuracy vs Coherence
subplot(2, 3, 1);
plot(CONFIG.coherence_levels, stats.mean_acc, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0]);
hold on;
errorbar(CONFIG.coherence_levels, stats.mean_acc, stats.std_acc, ...
    'LineStyle', 'none', 'LineWidth', 2.5, 'Color', [0 0 0], 'CapSize', 12);
yline(0.5, 'k--', 'LineWidth', 2.5, 'Label', 'Chance (0.5)', ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Accuracy (0 to 1)', 'FontSize', 13, 'FontWeight', 'bold');
title('Accuracy vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
ylim([0.35 1.05]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 2: Mean Tilt vs Coherence (with error bars showing SD)
subplot(2, 3, 2);
hold on;
tilt_coherence = squeeze(simulation_results.mean_tilt(coherence_strategy_idx, formulation_idx, :));
std_tilt_coherence = squeeze(simulation_results.std_tilt(coherence_strategy_idx, formulation_idx, :));
tilt_adaptive = squeeze(simulation_results.mean_tilt(adaptive_strategy_idx, formulation_idx, :));
std_tilt_adaptive = squeeze(simulation_results.std_tilt(adaptive_strategy_idx, formulation_idx, :));

errorbar(CONFIG.coherence_levels, tilt_coherence, std_tilt_coherence, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0.4 0.8], 'MarkerFaceColor', [0 0.4 0.8], ...
    'CapSize', 12, 'DisplayName', 'Coherence-dependent');
errorbar(CONFIG.coherence_levels, tilt_adaptive, std_tilt_adaptive, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0.8 0.2 0.2], 'MarkerFaceColor', [0.8 0.2 0.2], ...
    'CapSize', 12, 'DisplayName', 'Adaptive wagering');
xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Mean Tilt (0 to 1)', 'FontSize', 13, 'FontWeight', 'bold');
title('Mean Tilt vs Coherence (±SD)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10, 'Box', 'off');
grid on;
ylim([0 1]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 3: Mean Arc Width vs Coherence
subplot(2, 3, 3);
hold on;
arc_coherence = squeeze(simulation_results.mean_arc_width(coherence_strategy_idx, formulation_idx, :));
arc_adaptive = squeeze(simulation_results.mean_arc_width(adaptive_strategy_idx, formulation_idx, :));
plot(CONFIG.coherence_levels, arc_coherence, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0.4 0.8], 'MarkerFaceColor', [0 0.4 0.8], ...
    'DisplayName', 'Coherence-dependent');
plot(CONFIG.coherence_levels, arc_adaptive, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0.8 0.2 0.2], 'MarkerFaceColor', [0.8 0.2 0.2], ...
    'DisplayName', 'Adaptive wagering');
xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Mean Arc Width (degrees)', 'FontSize', 13, 'FontWeight', 'bold');
title('Mean Arc Width vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10, 'Box', 'off');
grid on;
ylim([0 200]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 4: Hit Rate vs Coherence
subplot(2, 3, 4);
hold on;
hit_coherence = squeeze(simulation_results.hit_rate(coherence_strategy_idx, formulation_idx, :));
hit_adaptive = squeeze(simulation_results.hit_rate(adaptive_strategy_idx, formulation_idx, :));
plot(CONFIG.coherence_levels, hit_coherence, '-o', 'LineWidth', 3, ...
    'MarkerSize', 14, 'Color', [0 0.4 0.8], 'MarkerFaceColor', [0 0.4 0.8], ...
    'DisplayName', 'Coherence-dependent');
plot(CONFIG.coherence_levels, hit_adaptive, '-o', 'LineWidth', 3, ...
    'MarkerSize', 14, 'Color', [0.8 0.2 0.2], 'MarkerFaceColor', [0.8 0.2 0.2], ...
    'DisplayName', 'Adaptive wagering');
xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Hit Rate (0 to 1)', 'FontSize', 13, 'FontWeight', 'bold');
title('Hit Rate vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10, 'Box', 'off');
grid on;
ylim([0 1.1]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 5: Mean Reward per Hit vs Coherence
subplot(2, 3, 5);
hold on;
reward_coherence = squeeze(simulation_results.mean_reward_per_hit(coherence_strategy_idx, formulation_idx, :));
reward_adaptive = squeeze(simulation_results.mean_reward_per_hit(adaptive_strategy_idx, formulation_idx, :));
plot(CONFIG.coherence_levels, reward_coherence, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0.4 0.8], 'MarkerFaceColor', [0 0.4 0.8], ...
    'DisplayName', 'Coherence-dependent');
plot(CONFIG.coherence_levels, reward_adaptive, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0.8 0.2 0.2], 'MarkerFaceColor', [0.8 0.2 0.2], ...
    'DisplayName', 'Adaptive wagering');
xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Mean Reward per Hit (ml)', 'FontSize', 13, 'FontWeight', 'bold');
title('Mean Reward per Hit vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10, 'Box', 'off');
grid on;
ylim([0 0.6]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 6: Cumulative Reward vs Coherence
subplot(2, 3, 6);
hold on;
cum_reward_coherence = squeeze(simulation_results.cumulative_reward(coherence_strategy_idx, formulation_idx, :));
cum_reward_adaptive = squeeze(simulation_results.cumulative_reward(adaptive_strategy_idx, formulation_idx, :));
total_coherence = sum(cum_reward_coherence);
total_adaptive = sum(cum_reward_adaptive);

plot(CONFIG.coherence_levels, cum_reward_coherence, '-o', 'LineWidth', 3, ...
    'MarkerSize', 14, 'Color', [0 0.4 0.8], 'MarkerFaceColor', [0 0.4 0.8], ...
    'DisplayName', 'Coherence-dependent');
xlim_current = xlim;
plot(xlim_current, [total_coherence total_coherence], '--', ...
    'LineWidth', 2, 'Color', [0 0.4 0.8]);

plot(CONFIG.coherence_levels, cum_reward_adaptive, '-o', 'LineWidth', 3, ...
    'MarkerSize', 14, 'Color', [0.8 0.2 0.2], 'MarkerFaceColor', [0.8 0.2 0.2], ...
    'DisplayName', 'Adaptive wagering');
plot(xlim_current, [total_adaptive total_adaptive], '--', ...
    'LineWidth', 2, 'Color', [0.8 0.2 0.2]);

xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Cumulative Reward (ml)', 'FontSize', 13, 'FontWeight', 'bold');
title('Cumulative Reward vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

end

function create_figure3_accuracy_tilt_scatter(CONFIG, accuracy_data, simulation_results)
% Figure 3: Scatter plots of accuracy vs tilt for each coherence level
% Shows both coherence-based (gray) and adaptive wagering (colored) strategies

fig = figure('Name', 'Accuracy vs Tilt by Coherence', 'Position', [100 100 1400 800]);
sgtitle('CPR Task: Accuracy vs Tilt Relationship (Coherence-based vs Adaptive Wagering)', ...
    'FontSize', 18, 'FontWeight', 'bold', 'Color', [0 0.3 0.6]);

n_coherence = length(CONFIG.coherence_levels);
colors = cool(n_coherence);

coherence_strategy_idx = find(strcmp(simulation_results.strategy_names, 'Coherence Scaled'));
adaptive_strategy_idx = find(strcmp(simulation_results.strategy_names, 'Adaptive Wagering'));
formulation_idx = 1;

for i = 1:n_coherence
    subplot(2, 2, i);
    hold on;
    
    trial_accuracies = accuracy_data{i};
    angular_errors = (1 - trial_accuracies) * 180;
    
    % Calculate theoretical hit/miss boundary line
    % Hit condition: angular_error < arc_width / 2
    % At boundary: (1 - accuracy) * 180 = (max_arc - (max_arc - min_arc) * tilt) / 2
    % Solving for accuracy as function of tilt:
    % accuracy = 1 - (max_arc - (max_arc - min_arc) * tilt) / (2 * 180)
    tilt_boundary = linspace(0, 1, 100);
    arc_width_boundary = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * tilt_boundary;
    accuracy_boundary = 1 - (arc_width_boundary / 2) / 180;
    accuracy_boundary = max(0, min(1, accuracy_boundary));  % Clip to valid range
    
    % Plot theoretical boundary line (black)
    plot(accuracy_boundary, tilt_boundary, 'k--', 'LineWidth', 2.5, ...
        'DisplayName', 'Hit/Miss boundary');
    
    % Plot coherence-based strategy (gray)
    trial_tilts_coherence = simulation_results.tilt_data{coherence_strategy_idx, formulation_idx, i};
    arc_widths_coherence = compute_arc_width(trial_tilts_coherence, CONFIG);
    hits_coherence = angular_errors < (arc_widths_coherence / 2);
    
    % Plot hits (filled circles) and misses (empty circles)
    if sum(hits_coherence) > 0
        scatter(trial_accuracies(hits_coherence), trial_tilts_coherence(hits_coherence), ...
            50, [0.5 0.5 0.5], 'o', 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.8);
    end
    if sum(~hits_coherence) > 0
        scatter(trial_accuracies(~hits_coherence), trial_tilts_coherence(~hits_coherence), ...
            50, [0.5 0.5 0.5], 'o', 'MarkerEdgeAlpha', 0.6, 'LineWidth', 1);
    end
    
    p_coherence = polyfit(trial_accuracies, trial_tilts_coherence, 1);
    x_fit = linspace(min(trial_accuracies), max(trial_accuracies), 100);
    y_fit_coherence = polyval(p_coherence, x_fit);
    plot(x_fit, y_fit_coherence, '-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5], ...
        'DisplayName', sprintf('Coherence: y=%.3fx+%.3f', p_coherence(1), p_coherence(2)));
    
    y_pred_coherence = polyval(p_coherence, trial_accuracies);
    ss_res_coherence = sum((trial_tilts_coherence - y_pred_coherence).^2);
    ss_tot_coherence = sum((trial_tilts_coherence - mean(trial_tilts_coherence)).^2);
    r_squared_coherence = 1 - (ss_res_coherence / ss_tot_coherence);
    
    % Plot adaptive wagering strategy (colored)
    trial_tilts_adaptive = simulation_results.tilt_data{adaptive_strategy_idx, formulation_idx, i};
    arc_widths_adaptive = compute_arc_width(trial_tilts_adaptive, CONFIG);
    hits_adaptive = angular_errors < (arc_widths_adaptive / 2);
    
    % Plot hits (filled circles) and misses (empty circles)
    if sum(hits_adaptive) > 0
        scatter(trial_accuracies(hits_adaptive), trial_tilts_adaptive(hits_adaptive), ...
            50, colors(i, :), 'o', 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.8);
    end
    if sum(~hits_adaptive) > 0
        scatter(trial_accuracies(~hits_adaptive), trial_tilts_adaptive(~hits_adaptive), ...
            50, colors(i, :), 'o', 'MarkerEdgeAlpha', 0.6, 'LineWidth', 1);
    end
    
    p_adaptive = polyfit(trial_accuracies, trial_tilts_adaptive, 1);
    y_fit_adaptive = polyval(p_adaptive, x_fit);
    plot(x_fit, y_fit_adaptive, '-', 'LineWidth', 3, 'Color', colors(i, :), ...
        'DisplayName', sprintf('Adaptive: y=%.3fx+%.3f', p_adaptive(1), p_adaptive(2)));
    
    y_pred_adaptive = polyval(p_adaptive, trial_accuracies);
    ss_res_adaptive = sum((trial_tilts_adaptive - y_pred_adaptive).^2);
    ss_tot_adaptive = sum((trial_tilts_adaptive - mean(trial_tilts_adaptive)).^2);
    r_squared_adaptive = 1 - (ss_res_adaptive / ss_tot_adaptive);
    
    xlabel('Accuracy', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Tilt', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('%d%% Coherence (Coherence R²=%.3f, Adaptive R²=%.3f)', ...
        CONFIG.coherence_levels(i), r_squared_coherence, r_squared_adaptive), ...
        'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9, 'Box', 'off');
    grid on;
    xlim([0 1]);
    ylim([0 1]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    hold off;
end

end
