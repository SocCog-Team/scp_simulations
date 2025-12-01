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
%   - Figure 2: Strategy comparison across reward formulations
%   - Figure 3: Detailed breakdown by coherence level
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
% Motion coherence in percentage (0 = no signal, 100 = perfect signal)
CONFIG.coherence_levels = [0 36 59 98];

% --- Task Parameters ---
CONFIG.n_trials_per_coherence = 1000;      % Trials per coherence level
CONFIG.target_width_deg = 5;              % Target width in degrees
CONFIG.targets_per_second = 0.5;           % Target presentation rate

% --- Arc Width Mapping ---
% Formula: arc_width = max_arc - (max_arc - min_arc) * tilt
CONFIG.min_arc_deg = 10;                   % Arc at maximum tilt (1.0)
CONFIG.max_arc_deg = 180;                  % Arc at minimum tilt (0.0)

% --- Reward Amounts ---
% Actual juice/water reward delivered (in ml)
CONFIG.reward_min_ml = 0.05;               % Reward for tilt=0 (minimal confidence)
CONFIG.reward_max_ml = 0.5;                % Reward for tilt=1 (maximal confidence)
% Note: Reward scales between min and max based on tilt and formulation

% --- Psychometric Function Parameters ---
% Maps coherence to accuracy: acc = baseline + (max-baseline)/(1+exp(-slope*(coh-threshold)))
% Accuracy defined as: 1 - (angular_error / 180), where:
%   0° error = 1.0 accuracy (perfect)
%   90° error = 0.5 accuracy (chance level for circular/directional data)
%   180° error = 0.0 accuracy (worst possible)
CONFIG.psychometric.baseline_acc = 0.5;    % Chance level (random responding, ~90° error)
CONFIG.psychometric.max_acc = 0.85;        % Ceiling accuracy (asymptote)
CONFIG.psychometric.slope = 0.08;         % Steepness of psychometric curve (higher = steeper)
CONFIG.psychometric.threshold = 50;       % Coherence at midpoint between chance and max accuracy
CONFIG.psychometric.variability = 0.1;    % Trial-by-trial variability

% --- Adaptive Wagering Parameters ---
CONFIG.adaptive_wagering.noise_std = 0.1;  % Standard deviation of Gaussian noise for adaptive wagering
CONFIG.adaptive_wagering.accuracy_scaling = 1.0;  % How much tilt scales with accuracy (higher = more aggressive)

% --- Strategies to Test ---
% Each strategy is a function: tilt = f(coherence, accuracy)
CONFIG.strategies = {
    'Fixed Low (0.2)',      @(coh, acc) 0.2;
    'Fixed Med (0.5)',      @(coh, acc) 0.5;
    'Fixed High (0.8)',     @(coh, acc) 0.8;
    'Coherence Linear',     @(coh, acc) coh/100;
    'Coherence Scaled',     @(coh, acc) 0.1 + 0.5*(coh/100);  % Maps 0-100 to 0.1-0.6
    'Match Accuracy',       @(coh, acc) acc;
};

% --- Reward Formulations to Test ---
% Each formulation: reward = f(tilt, hit, accuracy, CONFIG)
% hit is boolean, tilt and accuracy are in [0,1]
% Rewards are scaled from reward_min_ml to reward_max_ml
CONFIG.reward_formulations = {
    'Linear',               @(tilt, hit, acc, cfg) scale_reward(hit .* tilt, cfg);
    
    % To test other formulations, uncomment and add below:
    % 'Quadratic',            @(tilt, hit, acc, cfg) scale_reward(hit .* (tilt.^2), cfg);
    % 'Square Root',          @(tilt, hit, acc, cfg) scale_reward(hit .* sqrt(tilt), cfg);
    % 'Exponential (k=2)',    @(tilt, hit, acc, cfg) scale_reward(hit .* (exp(2*tilt) - 1) / (exp(2) - 1), cfg);
    % 'Accuracy Bonus',       @(tilt, hit, acc, cfg) scale_reward(hit .* tilt .* (1 + 0.5*acc), cfg);
    % 'Miss Penalty',         @(tilt, hit, acc, cfg) scale_reward(hit .* tilt, cfg) - (~hit) * 0.01;
    % 'Threshold Bonus',      @(tilt, hit, acc, cfg) scale_reward(hit .* (tilt + 0.3 * (tilt > 0.6)), cfg);
};

% --- Visualization Options ---
CONFIG.show_figures = true;
CONFIG.save_results_to_file = true;
CONFIG.results_filename = 'cpr_simulation_results.txt';

%% ========================================================================
%  MAIN SIMULATION
%  ========================================================================

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  CPR Task Wagering Simulation - Cursor Edition                ║\n');
fprintf('║  Testing: Fixed vs Adaptive Tilt Strategies                   ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

% Start output file if requested
if CONFIG.save_results_to_file
    diary(CONFIG.results_filename);
    diary on;
end

%% PART 1: Analyze Psychometric Function
fprintf('PART 1: Psychometric Function Analysis\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

[accuracy_data, psychometric_stats] = analyze_psychometric_function(CONFIG);

fprintf('  Coherence | Mean Accuracy | Std Accuracy | Angular Error (deg) | Min Acc | Max Acc\n');
fprintf('  ----------|---------------|--------------|---------------------|---------|--------\n');
for i = 1:length(CONFIG.coherence_levels)
    coh = CONFIG.coherence_levels(i);
    fprintf('  %8d%% | %13.3f | %12.3f | %18.1f | %7.3f | %7.3f\n', ...
        coh, psychometric_stats.mean_acc(i), ...
        psychometric_stats.std_acc(i), ...
        psychometric_stats.mean_error_deg(i), ...
        min(accuracy_data{i}), max(accuracy_data{i}));
end

if 0 % Debug: Check hit rate for tilt 0.8 at highest coherence
fprintf('\n  DEBUG - Tilt 0.8 analysis at 98%% coherence:\n');
highest_coh_idx = find(CONFIG.coherence_levels == 98);
if ~isempty(highest_coh_idx)
    tilt_test = 0.8;
    arc_test = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * tilt_test;
    trial_acc = accuracy_data{highest_coh_idx};
    ang_err = (1 - trial_acc) * 180;
    hits_test = ang_err < (arc_test/2);
    fprintf('    Arc width: %.1f° (threshold: %.1f°)\n', arc_test, arc_test/2);
    fprintf('    Mean angular error: %.1f°\n', mean(ang_err));
    fprintf('    Min/Max angular error: %.1f° / %.1f°\n', min(ang_err), max(ang_err));
    fprintf('    Trials with error < threshold: %d / %d\n', sum(hits_test), length(hits_test));
    fprintf('    Hit rate: %.3f\n', mean(hits_test));
end
end % of if Debug

%% PART 2: Test Strategies with Different Reward Formulations
fprintf('\n\nPART 2: Strategy Performance Across Reward Formulations\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

[strategy_results, formulation_comparison] = test_all_strategies(CONFIG);

% Display summary table
display_results_table(CONFIG, strategy_results, formulation_comparison);

%% PART 3: Key Insights and Recommendations
fprintf('\n\nPART 3: Key Insights\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

insights = generate_insights(CONFIG, formulation_comparison);
display_insights(insights);

%% PART 4: Generate Visualizations
if CONFIG.show_figures
    fprintf('\n\nPART 4: Generating Visualizations\n');
    fprintf('─────────────────────────────────────────────────────────────────\n');
    
    create_figure1_psychometric(CONFIG, accuracy_data, psychometric_stats);
    fprintf('  ✓ Figure 1: Accuracy and Hit Rate vs Coherence (Fixed Tilt Strategies)\n');
    
    % Calculate adaptive wagering tilts once for consistency across figures
    % Adaptive tilt is centered around coherence-dependent tilt but scales with accuracy
    n_coherence = length(CONFIG.coherence_levels);
    adaptive_tilt_data = cell(n_coherence, 1);
    tilt_at_coherence = @(coh) 0.1 + 0.5 * (coh / 100);  % Coherence-dependent tilt function
    
    for i = 1:n_coherence
        trial_accuracies = accuracy_data{i};
        mean_accuracy = mean(trial_accuracies);
        tilt_coherence = tilt_at_coherence(CONFIG.coherence_levels(i));
        
        % Adaptive wagering: tilt scales with accuracy around coherence-dependent mean
        % High accuracy → higher tilt (narrower arc, but still hits because error is small) → higher reward
        % Low accuracy → lower tilt (wider arc, compensates for large error) → lower reward but more hits
        % Formula: tilt = tilt_coherence + scaling * (accuracy - mean_accuracy) + noise
        % This ensures mean tilt = tilt_coherence (since mean of (accuracy - mean_accuracy) = 0)
        accuracy_deviation = trial_accuracies - mean_accuracy;
        
        % Scale the deviation more aggressively to maximize benefit
        scaled_deviation = CONFIG.adaptive_wagering.accuracy_scaling * accuracy_deviation;
        tilt_base = tilt_coherence + scaled_deviation;
        
        % Add small noise
        noise = randn(size(trial_accuracies)) * CONFIG.adaptive_wagering.noise_std;
        adaptive_tilt_data{i} = max(0, min(1, tilt_base + noise));
        
        % Verify mean tilt is approximately correct (for debugging)
        actual_mean_tilt = mean(adaptive_tilt_data{i});
        if abs(actual_mean_tilt - tilt_coherence) > 0.05
            warning('Adaptive wagering mean tilt (%.3f) deviates from coherence-dependent tilt (%.3f) at %d%% coherence', ...
                actual_mean_tilt, tilt_coherence, CONFIG.coherence_levels(i));
        end
    end
    
    create_figure2_coherence_strategy(CONFIG, accuracy_data, psychometric_stats, adaptive_tilt_data);
    fprintf('  ✓ Figure 2: Coherence-Dependent vs Adaptive Wagering Strategies\n');
    
    create_figure3_accuracy_tilt_scatter(CONFIG, accuracy_data, adaptive_tilt_data);
    fprintf('  ✓ Figure 3: Accuracy vs Tilt Scatter Plots by Coherence\n');
    
    % Other figures disabled for now
    % create_figure2_strategy_comparison(CONFIG, strategy_results, formulation_comparison);
    % create_figure3_detailed_analysis(CONFIG, strategy_results);
    % create_figure4_reward_surfaces(CONFIG);
    
    % fprintf('  ✓ Figure 2: Strategy comparison\n');
    % fprintf('  ✓ Figure 3: Detailed analysis by coherence\n');
    % fprintf('  ✓ Figure 4: Reward surfaces\n');
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
%  HELPER FUNCTIONS
%  ========================================================================

function [accuracy_data, stats] = analyze_psychometric_function(CONFIG)
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

function [strategy_results, formulation_comparison] = test_all_strategies(CONFIG)
% Run simulations for all strategy x formulation combinations

n_strategies = size(CONFIG.strategies, 1);
n_formulations = size(CONFIG.reward_formulations, 1);
n_coherence = length(CONFIG.coherence_levels);

% Initialize storage
strategy_results = struct();

for f = 1:n_formulations
    formulation_name = CONFIG.reward_formulations{f, 1};
    formulation_name_clean = clean_field_name(formulation_name);
    
    strategy_results.(formulation_name_clean) = zeros(n_strategies, 4);
    % Columns: hit_rate, reward_per_trial, reward_rate, mean_tilt
end

% Run simulations
for f = 1:n_formulations
    formulation_name = CONFIG.reward_formulations{f, 1};
    formulation_name_clean = clean_field_name(formulation_name);
    reward_func = CONFIG.reward_formulations{f, 2};
    
    for s = 1:n_strategies
        strategy_func = CONFIG.strategies{s, 2};
        
        % Simulate across all coherence levels
        [hit_rate, reward_per_trial, reward_rate, mean_tilt] = ...
            simulate_strategy(CONFIG, strategy_func, reward_func);
        
        strategy_results.(formulation_name_clean)(s, :) = ...
            [hit_rate, reward_per_trial, reward_rate, mean_tilt];
    end
end

% Compare formulations
formulation_comparison = struct();
formulation_comparison.names = {};
formulation_comparison.fixed_best = [];
formulation_comparison.adaptive_best = [];
formulation_comparison.adaptive_advantage = [];

for f = 1:n_formulations
    formulation_name = CONFIG.reward_formulations{f, 1};
    formulation_name_clean = clean_field_name(formulation_name);
    
    results = strategy_results.(formulation_name_clean);
    reward_rates = results(:, 3);
    
    % First 3 are fixed, rest are adaptive
    fixed_best = max(reward_rates(1:3));
    adaptive_best = max(reward_rates(4:end));
    advantage = (adaptive_best - fixed_best) / max(abs(fixed_best), 1e-6) * 100;
    
    formulation_comparison.names{f} = formulation_name;
    formulation_comparison.fixed_best(f) = fixed_best;
    formulation_comparison.adaptive_best(f) = adaptive_best;
    formulation_comparison.adaptive_advantage(f) = advantage;
end

end

function [hit_rate, reward_per_trial, reward_rate, mean_tilt] = ...
    simulate_strategy(CONFIG, strategy_func, reward_func)
% Simulate a single strategy with a single reward formulation

total_hits = 0;
total_trials = 0;
total_reward = 0;
total_tilt = 0;

for c = 1:length(CONFIG.coherence_levels)
    coh = CONFIG.coherence_levels(c);
    n_trials = CONFIG.n_trials_per_coherence;
    
    % Generate accuracy for this coherence
    [accuracy, ~, ~] = generate_accuracy_distribution(...
        coh, n_trials, CONFIG.psychometric);
    
    % Apply strategy to get tilt values
    tilt = zeros(n_trials, 1);
    for t = 1:n_trials
        tilt(t) = strategy_func(coh, accuracy(t));
        tilt(t) = max(0, min(1, tilt(t)));  % Clip to [0, 1]
    end
    
    % Calculate arc width
    arc_width = CONFIG.max_arc_deg - ...
        (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * tilt;
    
    % Determine hits (target hit when angular error < arc_width/2)
    angular_error = (1 - accuracy) * 180;
    hit = angular_error < (arc_width / 2);
    
    % Calculate rewards (pass CONFIG for scaling)
    reward = reward_func(tilt, hit, accuracy, CONFIG);
    
    % Accumulate statistics
    total_hits = total_hits + sum(hit);
    total_trials = total_trials + n_trials;
    total_reward = total_reward + sum(reward);
    total_tilt = total_tilt + sum(tilt);
end

hit_rate = total_hits / total_trials;
reward_per_trial = total_reward / total_trials;
reward_rate = reward_per_trial * CONFIG.targets_per_second;
mean_tilt = total_tilt / total_trials;

end

function [accuracy, mean_acc, std_acc] = generate_accuracy_distribution(...
    coherence, n_trials, psychometric_params)
% Generate accuracy distribution for given coherence using psychometric function

% Calculate mean accuracy from psychometric function (sigmoidal)
% At 0% coherence, should be exactly baseline (chance level)
if coherence == 0
    mean_acc = psychometric_params.baseline_acc;  % Exactly 0.5 at chance
else
    mean_acc = psychometric_params.baseline_acc + ...
        (psychometric_params.max_acc - psychometric_params.baseline_acc) / ...
        (1 + exp(-psychometric_params.slope * (coherence - psychometric_params.threshold)));
end

% Standard deviation decreases with coherence (less steep)
std_acc = psychometric_params.variability * sqrt(1 - coherence/100);
std_acc = max(std_acc, 0.02);  % Minimum floor to prevent too-tight distributions

% Generate samples using normal distribution (clipped to [0,1]) for symmetric distribution
accuracy = generate_normal_samples_clipped(mean_acc, std_acc, n_trials);

end

function samples = generate_normal_samples_clipped(mu, sigma, n)
% Generate normally-distributed random samples with specified mean and std
% Clipped to [0,1] to ensure valid accuracy values
% This creates symmetric distributions around the mean
% Note: Clipping may slightly shift the mean, but for mu near 0.5 and reasonable sigma, effect is minimal

% Generate normal samples
samples = normrnd(mu, sigma, n, 1);

% Clip to [0, 1] range
samples = max(0, min(1, samples));

% For 0% coherence (mu ≈ 0.5), ensure mean is exactly 0.5 by small adjustment
if abs(mu - 0.5) < 0.01
    actual_mean = mean(samples);
    if abs(actual_mean - 0.5) > 0.001
        % Small adjustment to center at 0.5
        samples = samples + (0.5 - actual_mean);
        samples = max(0, min(1, samples));  % Re-clip if needed
    end
end

end

function samples = generate_beta_samples(mu, sigma, n)
% Generate beta-distributed random samples with specified mean and std
% (Kept for reference, but not currently used)

% Ensure valid parameters
mu = max(0.01, min(0.99, mu));
sigma = min(sigma, sqrt(mu * (1 - mu)) * 0.99);

% Convert mean and std to beta distribution parameters
temp = (mu * (1 - mu) / sigma^2) - 1;
alpha = mu * temp;
beta_param = (1 - mu) * temp;

% Generate samples
samples = betarnd(alpha, beta_param, n, 1);

end

function reward_ml = scale_reward(normalized_reward, CONFIG)
% Scale normalized reward [0,1] to actual reward amount in ml
% Maps 0 -> reward_min_ml, 1 -> reward_max_ml

reward_ml = CONFIG.reward_min_ml + ...
    (CONFIG.reward_max_ml - CONFIG.reward_min_ml) * normalized_reward;

end

function display_results_table(CONFIG, strategy_results, formulation_comparison)
% Display comprehensive results table

n_formulations = length(formulation_comparison.names);

for f = 1:n_formulations
    formulation_name = formulation_comparison.names{f};
    formulation_name_clean = clean_field_name(formulation_name);
    
    fprintf('\n%s\n', formulation_name);
    fprintf('%s\n', repmat('─', 1, length(formulation_name)));
    
    results = strategy_results.(formulation_name_clean);
    
    fprintf('  %-20s | Hit Rate | Reward/Trial (ml) | Rate (ml/s) | Mean Tilt\n', 'Strategy');
    fprintf('  ---------------------|----------|-------------------|-------------|----------\n');
    
    for s = 1:size(CONFIG.strategies, 1)
        fprintf('  %-20s | %8.3f | %17.4f | %11.4f | %9.3f\n', ...
            CONFIG.strategies{s, 1}, results(s, 1), results(s, 2), ...
            results(s, 3), results(s, 4));
    end
    
    fprintf('  → Adaptive advantage: %.2f%%\n', ...
        formulation_comparison.adaptive_advantage(f));
end

% Summary comparison
fprintf('\n\nSUMMARY: Reward Formulation Rankings\n');
fprintf('═════════════════════════════════════════════════════════════════\n');
fprintf('  Rank | %-25s | Adaptive Advantage\n', 'Formulation');
fprintf('  -----|---------------------------|-------------------\n');

[~, sorted_idx] = sort(formulation_comparison.adaptive_advantage, 'descend');
for i = 1:n_formulations
    idx = sorted_idx(i);
    fprintf('  %4d | %-25s | %17.2f%%\n', i, ...
        formulation_comparison.names{idx}, ...
        formulation_comparison.adaptive_advantage(idx));
end

end

function insights = generate_insights(CONFIG, formulation_comparison)
% Generate insights and recommendations

insights = struct();

% Find best formulation
[max_advantage, best_idx] = max(formulation_comparison.adaptive_advantage);
insights.best_formulation = formulation_comparison.names{best_idx};
insights.best_advantage = max_advantage;

% Assess current formulation (assumed to be first)
insights.current_advantage = formulation_comparison.adaptive_advantage(1);

% Determine recommendation
if insights.current_advantage < 5
    insights.recommendation = 'CHANGE_REQUIRED';
elseif insights.current_advantage < 15
    insights.recommendation = 'CHANGE_SUGGESTED';
else
    insights.recommendation = 'CURRENT_OK';
end

% Count good formulations
insights.n_good_formulations = sum(formulation_comparison.adaptive_advantage > 15);
insights.n_moderate_formulations = sum(formulation_comparison.adaptive_advantage >= 5 & ...
    formulation_comparison.adaptive_advantage <= 15);

end

function display_insights(insights)
% Display insights and recommendations

fprintf('\n');
fprintf('┌─────────────────────────────────────────────────────────────┐\n');
fprintf('│ INSIGHTS & RECOMMENDATIONS                                  │\n');
fprintf('└─────────────────────────────────────────────────────────────┘\n');
fprintf('\n');

fprintf('Current formulation (Linear):\n');
fprintf('  Adaptive advantage: %.2f%%\n', insights.current_advantage);

if insights.current_advantage < 5
    fprintf('  ⚠️  WARNING: Minimal incentive for adaptive wagering\n');
elseif insights.current_advantage < 15
    fprintf('  ⚠️  CAUTION: Moderate incentive - may be insufficient\n');
else
    fprintf('  ✓ GOOD: Sufficient incentive for adaptive wagering\n');
end

fprintf('\nBest formulation: %s\n', insights.best_formulation);
fprintf('  Adaptive advantage: %.2f%%\n', insights.best_advantage);
fprintf('  Improvement over current: %.2f%%\n', ...
    insights.best_advantage - insights.current_advantage);

fprintf('\nRecommendation: ');
switch insights.recommendation
    case 'CHANGE_REQUIRED'
        fprintf('CHANGE REWARD FORMULATION\n');
        fprintf('  Current formulation provides insufficient incentive.\n');
        fprintf('  Monkey will likely use fixed tilt regardless of coherence.\n');
        
    case 'CHANGE_SUGGESTED'
        fprintf('CONSIDER CHANGING FORMULATION\n');
        fprintf('  Current formulation provides only moderate incentive.\n');
        fprintf('  Alternative formulations could improve learning.\n');
        
    case 'CURRENT_OK'
        fprintf('CURRENT FORMULATION ACCEPTABLE\n');
        fprintf('  Sufficient incentive for coherence-dependent wagering.\n');
end

fprintf('\nAlternative formulations with >15%% advantage: %d\n', ...
    insights.n_good_formulations);
fprintf('Alternative formulations with 5-15%% advantage: %d\n', ...
    insights.n_moderate_formulations);

end

function name_clean = clean_field_name(name)
% Convert formulation name to valid MATLAB field name

name_clean = strrep(name, ' ', '_');
name_clean = strrep(name_clean, '(', '');
name_clean = strrep(name_clean, ')', '');
name_clean = strrep(name_clean, '=', '');
name_clean = strrep(name_clean, '.', '');

end

%% ========================================================================
%  VISUALIZATION FUNCTIONS
%  ========================================================================

function create_figure1_psychometric(CONFIG, accuracy_data, stats)
% Figure 1: Accuracy and hit rate as function of coherence

fig = figure('Name', 'Accuracy and Hit Rate vs Coherence', 'Position', [100 100 1400 800]);
sgtitle('CPR Task: Accuracy and Hit Rate as Function of Coherence', ...
    'FontSize', 18, 'FontWeight', 'bold', 'Color', [0 0.3 0.6]);

n_coherence = length(CONFIG.coherence_levels);

% Subplot 1: Accuracy (0-1) vs Coherence
subplot(2, 3, 1);
plot(CONFIG.coherence_levels, stats.mean_acc, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0]);
hold on;
% Error bars showing trial-by-trial variability
errorbar(CONFIG.coherence_levels, stats.mean_acc, stats.std_acc, ...
    'LineStyle', 'none', 'LineWidth', 2.5, 'Color', [0 0 0], 'CapSize', 12);
% Chance line at 0.5 (random responding, 90° error)
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

% Subplot 2: Angular Error vs Coherence (inverse of accuracy)
subplot(2, 3, 2);
plot(CONFIG.coherence_levels, stats.mean_error_deg, '-s', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0]);
hold on;
% Chance level at 90°
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

% Subplot 3: Accuracy distributions (trial-by-trial variability)
subplot(2, 3, 3);
hold on;
colors = cool(n_coherence);  % Cool colormap for coherence levels
for i = 1:n_coherence
    [counts, edges] = histcounts(accuracy_data{i}, 30);
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    % Convert to probability density (normalize by bin width and total count)
    bin_width = edges(2) - edges(1);
    probability = counts / (sum(counts) * bin_width);  % Probability density
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

% Subplot 4: Hit Rate vs Tilt (shows which tilts work at each coherence)
subplot(2, 3, 4);
hold on;
tilt_range = 0:0.01:1;

for i = 1:n_coherence
    % Calculate hit rate for each tilt value using trial distribution
    trial_accuracies = accuracy_data{i};
    angular_errors = (1 - trial_accuracies) * 180;
    hit_prob = zeros(size(tilt_range));
    
    for t_idx = 1:length(tilt_range)
        arc_width = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * tilt_range(t_idx);
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

% Subplot 5: Average Hit Rate vs Coherence (for different fixed tilts)
% 
% Hit/Miss Determination:
% For each target presentation (trial):
%   1. Subject has an accuracy value (from psychometric distribution)
%   2. Convert to angular error: error = (1 - accuracy) × 180°
%   3. Arc width is set by tilt: arc_width = 180 - 170×tilt
%   4. Hit occurs if: angular_error < arc_width/2
%   5. Miss occurs if: angular_error ≥ arc_width/2
% Hit rate = proportion of trials where hit occurs
%
subplot(2, 3, 5);
hold on;
tilt_strategies = [0.2, 0.4, 0.6, 0.8];  % Different fixed tilts to test
% Use parula color scale for tilt strategies
tilt_colors = parula(length(tilt_strategies));

for t_idx = 1:length(tilt_strategies)
    tilt_val = tilt_strategies(t_idx);
    arc_width_val = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * tilt_val;
    
    hit_rates = zeros(n_coherence, 1);
    for i = 1:n_coherence
        % Calculate hit rate from trial-by-trial accuracy distribution
        trial_accuracies = accuracy_data{i};              % Get all trial accuracies
        angular_errors = (1 - trial_accuracies) * 180;    % Convert to angular error (degrees)
        hits = angular_errors < (arc_width_val / 2);      % HIT when error < half arc width
        hit_rates(i) = mean(hits);                        % Proportion of hits = hit rate
    end
    
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
tilt_strategies = [0.2, 0.4, 0.6, 0.8];  % Different fixed tilts to test
tilt_colors = parula(length(tilt_strategies));  % Parula colormap for tilt strategies

total_reward_per_tilt = zeros(length(tilt_strategies), 1);  % Store totals for dashed lines

for t_idx = 1:length(tilt_strategies)
    tilt_val = tilt_strategies(t_idx);
    arc_width_val = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * tilt_val;
    
    cumulative_reward = zeros(n_coherence, 1);
    for i = 1:n_coherence
        % Calculate hit rate from trial-by-trial accuracy distribution
        trial_accuracies = accuracy_data{i};
        angular_errors = (1 - trial_accuracies) * 180;
        hits = angular_errors < (arc_width_val / 2);
        n_hits = sum(hits);  % Number of hits across all trials
        
        % Calculate reward per hit (scaled from reward_min_ml to reward_max_ml)
        reward_per_hit_ml = CONFIG.reward_min_ml + ...
            (CONFIG.reward_max_ml - CONFIG.reward_min_ml) * tilt_val;
        
        % Cumulative reward = total reward across all trials at this coherence
        cumulative_reward(i) = n_hits * reward_per_hit_ml;
    end
    
    % Calculate total (sum) across all coherences for this tilt
    total_reward_per_tilt(t_idx) = sum(cumulative_reward);
    
    plot(CONFIG.coherence_levels, cumulative_reward, '-o', 'LineWidth', 3, ...
        'MarkerSize', 12, 'Color', tilt_colors(t_idx, :), ...
        'MarkerFaceColor', tilt_colors(t_idx, :));
    
    % Add horizontal dashed line showing total across all coherences
    xlim_current = xlim;
    plot(xlim_current, [total_reward_per_tilt(t_idx) total_reward_per_tilt(t_idx)], '--', ...
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

function create_figure2_coherence_strategy(CONFIG, accuracy_data, stats, adaptive_tilt_data)
% Figure 2: Comparison of coherence-dependent and adaptive wagering strategies
% Blue: Coherence-dependent (Tilt = 0.1 + 0.5×Coherence)
% Red: Adaptive wagering (Tilt = Accuracy + Gaussian noise)

fig = figure('Name', 'Strategy Comparison', 'Position', [100 100 1400 800]);
sgtitle('CPR Task: Coherence-Dependent vs Adaptive Wagering Strategies', ...
    'FontSize', 18, 'FontWeight', 'bold', 'Color', [0 0.3 0.6]);

n_coherence = length(CONFIG.coherence_levels);

% Define coherence-dependent tilt function: tilt = 0.1 + 0.5 * (coherence/100)
tilt_at_coherence = @(coh) 0.1 + 0.5 * (coh / 100);

% Subplot 1: Accuracy (0-1) vs Coherence (same as Figure 1)
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

% Subplot 2: Mean Tilt vs Coherence (shows the strategies)
subplot(2, 3, 2);
hold on;
% Coherence-dependent strategy (blue)
tilt_values_coherence = arrayfun(tilt_at_coherence, CONFIG.coherence_levels);
plot(CONFIG.coherence_levels, tilt_values_coherence, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0.4 0.8], 'MarkerFaceColor', [0 0.4 0.8], ...
    'DisplayName', 'Coherence-dependent');
% Adaptive wagering strategy (red) - mean tilt per coherence
mean_tilt_adaptive = zeros(n_coherence, 1);
for i = 1:n_coherence
    mean_tilt_adaptive(i) = mean(adaptive_tilt_data{i});
end
plot(CONFIG.coherence_levels, mean_tilt_adaptive, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0.8 0.2 0.2], 'MarkerFaceColor', [0.8 0.2 0.2], ...
    'DisplayName', 'Adaptive wagering');
xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Mean Tilt (0 to 1)', 'FontSize', 13, 'FontWeight', 'bold');
title('Mean Tilt vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10, 'Box', 'off');
grid on;
ylim([0 1]);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

% Subplot 3: Mean Arc Width vs Coherence
subplot(2, 3, 3);
hold on;
% Coherence-dependent strategy (blue)
arc_widths_coherence = zeros(n_coherence, 1);
for i = 1:n_coherence
    tilt_val = tilt_at_coherence(CONFIG.coherence_levels(i));
    arc_widths_coherence(i) = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * tilt_val;
end
plot(CONFIG.coherence_levels, arc_widths_coherence, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0.4 0.8], 'MarkerFaceColor', [0 0.4 0.8], ...
    'DisplayName', 'Coherence-dependent');
% Adaptive wagering strategy (red) - mean arc width per coherence
mean_arc_width_adaptive = zeros(n_coherence, 1);
for i = 1:n_coherence
    mean_tilt = mean(adaptive_tilt_data{i});
    mean_arc_width_adaptive(i) = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * mean_tilt;
end
plot(CONFIG.coherence_levels, mean_arc_width_adaptive, '-o', ...
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
% Coherence-dependent strategy (blue)
hit_rates_coherence = zeros(n_coherence, 1);
for i = 1:n_coherence
    tilt_val = tilt_at_coherence(CONFIG.coherence_levels(i));
    arc_width_val = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * tilt_val;
    
    trial_accuracies = accuracy_data{i};
    angular_errors = (1 - trial_accuracies) * 180;
    hits = angular_errors < (arc_width_val / 2);
    hit_rates_coherence(i) = mean(hits);
end
plot(CONFIG.coherence_levels, hit_rates_coherence, '-o', 'LineWidth', 3, ...
    'MarkerSize', 14, 'Color', [0 0.4 0.8], 'MarkerFaceColor', [0 0.4 0.8], ...
    'DisplayName', 'Coherence-dependent');
% Adaptive wagering strategy (red)
hit_rates_adaptive = zeros(n_coherence, 1);
for i = 1:n_coherence
    trial_accuracies = accuracy_data{i};
    trial_tilts = adaptive_tilt_data{i};
    angular_errors = (1 - trial_accuracies) * 180;
    
    % Calculate hits trial-by-trial (each trial has its own tilt/arc width)
    hits = zeros(size(trial_accuracies));
    for t = 1:length(trial_accuracies)
        arc_width_val = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts(t);
        hits(t) = angular_errors(t) < (arc_width_val / 2);
    end
    hit_rates_adaptive(i) = mean(hits);
end
plot(CONFIG.coherence_levels, hit_rates_adaptive, '-o', 'LineWidth', 3, ...
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
% Coherence-dependent strategy (blue)
reward_per_hit_coherence = zeros(n_coherence, 1);
for i = 1:n_coherence
    tilt_val = tilt_at_coherence(CONFIG.coherence_levels(i));
    reward_per_hit_coherence(i) = CONFIG.reward_min_ml + ...
        (CONFIG.reward_max_ml - CONFIG.reward_min_ml) * tilt_val;
end
plot(CONFIG.coherence_levels, reward_per_hit_coherence, '-o', ...
    'LineWidth', 3, 'MarkerSize', 14, 'Color', [0 0.4 0.8], 'MarkerFaceColor', [0 0.4 0.8], ...
    'DisplayName', 'Coherence-dependent');
% Adaptive wagering strategy (red) - mean reward per hit
mean_reward_per_hit_adaptive = zeros(n_coherence, 1);
for i = 1:n_coherence
    trial_tilts = adaptive_tilt_data{i};
    mean_tilt = mean(trial_tilts);
    mean_reward_per_hit_adaptive(i) = CONFIG.reward_min_ml + ...
        (CONFIG.reward_max_ml - CONFIG.reward_min_ml) * mean_tilt;
end
plot(CONFIG.coherence_levels, mean_reward_per_hit_adaptive, '-o', ...
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
% Coherence-dependent strategy (blue)
cumulative_reward_coherence = zeros(n_coherence, 1);
for i = 1:n_coherence
    tilt_val = tilt_at_coherence(CONFIG.coherence_levels(i));
    arc_width_val = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * tilt_val;
    
    trial_accuracies = accuracy_data{i};
    angular_errors = (1 - trial_accuracies) * 180;
    hits = angular_errors < (arc_width_val / 2);
    n_hits = sum(hits);
    
    reward_per_hit_ml = CONFIG.reward_min_ml + ...
        (CONFIG.reward_max_ml - CONFIG.reward_min_ml) * tilt_val;
    
    cumulative_reward_coherence(i) = n_hits * reward_per_hit_ml;
end
total_reward_coherence = sum(cumulative_reward_coherence);
plot(CONFIG.coherence_levels, cumulative_reward_coherence, '-o', 'LineWidth', 3, ...
    'MarkerSize', 14, 'Color', [0 0.4 0.8], 'MarkerFaceColor', [0 0.4 0.8], ...
    'DisplayName', 'Coherence-dependent');
xlim_current = xlim;
plot(xlim_current, [total_reward_coherence total_reward_coherence], '--', ...
    'LineWidth', 2, 'Color', [0 0.4 0.8]);

% Adaptive wagering strategy (red)
cumulative_reward_adaptive = zeros(n_coherence, 1);
for i = 1:n_coherence
    trial_accuracies = accuracy_data{i};
    trial_tilts = adaptive_tilt_data{i};
    angular_errors = (1 - trial_accuracies) * 180;
    
    % Calculate reward trial-by-trial
    total_reward_coh = 0;
    for t = 1:length(trial_accuracies)
        arc_width_val = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * trial_tilts(t);
        is_hit = angular_errors(t) < (arc_width_val / 2);
        if is_hit
            reward_per_hit_ml = CONFIG.reward_min_ml + ...
                (CONFIG.reward_max_ml - CONFIG.reward_min_ml) * trial_tilts(t);
            total_reward_coh = total_reward_coh + reward_per_hit_ml;
        end
    end
    cumulative_reward_adaptive(i) = total_reward_coh;
end
total_reward_adaptive = sum(cumulative_reward_adaptive);
plot(CONFIG.coherence_levels, cumulative_reward_adaptive, '-o', 'LineWidth', 3, ...
    'MarkerSize', 14, 'Color', [0.8 0.2 0.2], 'MarkerFaceColor', [0.8 0.2 0.2], ...
    'DisplayName', 'Adaptive wagering');
plot(xlim_current, [total_reward_adaptive total_reward_adaptive], '--', ...
    'LineWidth', 2, 'Color', [0.8 0.2 0.2]);

xlabel('Coherence (%)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Cumulative Reward (ml)', 'FontSize', 13, 'FontWeight', 'bold');
title('Cumulative Reward vs Coherence', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10, 'Box', 'off');
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box on;
hold off;

end

function create_figure3_accuracy_tilt_scatter(CONFIG, accuracy_data, adaptive_tilt_data)
% Figure 3: Scatter plots of accuracy vs tilt for each coherence level
% Shows linear regression and uses cool colormap (same as Figure 1)

fig = figure('Name', 'Accuracy vs Tilt by Coherence', 'Position', [100 100 1400 800]);
sgtitle('CPR Task: Accuracy vs Tilt Relationship (Adaptive Wagering)', ...
    'FontSize', 18, 'FontWeight', 'bold', 'Color', [0 0.3 0.6]);

n_coherence = length(CONFIG.coherence_levels);
colors = cool(n_coherence);  % Same color scale as Figure 1

% Create subplots for each coherence level
for i = 1:n_coherence
    subplot(2, 2, i);
    hold on;
    
    trial_accuracies = accuracy_data{i};
    trial_tilts = adaptive_tilt_data{i};
    
    % Scatter plot (with some transparency for better visibility)
    scatter(trial_accuracies, trial_tilts, 20, colors(i, :), 'filled', ...
        'MarkerFaceAlpha', 0.3, 'MarkerEdgeAlpha', 0.3);
    
    % Linear regression
    p = polyfit(trial_accuracies, trial_tilts, 1);
    x_fit = linspace(min(trial_accuracies), max(trial_accuracies), 100);
    y_fit = polyval(p, x_fit);
    plot(x_fit, y_fit, '-', 'LineWidth', 3, 'Color', colors(i, :), ...
        'DisplayName', sprintf('y=%.3fx+%.3f', p(1), p(2)));
    
    % Calculate R-squared
    y_pred = polyval(p, trial_accuracies);
    ss_res = sum((trial_tilts - y_pred).^2);
    ss_tot = sum((trial_tilts - mean(trial_tilts)).^2);
    r_squared = 1 - (ss_res / ss_tot);
    
    xlabel('Accuracy', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Tilt', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('%d%% Coherence (R²=%.3f)', CONFIG.coherence_levels(i), r_squared), ...
        'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0 1]);
    ylim([0 1]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    box on;
    hold off;
end

end

function create_figure2_strategy_comparison(CONFIG, strategy_results, formulation_comparison)
% Figure 2: Strategy comparison across reward formulations

figure('Name', 'Strategy Comparison', 'Position', [120 80 1600 900]);

n_strategies = size(CONFIG.strategies, 1);
n_formulations = length(formulation_comparison.names);

% Subplot 1: Reward rates by formulation
subplot(2, 3, 1);
hold on;
colors = lines(n_strategies);
x_pos = 1:n_formulations;

for s = 1:n_strategies
    y_vals = zeros(n_formulations, 1);
    for f = 1:n_formulations
        formulation_name_clean = clean_field_name(formulation_comparison.names{f});
        results = strategy_results.(formulation_name_clean);
        y_vals(f) = results(s, 3);  % reward_rate
    end
    plot(x_pos, y_vals, '-o', 'LineWidth', 2, 'Color', colors(s, :), ...
        'DisplayName', CONFIG.strategies{s, 1}, 'MarkerSize', 8);
end

xlabel('Reward Formulation');
ylabel('Reward Rate (ml/s)');
title('Reward Rates by Strategy & Formulation');
set(gca, 'XTick', x_pos, 'XTickLabel', 1:n_formulations, 'FontSize', 10);
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;

% Subplot 2: Adaptive advantage comparison
subplot(2, 3, 2);
bar(formulation_comparison.adaptive_advantage);
hold on;
yline(0, 'k--', 'LineWidth', 1.5);
yline(10, 'g--', 'LineWidth', 1.5, 'Label', 'Good (>10%)');
xlabel('Formulation #');
ylabel('Adaptive Advantage (%)');
title('Adaptive Over Fixed Strategy');
set(gca, 'XTick', 1:n_formulations, 'FontSize', 10);
grid on;

% Color code bars
for f = 1:n_formulations
    adv = formulation_comparison.adaptive_advantage(f);
    if adv > 15
        color = [0 0.7 0];  % Green
    elseif adv > 5
        color = [0.9 0.6 0];  % Orange
    else
        color = [0.8 0.2 0.2];  % Red
    end
    bar(f, adv, 'FaceColor', color);
end

% Subplot 3: Best formulation detailed
[~, best_idx] = max(formulation_comparison.adaptive_advantage);
subplot(2, 3, 3);
best_name_clean = clean_field_name(formulation_comparison.names{best_idx});
best_results = strategy_results.(best_name_clean);

bar(best_results(:, 3));
set(gca, 'XTickLabel', cellfun(@(x) x(1:min(12,end)), ...
    CONFIG.strategies(:,1), 'UniformOutput', false), ...
    'XTickLabelRotation', 45, 'FontSize', 9);
ylabel('Reward Rate (ml/s)');
title(sprintf('Best: %s', formulation_comparison.names{best_idx}), ...
    'Interpreter', 'none');
grid on;

% Highlight adaptive strategies
hold on;
for s = 4:n_strategies
    bar(s, best_results(s, 3), 'FaceColor', [0.2 0.7 0.2]);
end

% Subplot 4: Hit rates comparison
subplot(2, 3, 4);
hold on;
for s = 1:n_strategies
    y_vals = zeros(n_formulations, 1);
    for f = 1:n_formulations
        formulation_name_clean = clean_field_name(formulation_comparison.names{f});
        results = strategy_results.(formulation_name_clean);
        y_vals(f) = results(s, 1);  % hit_rate
    end
    plot(x_pos, y_vals, '-o', 'LineWidth', 2, 'Color', colors(s, :), ...
        'MarkerSize', 6);
end
xlabel('Formulation #');
ylabel('Hit Rate');
title('Hit Rates Across Formulations');
grid on;
set(gca, 'FontSize', 10);

% Subplot 5: Mean tilt usage
subplot(2, 3, 5);
hold on;
for s = 1:n_strategies
    y_vals = zeros(n_formulations, 1);
    for f = 1:n_formulations
        formulation_name_clean = clean_field_name(formulation_comparison.names{f});
        results = strategy_results.(formulation_name_clean);
        y_vals(f) = results(s, 4);  % mean_tilt
    end
    plot(x_pos, y_vals, '-o', 'LineWidth', 2, 'Color', colors(s, :), ...
        'MarkerSize', 6);
end
xlabel('Formulation #');
ylabel('Mean Tilt');
title('Tilt Usage Across Formulations');
grid on;
set(gca, 'FontSize', 10);

% Subplot 6: Summary ranking
subplot(2, 3, 6);
axis off;
text_x = 0.05;
text_y = 0.95;
line_height = 0.08;

text(text_x, text_y, 'FORMULATION RANKINGS', 'FontSize', 12, 'FontWeight', 'bold');
text_y = text_y - line_height * 1.3;

[~, sorted_idx] = sort(formulation_comparison.adaptive_advantage, 'descend');
for i = 1:min(6, n_formulations)
    idx = sorted_idx(i);
    adv = formulation_comparison.adaptive_advantage(idx);
    
    if adv > 15
        marker = '✓✓';
        col = [0 0.6 0];
    elseif adv > 5
        marker = '✓ ';
        col = [0.8 0.5 0];
    else
        marker = '  ';
        col = [0.7 0 0];
    end
    
    text(text_x, text_y, sprintf('%s %d. %s', marker, i, ...
        formulation_comparison.names{idx}), ...
        'FontSize', 9, 'Color', col, 'Interpreter', 'none');
    text_y = text_y - line_height * 0.7;
    
    text(text_x + 0.05, text_y, sprintf('Advantage: %.1f%%', adv), ...
        'FontSize', 8, 'Color', col * 0.8);
    text_y = text_y - line_height;
end

end

function create_figure3_detailed_analysis(CONFIG, strategy_results)
% Figure 3: Detailed breakdown by coherence for key formulations

figure('Name', 'Detailed Analysis', 'Position', [140 60 1400 800]);

% Focus on Linear (current) and best alternative
formulation_names_clean = fieldnames(strategy_results);
linear_results = strategy_results.(formulation_names_clean{1});  % Assumed first

% Subplot 1: Fixed strategies comparison
subplot(2, 3, 1);
fixed_indices = 1:3;
bar(linear_results(fixed_indices, 3));
set(gca, 'XTickLabel', cellfun(@(x) x(1:min(10,end)), ...
    CONFIG.strategies(fixed_indices,1), 'UniformOutput', false), ...
    'XTickLabelRotation', 0, 'FontSize', 10);
ylabel('Reward Rate (ml/s)');
title('Fixed Tilt Strategies (Linear Reward)');
grid on;

% Add values on bars
for i = 1:3
    text(i, linear_results(i, 3), sprintf('  %.4f', linear_results(i, 3)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 9);
end

% Subplot 2: Hit rate vs tilt tradeoff
subplot(2, 3, 2);
scatter(linear_results(:, 4), linear_results(:, 1), 100, 1:size(CONFIG.strategies, 1), 'filled');
colormap(lines);
xlabel('Mean Tilt');
ylabel('Hit Rate');
title('Hit Rate vs Tilt Usage');
grid on;
set(gca, 'FontSize', 10);

% Add strategy labels
for s = 1:size(CONFIG.strategies, 1)
    text(linear_results(s, 4), linear_results(s, 1), ...
        sprintf('  %d', s), 'FontSize', 8);
end

% Subplot 3: Reward per trial vs hit rate
subplot(2, 3, 3);
scatter(linear_results(:, 1), linear_results(:, 2), 100, 1:size(CONFIG.strategies, 1), 'filled');
xlabel('Hit Rate');
ylabel('Reward per Trial (ml)');
title('Reward Structure');
grid on;
set(gca, 'FontSize', 10);

% Subplot 4: Strategy labels
subplot(2, 3, 4);
axis off;
text_x = 0.1;
text_y = 0.9;
line_height = 0.08;

text(text_x, text_y, 'STRATEGY KEY', 'FontSize', 11, 'FontWeight', 'bold');
text_y = text_y - line_height * 1.3;

for s = 1:size(CONFIG.strategies, 1)
    text(text_x, text_y, sprintf('%d. %s', s, CONFIG.strategies{s, 1}), ...
        'FontSize', 9);
    text_y = text_y - line_height;
end

% Subplot 5: Efficiency metric
subplot(2, 3, 5);
efficiency = linear_results(:, 3) ./ max(linear_results(:, 3));
bar(efficiency);
set(gca, 'XTickLabel', 1:size(CONFIG.strategies, 1), 'FontSize', 10);
ylabel('Relative Efficiency');
title('Strategy Efficiency (Normalized)');
grid on;
ylim([0 1.1]);

% Highlight best
[~, best_s] = max(efficiency);
hold on;
bar(best_s, efficiency(best_s), 'FaceColor', [0 0.7 0]);

% Subplot 6: Interpretation
subplot(2, 3, 6);
axis off;
text_x = 0.05;
text_y = 0.95;
line_height = 0.08;

text(text_x, text_y, 'INTERPRETATION', 'FontSize', 11, 'FontWeight', 'bold');
text_y = text_y - line_height * 1.5;

[best_reward, best_idx] = max(linear_results(:, 3));
best_strategy = CONFIG.strategies{best_idx, 1};

text(text_x, text_y, sprintf('Best strategy: %s', best_strategy), ...
    'FontSize', 9, 'Interpreter', 'none');
text_y = text_y - line_height;

text(text_x, text_y, sprintf('Reward rate: %.4f ml/s', best_reward), ...
    'FontSize', 9);
text_y = text_y - line_height * 1.5;

% Check if best is fixed or adaptive
if best_idx <= 3
    text(text_x, text_y, '⚠️  Best strategy is FIXED', ...
        'FontSize', 9, 'Color', [0.8 0.2 0], 'FontWeight', 'bold');
    text_y = text_y - line_height;
    text(text_x, text_y, 'Monkey will not modulate', ...
        'FontSize', 8, 'Color', [0.8 0.2 0]);
    text_y = text_y - line_height;
    text(text_x, text_y, 'tilt with coherence!', ...
        'FontSize', 8, 'Color', [0.8 0.2 0]);
else
    text(text_x, text_y, '✓ Best strategy is ADAPTIVE', ...
        'FontSize', 9, 'Color', [0 0.6 0], 'FontWeight', 'bold');
    text_y = text_y - line_height;
    text(text_x, text_y, 'Good incentive structure', ...
        'FontSize', 8, 'Color', [0 0.6 0]);
end

end

function create_figure4_reward_surfaces(CONFIG)
% Figure 4: Reward surface visualization for different formulations

figure('Name', 'Reward Surfaces', 'Position', [160 40 1400 800]);

tilt_range = 0:0.02:1;
acc_range = 0:0.02:1;
[Tilt, Acc] = meshgrid(tilt_range, acc_range);

% Calculate arc width
Arc_width = CONFIG.max_arc_deg - (CONFIG.max_arc_deg - CONFIG.min_arc_deg) * Tilt;

% Calculate angular error
Angular_error = (1 - Acc) * 180;

% Determine hits
Hit = double(Angular_error < Arc_width / 2);

% Plot surfaces for different reward formulations
n_to_plot = min(6, size(CONFIG.reward_formulations, 1));

for f = 1:n_to_plot
    subplot(2, 3, f);
    
    reward_func = CONFIG.reward_formulations{f, 2};
    formulation_name = CONFIG.reward_formulations{f, 1};
    
    % Calculate reward surface (pass CONFIG for scaling)
    Reward = reward_func(Tilt, Hit, Acc, CONFIG);
    
    % Plot
    surf(Tilt, Acc, Reward, 'EdgeColor', 'none');
    view(45, 30);
    xlabel('Tilt');
    ylabel('Accuracy');
    zlabel('Reward (ml)');
    title(formulation_name, 'Interpreter', 'none', 'FontSize', 10);
    colormap(jet);
    colorbar;
    set(gca, 'FontSize', 9);
    
    % Add contour on bottom
    hold on;
    contour3(Tilt, Acc, Reward, 10, 'k', 'LineWidth', 0.5);
end

end

