% Gemini 2.5 Pro
% --- Simulation Parameters ---
totalDuration_s = 10 * 60;       % 10 minutes in seconds
fs = 100;                      % Sampling rate in Hz
nSamples = totalDuration_s * fs; % Total number of samples
time_s = (0:nSamples-1)' / fs;   % Time vector

% Stimulus Parameters
stimChangeMin_s = 1.5;           % Minimum duration for a stimulus direction
stimChangeMax_s = 3.0;           % Maximum duration for a stimulus direction
stimMaxAngleChange_deg = 90;     % Max change in degrees between directions

% Response Parameters
respAvgChange_s = 2.0;           % Average duration for a response direction
respStdChange_s = 0.5;           % Standard deviation for response duration
minRespDuration_s = 0.1;         % Minimum response hold duration in seconds

% Repetition Parameters
nRepetitions = 1000;

% --- Check for Circular Statistics Toolbox ---
if exist('circ_dist', 'file') ~= 2
    error(['Circular Statistics Toolbox not found or not added to the MATLAB path. ', ...
           'Download from: https://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics']);
end

% --- 1. Generate Stimulus Vector (Same as before) ---
fprintf('Generating stimulus...\n');
stimulus_deg = zeros(nSamples, 1);
currentIndex = 1;
currentDir_deg = rand() * 360; % Initial random direction

while currentIndex <= nSamples
    duration_s = stimChangeMin_s + (stimChangeMax_s - stimChangeMin_s) * rand();
    nSegmentSamples = round(duration_s * fs);
    endIndex = min(currentIndex + nSegmentSamples - 1, nSamples);
    stimulus_deg(currentIndex:endIndex) = currentDir_deg;
    change_deg = (rand() * 2 - 1) * stimMaxAngleChange_deg;
    nextDir_deg = mod(currentDir_deg + change_deg, 360);
    currentDir_deg = nextDir_deg;
    currentIndex = endIndex + 1;
end
fprintf('Stimulus generation complete.\n');

% Convert stimulus to radians once for calculations
stimulus_rad = deg2rad(stimulus_deg); % Size: nSamples x 1

% --- 2. & 3. Generate Responses and Calculate Accuracy (Vectorized Approach) ---

% Define a function handle for generating a single response trace
generate_single_response = @(~) generate_response_trace(nSamples, fs, respAvgChange_s, respStdChange_s, minRespDuration_s);

fprintf('Generating %d responses using cellfun...\n', nRepetitions);
% Use cellfun to generate all responses. Each cell will contain one response vector.
% 'UniformOutput', false is necessary because the output of the function is a vector, not a scalar.
response_cells = cellfun(generate_single_response, num2cell(1:nRepetitions), 'UniformOutput', false);
fprintf('Response generation complete.\n');

% Concatenate the cell array of responses into a single matrix
% Each column corresponds to one repetition. Size: nSamples x nRepetitions
all_responses_deg = cat(2, response_cells{:});

% Clear the cell array to save memory if needed
% clear response_cells; 

fprintf('Calculating accuracies using matrix operations...\n');
% Convert all responses to radians. Size: nSamples x nRepetitions
all_responses_rad = deg2rad(all_responses_deg);

% Calculate circular distance between the stimulus vector and *each column*
% of the response matrix. MATLAB's implicit expansion (broadcasting) should handle this.
% circ_dist(stimulus_rad, all_responses_rad) compares the nSamples x 1 vector
% element-wise against each column of the nSamples x nRepetitions matrix.
% Result is an nSamples x nRepetitions matrix of angular differences in radians.
angular_differences_rad_matrix = abs(circ_dist(stimulus_rad, all_responses_rad));

% Calculate the mean difference *down each column* (dimension 1)
% Result is a 1 x nRepetitions row vector of mean differences in radians.
mean_diffs_rad = mean(angular_differences_rad_matrix, 1);

% Convert mean differences to degrees and transpose to a column vector
% Size: nRepetitions x 1
allAccuracies_deg = rad2deg(mean_diffs_rad)'; 

fprintf('Accuracy calculation complete.\n');

% Calculate Grand Mean Accuracy
grandMeanAccuracy_deg = mean(allAccuracies_deg);
fprintf('Grand Mean Accuracy (Average Angular Difference): %.2f degrees\n', grandMeanAccuracy_deg);

% --- 4. Plotting (Same as before, but use one column from the matrix) ---

% Use the last generated response (last column) as the example
example_response_deg = all_responses_deg(:, end); 

% Plot Stimulus and Example Response
figure;
plot(time_s, stimulus_deg, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Stimulus');
hold on;
plot(time_s, example_response_deg, 'r-', 'LineWidth', 1.0, 'DisplayName', 'Example Response');
hold off;
ylim([0 360]);
yticks(0:45:360);
xlabel('Time (s)');
ylabel('Direction (degrees)');
title('Stimulus and Example Response Directions Over Time');
legend('show');
grid on;

% Plot Distribution of Accuracies
figure;
histogram(allAccuracies_deg, 50); % Use 50 bins, adjust as needed
hold on;
xline(grandMeanAccuracy_deg, 'k--', 'LineWidth', 2, ...
      'Label', sprintf('Grand Mean: %.2f deg', grandMeanAccuracy_deg), ...
      'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
hold off;
xlabel('Average Accuracy (Mean Angular Difference, degrees)');
ylabel('Frequency (Number of Repetitions)');
title(sprintf('Distribution of Average Accuracies over %d Repetitions (Vectorized)', nRepetitions));
grid on;

fprintf('Plotting complete.\n');

% --- Helper Function for Response Generation ---
function response_deg = generate_response_trace(nSamples, fs, respAvgChange_s, respStdChange_s, minRespDuration_s)
    % Generates a single random response vector based on normal distribution durations
    response_deg = zeros(nSamples, 1);
    currentIndex = 1;
    currentRespDir_deg = rand() * 360; % Initial random response direction

    while currentIndex <= nSamples
        % Determine duration for this response direction (Normal distribution)
        duration_s = normrnd(respAvgChange_s, respStdChange_s);
        % Ensure duration is positive and reasonably long
        duration_s = max(duration_s, minRespDuration_s); 
        nSegmentSamples = round(duration_s * fs);
        
        % Ensure we don't exceed total samples
        endIndex = min(currentIndex + nSegmentSamples - 1, nSamples);
        
        % Assign current response direction to the segment
        response_deg(currentIndex:endIndex) = currentRespDir_deg;
        
        % Determine the next *random* response direction
        nextRespDir_deg = rand() * 360; 
        
        % Update for next iteration
        currentRespDir_deg = nextRespDir_deg;
        currentIndex = endIndex + 1;
    end
end

% % --- Simulation Parameters ---
% totalDuration_s = 10 * 60;       % 10 minutes in seconds
% fs = 100;                      % Sampling rate in Hz
% nSamples = totalDuration_s * fs; % Total number of samples
% time_s = (0:nSamples-1)' / fs;   % Time vector
% 
% % Stimulus Parameters
% stimChangeMin_s = 1.5;           % Minimum duration for a stimulus direction
% stimChangeMax_s = 3.0;           % Maximum duration for a stimulus direction
% stimMaxAngleChange_deg = 90;     % Max change in degrees between directions
% 
% % Response Parameters
% respAvgChange_s = 2.0;           % Average duration for a response direction
% respStdChange_s = 0.5;           % Standard deviation for response duration (adjust as needed)
% % Ensure minimum reasonable duration to avoid issues with very small std dev
% minRespDuration_s = 0.1;         % Minimum response hold duration in seconds
% 
% % Repetition Parameters
% nRepetitions = 10000;
% 
% % --- Check for Circular Statistics Toolbox ---
% if exist('circ_dist', 'file') ~= 2
%     error(['Circular Statistics Toolbox not found or not added to the MATLAB path. ', ...
%            'Download from: https://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics']);
% end
% 
% % --- 1. Generate Stimulus Vector ---
% fprintf('Generating stimulus...\n');
% stimulus_deg = zeros(nSamples, 1);
% currentIndex = 1;
% currentDir_deg = rand() * 360; % Initial random direction
% 
% while currentIndex <= nSamples
%     % Determine duration for this direction
%     duration_s = stimChangeMin_s + (stimChangeMax_s - stimChangeMin_s) * rand();
%     nSegmentSamples = round(duration_s * fs);
%     
%     % Ensure we don't exceed total samples
%     endIndex = min(currentIndex + nSegmentSamples - 1, nSamples);
%     
%     % Assign current direction to the segment
%     stimulus_deg(currentIndex:endIndex) = currentDir_deg;
%     
%     % Determine the next direction
%     change_deg = (rand() * 2 - 1) * stimMaxAngleChange_deg; % Change between -max and +max
%     nextDir_deg = mod(currentDir_deg + change_deg, 360); % Wrap around 0/360
%     
%     % Update for next iteration
%     currentDir_deg = nextDir_deg;
%     currentIndex = endIndex + 1;
% end
% fprintf('Stimulus generation complete.\n');
% 
% % Convert stimulus to radians once for calculations
% stimulus_rad = deg2rad(stimulus_deg);
% 
% % --- 2. & 3. Generate Responses and Calculate Accuracy ---
% fprintf('Generating %d responses and calculating accuracies...\n', nRepetitions);
% allAccuracies_deg = zeros(nRepetitions, 1);
% example_response_deg = zeros(nSamples, 1); % To store one example response
% 
% for i = 1:nRepetitions
%     response_deg = zeros(nSamples, 1);
%     currentIndex = 1;
%     currentRespDir_deg = rand() * 360; % Initial random response direction
% 
%     while currentIndex <= nSamples
%         % Determine duration for this response direction (Normal distribution)
%         duration_s = normrnd(respAvgChange_s, respStdChange_s);
%         % Ensure duration is positive and reasonably long
%         duration_s = max(duration_s, minRespDuration_s); 
%         nSegmentSamples = round(duration_s * fs);
%         
%         % Ensure we don't exceed total samples
%         endIndex = min(currentIndex + nSegmentSamples - 1, nSamples);
%         
%         % Assign current response direction to the segment
%         response_deg(currentIndex:endIndex) = currentRespDir_deg;
%         
%         % Determine the next *random* response direction (no constraint on change)
%         nextRespDir_deg = rand() * 360; 
%         
%         % Update for next iteration
%         currentRespDir_deg = nextRespDir_deg;
%         currentIndex = endIndex + 1;
%     end
%     
%     % Store the last response as an example for plotting
%     if i == nRepetitions
%         example_response_deg = response_deg;
%     end
%     
%     % Convert current response to radians
%     response_rad = deg2rad(response_deg);
%     
%     % Calculate circular distance (element-wise) -> Result is in radians
%     % circ_dist calculates the shortest angle between two circular data points
%     angular_differences_rad = abs(circ_dist(stimulus_rad, response_rad));
%     
%     % Calculate the average difference for this repetition
%     % Note: Taking the mean of shortest angles. Another metric could be circ_mean(angular_differences_rad)
%     % but the mean of magnitudes is often more intuitive for "average error".
%     mean_diff_rad = mean(angular_differences_rad); 
%     
%     % Store the mean difference in degrees
%     allAccuracies_deg(i) = rad2deg(mean_diff_rad);
% 
%     % Optional: Progress indicator
%     if mod(i, 100) == 0
%         fprintf('  Completed %d / %d repetitions...\n', i, nRepetitions);
%     end
% end
% fprintf('Response generation and accuracy calculation complete.\n');
% 
% % Calculate Grand Mean Accuracy
% grandMeanAccuracy_deg = mean(allAccuracies_deg);
% fprintf('Grand Mean Accuracy (Average Angular Difference): %.2f degrees\n', grandMeanAccuracy_deg);
% 
% % --- 4. Plotting ---
% 
% % Plot Stimulus and Example Response
% figure;
% plot(time_s, stimulus_deg, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Stimulus');
% hold on;
% plot(time_s, example_response_deg, 'r-', 'LineWidth', 1.0, 'DisplayName', 'Example Response');
% hold off;
% ylim([0 360]);
% yticks(0:45:360);
% xlabel('Time (s)');
% ylabel('Direction (degrees)');
% title('Stimulus and Example Response Directions Over Time');
% legend('show');
% grid on;
% 
% % Plot Distribution of Accuracies
% figure;
% histogram(allAccuracies_deg, 50); % Use 50 bins, adjust as needed
% hold on;
% xline(grandMeanAccuracy_deg, 'k--', 'LineWidth', 2, ...
%       'Label', sprintf('Grand Mean: %.2f deg', grandMeanAccuracy_deg), ...
%       'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
% hold off;
% xlabel('Average Accuracy (Mean Angular Difference, degrees)');
% ylabel('Frequency (Number of Repetitions)');
% title(sprintf('Distribution of Average Accuracies over %d Repetitions', nRepetitions));
% grid on;
% 
% fprintf('Plotting complete.\n');