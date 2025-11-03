clear; clc; close all;

%% paths and setup stuff
% define where our data lives and where to save results
data_path = "C:\Users\aaron\OneDrive\Desktop\DDF 2025\Test Subject Preprocessed";
results_path = "C:\Users\aaron\OneDrive\Desktop\DDF 2025\Biometric_Results";

% make results folder if it doesnt exist
if ~exist(results_path, 'dir')
    mkdir(results_path);
end

% random seed so we get same results each time
% important for reproducibility in research
rng(42);

%% Load and organize files 

% get all the processed files
% dir() returns struct array with file info - name, folder, date, etc
% fullfile() combines path parts with proper slashes for any OS
all_files = dir(fullfile(data_path, 'Processed_*.mat'));

% storage for organizing everything
subject_names = {};     % all unique subject names we find
session1_files = {};    % training session files (S1)
session2_files = {};    % test session files (S2)

% map names to numbers for neural network (like aaron = 0, jannon = 1, etc)
% containers.Map is matlab's dictionary/hashmap implementation
name_to_number = containers.Map();

% go through each file and organize them
for i = 1:length(all_files)
    
    filename = all_files(i).name;
    
    % extract name and session number from filename using regex
    % pattern: Processed_data_received_AARON_S1.mat
    % tokens gives us the parts in parentheses - (.+) and (\d)
    % .+ means one or more of any character (the name)
    % \d means a digit (the session number)
    tokens = regexp(filename, 'Processed_data_received_(.+)_S(\d)\.mat', 'tokens');
    
    % skip if filename doesnt match our expected pattern
    if isempty(tokens)
        continue;
    end
    
    % extract the matched groups
    person_name = tokens{1}{1};             % first capture group - the name
    session_num = str2double(tokens{1}{2}); % second capture group - session number
    
    % add new person if we havent seen them before
    % strcmp compares strings, returns 1 if match, 0 if not
    % any() checks if any element in array is true/nonzero
    if ~any(strcmp(subject_names, person_name))
        subject_names{end+1} = person_name;
        % assign them the next available number
        name_to_number(person_name) = length(subject_names);
    end
    
    % store file info in a struct for easy access later
    % struct holds multiple fields like a mini database entry
    file_info = struct(...
        'subject', person_name, ...
        'file', fullfile(data_path, filename), ...  % full path to file
        'label', name_to_number(person_name) - 1 ... % zero indexed labels for nn
    );
    
    % sort into session 1 or 2 based on filename
    if session_num == 1
        session1_files{end+1} = file_info;
    else
        session2_files{end+1} = file_info;
    end
end

% get total unique subjects found
total_subjects = length(subject_names);
fprintf('Found %d subjects\n', total_subjects);

%% Load session 1 data WITHOUT augmentation
% session 1 is used for training and validation

% will be 4d array: [channels × time × 1 × num_templates]
% channels = 14 eeg channels
% time = 625 samples (5 seconds at 125hz after downsampling)
% 1 = grayscale channel (cnn expects 3d input minimum, like images)
% num_templates = total number across all subjects
all_templates = [];
all_labels = [];     % subject label for each template (0,1,2,etc)

% load each persons session 1 data
for i = 1:length(session1_files)
    
    % load the mat file - contains preprocessed eeg data
    data = load(session1_files{i}.file);
    
    % get their label number (0-indexed for neural network)
    subject_label = session1_files{i}.label;
    
    % make sure the data has what we need
    % isfield checks if struct has a specific field/variable
    if ~isfield(data, 'Combined_Template')
        continue;
    end
    
    % get the templates - should be 14×625×40
    % 14 channels, 625 time points, 40 five-second segments
    % (14 EC + 13 EO + 13 MI templates with 50% overlap)
    templates = data.Combined_Template;
    
    % Original templates only - NO augmentation
    % we're using raw data to see baseline performance
    for t = 1:size(templates, 3)
        
        % extract one 14×625 template
        single_template = templates(:, :, t);
        
        % reshape for cnn - needs to be 14×625×1
        % adds a singleton dimension for "color channel"
        % like converting grayscale image to proper format
        template_3d = reshape(single_template, 14, 625, 1);
        
        % add to our big array
        % cat(4,...) concatenates along 4th dimension (samples)
        all_templates = cat(4, all_templates, template_3d);
        
        % remember which person this template belongs to
        all_labels(end+1, 1) = subject_label;
    end
end

% convert numeric labels to categorical for neural network
% matlab's nn toolbox needs categorical not just numbers
labels_categorical = categorical(all_labels);

fprintf("Got %d templates from session 1 (no augmentation)\n", length(all_labels));

%% Split data into train and validation sets

% cvpartition creates cross-validation partition object
% 'HoldOut' means single split (not k-fold cross validation)
% 0.2 = hold out 20% for validation, use 80% for training
cv = cvpartition(length(all_labels), 'HoldOut', 0.2);

% training() and test() return logical indices (true/false arrays)
% true where that sample should go to that set
train_idx = training(cv);  % 80% will be true
val_idx = test(cv);        % 20% will be true

% split the data using logical indexing
% X_train will be about 80% of templates (around 800 samples)
X_train = all_templates(:, :, :, train_idx);
y_train = labels_categorical(train_idx);

% X_val will be about 20% for checking during training (around 200 samples)
X_val = all_templates(:, :, :, val_idx);
y_val = labels_categorical(val_idx);

%% Load session 2 for testing
% completely separate from training, like real world test
% this simulates identifying someone on a different day

X_test = [];
y_test = [];

for i = 1:length(session2_files)
    
    data = load(session2_files{i}.file);
    subject_label = session2_files{i}.label;
    
    if ~isfield(data, 'Combined_Template')
        continue;
    end
    
    templates = data.Combined_Template;
    
    % get all templates from this person's second session
    for t = 1:size(templates, 3)
        
        single_template = templates(:, :, t);
        template_3d = reshape(single_template, 14, 625, 1);
        
        X_test = cat(4, X_test, template_3d);
        y_test(end+1, 1) = subject_label;
    end
end

fprintf("Got %d templates from session 2\n", length(y_test));

%% Optimized CNN Architecture WITHOUT GAP (Global Average Pooling)
% this architecture progressively reduces dimensions through convolutions
% instead of using gap, we reduce to [1×1×channels] then flatten

layers = [
    % Input layer - expects [14 × 625 × 1] eeg segments
    % normalization disabled because data is already z-scored in preprocessing
    imageInputLayer([14 625 1], 'Name', 'input', ...
        'Normalization', 'none')
    
    % Conv1: [5×15] as specified by professor
    % processes 5 channels × 15 time samples
    % Output: [10 × 611 × 32] (14-5+1 × 625-15+1)
    convolution2dLayer([5 15], 32, 'Padding', [0 0], 'Name', 'conv1', ...
        'WeightsInitializer', 'he')  % he initialization good for relu
    batchNormalizationLayer('Name', 'bn1')  % normalize activations
    reluLayer('Name', 'relu1')               % non-linearity
    
    % Pool1: [2×2] as specified
    % reduces both spatial and temporal dimensions
    % Output: [5 × 305 × 32] (floor(10/2) × floor(611/2))
    maxPooling2dLayer([2 2], 'Stride', [2 2], 'Name', 'pool1')
    
    % Conv2: [5×15] as specified
    % continues processing spatial-temporal features
    % Output: [1 × 291 × 64] (5-5+1 × 305-15+1)
    convolution2dLayer([5 15], 64, 'Padding', [0 0], 'Name', 'conv2', ...
        'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    % Pool2: [1×2] with stride [1×2] as specified
    % only reduces temporal dimension
    % Output: [1 × 145 × 64] (1 × floor((291-2)/2)+1)
    maxPooling2dLayer([1 2], 'Stride', [1 2], 'Name', 'pool2')
    
    % Conv3: [1×15] as specified
    % temporal convolution on flattened spatial dimension
    % Output: [1 × 131 × 128] (1 × 145-15+1)
    convolution2dLayer([1 15], 128, 'Padding', [0 0], 'Name', 'conv3', ...
        'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    % dropout after main feature extraction for regularization
    % prevents overfitting
    dropoutLayer(0.3, 'Name', 'spatial_dropout')
    
    % Additional layers to reach [1×1]
    % Pool3: Aggressive reduction
    % Output: [1 × 26 × 128] (floor(131/5))
    maxPooling2dLayer([1 5], 'Stride', [1 5], 'Name', 'pool3')
    
    % Conv4: Further temporal reduction
    % Output: [1 × 14 × 256] (26-13+1)
    convolution2dLayer([1 13], 256, 'Padding', [0 0], 'Name', 'conv4', ...
        'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    
    % Conv5: Final reduction to [1 × 1 × 512]
    % [1×14] kernel exactly matches remaining temporal dimension
    % Output: [1 × 1 × 512] (14-14+1 = 1)
    convolution2dLayer([1 14], 512, 'Padding', [0 0], 'Name', 'conv5_final', ...
        'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    
    % Flatten: [1 × 1 × 512] → [512]
    % converts 3d tensor to 1d vector for fully connected layers
    flattenLayer('Name', 'flatten')
    
    % FC layers - following your structure
    % FC1: First fully connected layer
    fullyConnectedLayer(256, 'Name', 'fc1', ...
        'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn_fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.5, 'Name', 'dropout1')  % 50% dropout for regularization
    
    % FC2: Second fully connected layer
    fullyConnectedLayer(128, 'Name', 'fc2', ...
        'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn_fc2')
    reluLayer('Name', 'relu_fc2')
    dropoutLayer(0.4, 'Name', 'dropout2')  % slightly less dropout
    
    % Output layer: one neuron per subject
    fullyConnectedLayer(total_subjects, 'Name', 'fc_output', ...
        'WeightsInitializer', 'glorot')  % glorot good for output layer
    
    % softmax converts outputs to probabilities (sum to 1)
    softmaxLayer('Name', 'softmax')
    
    % classification layer computes cross-entropy loss
    classificationLayer('Name', 'classification')
];

%% Dimension tracking summary for reference
% Layer                    Output Shape         Details
% --------------------------------------------------------
% Input                    [14 × 625 × 1]       Raw EEG input
% Conv1 [5×15]            [10 × 611 × 32]      Professor's spec
% Pool1 [2×2]             [5 × 305 × 32]       Professor's spec
% Conv2 [5×15]            [1 × 291 × 64]       Professor's spec
% Pool2 [1×2] s[1×2]      [1 × 145 × 64]       Professor's spec
% Conv3 [1×15]            [1 × 131 × 128]      Professor's spec
% Pool3 [1×5] s[1×5]      [1 × 26 × 128]       Additional
% Conv4 [1×13]            [1 × 14 × 256]       Additional
% Conv5 [1×14]            [1 × 1 × 512]        Final reduction
% Flatten                 [512]                Ready for FC
% FC1                     [256]                With BN & Dropout
% FC2                     [128]                With BN & Dropout
% Output                  [total_subjects]     Classification

%% Optimized Training Options for limited data (no augmentation)
% with only ~1000 samples for 25 subjects, need careful hyperparameters

opts = trainingOptions('adam', ...           % adam optimizer - adaptive learning rates
    'InitialLearnRate', 0.0001, ...          % lower lr for limited data (was 0.0001)
    'LearnRateSchedule', 'piecewise', ...    % drop lr at specific epochs
    'LearnRateDropPeriod', 100, ...           % less frequent drops (was 100)
    'LearnRateDropFactor', 0.1, ...          % gentler drops (was 0.1 = too harsh)
    'MaxEpochs', 700, ...                    % more epochs since less data variety
    'MiniBatchSize',8, ...                 % smaller batch for ~1000 samples (was 8)
    'Shuffle', 'every-epoch', ...            % randomize order each epoch
    'ValidationData', {X_val, y_val}, ...    % check performance on val set
    'ValidationFrequency', 10, ...           % validate every 10 iterations
    'ValidationPatience', Inf, ...            % stop if val loss doesn't improve for 40 checks
    'Plots', 'training-progress', ...        % show live training plot
    'Verbose', true, ...                     % print progress to command window
    'L2Regularization', 0.0001, ...           % l2 penalty on weights to prevent overfitting
    'GradientThreshold', 1, ...              % clip gradients to prevent explosion
    'ExecutionEnvironment', 'auto');         % use gpu if available, else cpu

%% Train the main network

fprintf("\n=== Training main network ===\n");

% tic starts a timer to measure training time
tic;

% trainNetwork does the actual training
% takes data, labels, architecture, and options
% returns trained network and training history info
[trained_net, training_info] = trainNetwork(X_train, y_train, layers, opts);

% toc gets elapsed time since tic
time_taken = toc;
fprintf("Training completed in %.2f minutes\n", time_taken/60);

%% Test on session 2
% evaluate how well the model generalizes to new data

fprintf("\n=== Testing on session 2 ===\n");

% classify runs the trained network on new data
% returns predicted class for each sample
predictions = classify(trained_net, X_test);

% convert categorical back to numbers for comparison
% subtract 1 because we use 0-indexed labels
pred_numbers = double(predictions) - 1;

% template level accuracy
% how many individual 5-second segments were correctly classified
template_accuracy = mean(pred_numbers == y_test) * 100;
fprintf("Template accuracy: %.2f%%\n", template_accuracy);

%% Subject level accuracy with majority voting
% each person has 40 templates so we use voting
% more realistic for biometric system - identify person not just segment

correct_subjects = 0;
subject_confidences = zeros(total_subjects, 1);

% print header for results table
fprintf("\nPer-subject results:\n");
fprintf("%-15s %8s %10s %8s %10s\n", "Subject", "Actual", "Predicted", "Result", "Confidence");
fprintf("%s\n", repmat('-', 60, 1));  % print line of dashes

% check each subject's results
for i = 1:total_subjects
    
    % find all test templates for this person
    % find() returns indices where condition is true
    person_idx = find(y_test == i-1);
    
    % handle case where person has no test data (shouldn't happen)
    if isempty(person_idx)
        fprintf("%-15s %8d %10s %8s %10s\n", subject_names{i}, i-1, "N/A", "?", "N/A");
        continue;
    end
    
    % get all predictions for this person's templates
    person_preds = pred_numbers(person_idx);
    
    % majority voting - find most common prediction
    % unique() gets unique values and their indices
    % accumarray() counts occurrences of each unique value
    [unique_preds, ~, idx_unique] = unique(person_preds);
    counts = accumarray(idx_unique, 1);
    
    % find which prediction occurred most often
    [max_count, max_idx] = max(counts);
    majority_prediction = unique_preds(max_idx);
    
    % confidence = percentage of templates that agreed with majority
    confidence = max_count / length(person_preds) * 100;
    subject_confidences(i) = confidence;
    
    % check if we got it right
    is_correct = (majority_prediction == i-1);
    
    if is_correct
        mark = '✓';  % checkmark for correct
        correct_subjects = correct_subjects + 1;
    else
        mark = '✗';  % x for incorrect
    end
    
    % print this subject's results
    fprintf("%-15s %8d %10d %8s %9.1f%%\n", ...
        subject_names{i}, i-1, majority_prediction, mark, confidence);
end

% calculate and print overall accuracy
subject_accuracy = correct_subjects / total_subjects * 100;
fprintf("%s\n", repmat('-', 60, 1));
fprintf("Overall subject accuracy: %.2f%% (%d/%d)\n", ...
    subject_accuracy, correct_subjects, total_subjects);
fprintf("Average confidence: %.1f%%\n", mean(subject_confidences));

%% Enhanced confusion matrix and visualizations

% confusionmat builds matrix of actual vs predicted
% rows = actual class, columns = predicted class
confusion = confusionmat(y_test, pred_numbers);

% create figure with subplots for multiple visualizations
figure('Position', [100 100 900 700]);

imagesc(confusion);     % Display matrix as image
colormap(hot);          % Use 'hot' colormap (dark = low, bright = high)
colorbar;               % Show colour scale
axis square;            % Make the plot square
title('Confusion Matrix'); % Optional: Add a title
xlabel('Predicted');
ylabel('Actual');


title(sprintf('Template-Level Confusion Matrix - %d subjects', total_subjects));
xlabel('Predicted');
ylabel('Actual');

% add grid for easier reading
grid on;
set(gca, 'GridColor', 'w', 'GridAlpha', 0.3);

% add text annotations for better readability
% only if we have 20 or fewer subjects (otherwise too crowded)
if total_subjects <= 20
    for i = 1:total_subjects
        for j = 1:total_subjects
            % choose text color based on background brightness
            % white text on dark background, black on light
            if confusion(i,j) < max(confusion(:))/2
                text_color = 'w';
            else
                text_color = 'k';
            end
            
            % add count number to each cell
            text(j, i, num2str(confusion(i,j)), ...
                'HorizontalAlignment', 'center', ...
                'Color', text_color);
        end
    end
end

% % subplot 2: training progress curves
% subplot(2, 2, 3);
% plot(training_info.TrainingAccuracy);
% hold on;
% plot(training_info.ValidationAccuracy);
% xlabel('Iteration');
% ylabel('Accuracy (%)');
% title('Training Progress');
% legend('Training', 'Validation');
% grid on;

% % subplot 3: per-subject confidence scores
% subplot(2, 2, 4);
% bar(subject_confidences);
% xlabel('Subject ID');
% ylabel('Confidence (%)');
% title('Per-Subject Classification Confidence');
% ylim([0 105]);  % set y limits with small margin
% grid on;

%% Save results for later analysis

% save everything important to a mat file
save(fullfile(results_path, 'biometric_results.mat'), ...
    'trained_net', ...        % the trained model
    'training_info', ...      % training history
    'confusion', ...          % confusion matrix
    'template_accuracy', ...  % segment-level accuracy
    'subject_accuracy', ...   % person-level accuracy  
    'subject_names');         % name mapping

fprintf("\nResults saved to: %s\n", fullfile(results_path, 'biometric_results.mat'));