clear;
% --- Paths ---
input_path = "C:\Users\aaron\OneDrive\Desktop\DDF 2025\EEG Data Subject-1\EEG Data Subject-1";
output_path = "C:\Users\aaron\OneDrive\Desktop\DDF 2025\Test Subject Preprocessed";
if ~exist(output_path, 'dir')
    mkdir(output_path);
end
Files = dir(fullfile(input_path, '*.mat'));

% --- Bandpass Filter Design ---
OSR = 250;
p = 1;
q = 2;
d = fdesign.bandpass(8.1, 8.4, 12, 12.3, 40, 0.2, 40, OSR);
Hd = design(d);

% --- Process Each File ---
for i = 1:length(Files)
    filename = Files(i).name;
    full_path = fullfile(input_path, filename);
    raw = load(full_path);
    Data = raw.data_received;
    
    % Generate templates - now returns combined template of size [14 × 625 × 40]
    Combined_Template = template_generation_40(Data, Hd, p, q);
    
    % Save the combined template
    save(fullfile(output_path, strcat("Processed_", filename)), 'Combined_Template');
    
    fprintf('Processed %s: Template size [%d × %d × %d]\n', ...
        filename, size(Combined_Template, 1), size(Combined_Template, 2), size(Combined_Template, 3));
end

%% ----------- Template Generation Function for 40 Templates -----------
function Combined_Template = template_generation_40(data_received_d1, Hd, p, q)
    % Extract 14 EEG channels only
    Data_1 = data_received_d1(1:45000, 3:16); % shape: [45000 × 14]
    
    % Extract data for each condition (40s each at 250 Hz = 10000 samples)
    EC_Data = Data_1(1:10000, :)';         % 0-40s
    EO_Data = Data_1(15001:25000, :)';     % 60-100s
    MI_Data = Data_1(30001:40000, :)';     % 120-160s
    
    % Preprocess each condition
    [~, DS_EC] = PreProcessing(EC_Data, Hd, p, q);  % [14 × 5000]
    [~, DS_EO] = PreProcessing(EO_Data, Hd, p, q);  % [14 × 5000]
    [~, DS_MI] = PreProcessing(MI_Data, Hd, p, q);  % [14 × 5000]
    
    % Now we need to create ~40 templates total
    % Each condition has 5000 samples (40s at 125 Hz)
    % We'll use overlapping windows to create more templates
    
    window_length = 625;  % 5 seconds at 125 Hz
    
    % For 40 templates total, we want about 13-14 per condition
    % We'll use 50% overlap to create 14 templates per condition
    overlap = 312;  % ~50% overlap
    step = window_length - overlap;
    
    all_templates = [];
    
    % Process EC templates
    num_templates_EC = 0;
    for start_idx = 1:step:(size(DS_EC, 2) - window_length + 1)
        end_idx = start_idx + window_length - 1;
        template = DS_EC(:, start_idx:end_idx);  % [14 × 625]
        all_templates = cat(3, all_templates, template);
        num_templates_EC = num_templates_EC + 1;
        if num_templates_EC >= 14
            break;
        end
    end
    
    % Process EO templates
    num_templates_EO = 0;
    for start_idx = 1:step:(size(DS_EO, 2) - window_length + 1)
        end_idx = start_idx + window_length - 1;
        template = DS_EO(:, start_idx:end_idx);  % [14 × 625]
        all_templates = cat(3, all_templates, template);
        num_templates_EO = num_templates_EO + 1;
        if num_templates_EO >= 13
            break;
        end
    end
    
    % Process MI templates
    num_templates_MI = 0;
    for start_idx = 1:step:(size(DS_MI, 2) - window_length + 1)
        end_idx = start_idx + window_length - 1;
        template = DS_MI(:, start_idx:end_idx);  % [14 × 625]
        all_templates = cat(3, all_templates, template);
        num_templates_MI = num_templates_MI + 1;
        if num_templates_MI >= 13
            break;
        end
    end
    
    % Ensure we have exactly 40 templates
    if size(all_templates, 3) > 40
        all_templates = all_templates(:, :, 1:40);
    elseif size(all_templates, 3) < 40
        % If we have fewer than 40, duplicate some templates
        num_to_add = 40 - size(all_templates, 3);
        indices = randi(size(all_templates, 3), [1, num_to_add]);
        all_templates = cat(3, all_templates, all_templates(:, :, indices));
    end
    
    Combined_Template = all_templates;  % [14 × 625 × 40]
    
    fprintf('  Created %d EC + %d EO + %d MI = %d total templates\n', ...
        num_templates_EC, num_templates_EO, num_templates_MI, size(Combined_Template, 3));
end

%% ----------- PreProcessing Function -----------
function [filtered_EEG, Downsampled_EEG] = PreProcessing(temp_EEG, Hd, p, q)
    % Bandpass filtering (zero-phase)
    temp_EEG = filtfilt(Hd.Numerator, 1, temp_EEG')';
    
    % Z-score normalisation (per channel)
    for ch = 1:size(temp_EEG, 1)
        temp_EEG(ch, :) = zscore(temp_EEG(ch, :));
    end
    
    % Detrending
    temp_EEG = detrend(temp_EEG')';
    
    % Downsample from 250 Hz to 125 Hz
    Downsampled_EEG = resample(temp_EEG', p, q, 50)'; % final shape: [14 × 5000]
    filtered_EEG = temp_EEG;
end