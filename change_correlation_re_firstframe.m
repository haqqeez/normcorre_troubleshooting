

crop = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get a list of all files in the current directory
files = dir;

% Initialize an empty cell array to store the video names
videoNames = {};

% Loop over the files
for i = 1:length(files)
    % If the file is a .avi video, add its name to the list
    if endsWith(files(i).name, '.avi')
        videoNames{end+1} = files(i).name;
    end
end

% Convert the cell array to a string array
videoNames = string(videoNames)


if crop == true
    cropped_region = select_crop_region(videoNames(1), 300);
end

% Initialize a cell array to store the correlation values for all videos
allCorrelationValues = cell(1, length(videoNames));

% Loop over the video files
for i = 1:length(videoNames)
    % Load the video
    v = VideoReader(videoNames{i});

    % Read the first frame
    v.CurrentTime = 0; % Ensure we're at the start of the video
    referenceFrame = readFrame(v);

    % Initialize an array to store the correlation values
    correlationValues = [];

    % Loop over the rest of the frames
    while hasFrame(v)
        currentFrame = readFrame(v);

        % Convert frames to grayscale
        referenceFrameGray = im2gray(referenceFrame);
        currentFrameGray = im2gray(currentFrame);

        if crop == true
%           disp('cropping')
            referenceFrameGray = imcrop(referenceFrameGray, cropped_region);
            currentFrameGray = imcrop(currentFrameGray, cropped_region);
        end

        % Calculate the correlation with the reference frame
        correlation = corr2(referenceFrameGray, currentFrameGray);

        % Store the correlation value
        correlationValues = [correlationValues, correlation];
    end

    % Store the correlation values for this video
    allCorrelationValues{i} = correlationValues;
    disp('')
end

% Define markers for the plot
markers = {'o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};

% Plot the correlation values for all videos
figure;
hold on;
for i = 1:length(videoNames)
    plot(allCorrelationValues{i}, 'LineWidth', 2.5);
end
hold off;
xlabel('Frame number');
ylabel('Correlation with first frame');
legend(videoNames);

% Plot the derivative of the correlation values for all videos
figure;
hold on;
for i = 1:length(videoNames)
    plot(diff(allCorrelationValues{i}), 'LineWidth', 2); % Calculate and plot the derivative
end
hold off;
xlabel('Frame number');
ylabel('Change in correlation with first frame');
legend(videoNames);

% Plot the kernel density of the change in correlation values for all videos
figure;
hold on;
for i = 1:length(videoNames)
    [f,xi] = ksdensity(allCorrelationValues{i}); % Calculate the kernel density
    plot(xi,f, 'LineWidth', 2); % Plot the kernel density
end
hold off;
xlabel('Correlation with first frame');
ylabel('Density');
legend(videoNames);

% Plot the kernel density of the change in correlation values for all videos
figure;
hold on;
for i = 1:length(videoNames)
    [f,xi] = ksdensity(diff(allCorrelationValues{i})); % Calculate the kernel density
    plot(xi,f, 'LineWidth', 2); % Plot the kernel density
end
hold off;
xlabel('Change in correlation with first frame');
ylabel('Density');
legend(videoNames);