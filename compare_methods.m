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

cropped_region = select_crop_region(videoNames(1), 10);

% First figure
figure;
hold on;

% Loop over video names
for i = 1:length(videoNames)
    % Calculate correlation matrix and frame differences
    [differences, ~] = calculate_correlation_matrix(videoNames{i}, 1, true, cropped_region);

    % Create the KDE
    [f, x] = ksdensity(differences);

    % Plot the KDE without specifying a color
    plot(x, f, 'LineWidth', 2);
    hold on;
end

% Add a legend with video names without ".avi"
legend(erase(videoNames, '.avi'));
legend(erase(videoNames, 'msvideo'));

% Add title and labels
title('Kernel Density Estimation of Mean Frame diff to every other frame');
xlabel('Mean Frame Difference');
ylabel('Density');

% Display the plot
hold off;

% Second figure
figure;
hold on;

% Loop over video names
for i = 1:length(videoNames)
    % Calculate frame differences
    d = calculate_frame_differences(videoNames{i}, 1, true, cropped_region);

    % Create the KDE
    [f, x] = ksdensity(d);

    % Plot the KDE without specifying a color
    plot(x, f, 'LineWidth', 2);
    hold on;
end

% Add a legend with video names without ".avi"
legend(erase(videoNames, '.avi'));
legend(erase(videoNames, 'msvideo'));

% Add title and labels
title('Kernel Density Estimation of Frame Differences');
xlabel('Frame Difference');
ylabel('Density');

% Display the plot
hold off;
