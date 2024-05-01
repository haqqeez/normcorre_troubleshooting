function diff_distribution = frame_difference_distribution(video_file, n, crop, crop_region)
    % Read the video file
    v = VideoReader(video_file);

    % Initialize a variable to store all frames
    frames = [];

    % Initialize a frame counter
    frame_counter = 0;

    % Read every nth frame
    while hasFrame(v)
        frame = readFrame(v);
        frame_counter = frame_counter + 1;
        if mod(frame_counter, n) == 0
            % Convert the frame to grayscale
%             frame = rgb2gray(frame);
            frame = im2gray(frame);
            % Convert the frame to double
            frame = double(frame);

            if crop == true
                frame = imcrop(frame, crop_region);
            end

            % Flatten the frame and store it
            frames = [frames; frame(:)'];
        end
    end

    % Calculate the mean frame
    mean_frame = mean(frames, 1);
    disp(size(mean_frame))

    % Initialize a variable to store the mean differences
    mean_diffs = [];

    % Loop through each frame
    for i = 1:size(frames, 1)
        % Calculate the difference of the frame from the mean frame
%         diff_frame = frames(i, :) - mean_frame;
        diff_frame = immse(mean_frame, frames(i, :));
        % Calculate the mean of the difference
%         mean_diff = mean(diff_frame);
        mean_diff = diff_frame;
        % Store the mean difference
        mean_diffs = [mean_diffs; mean_diff];
    end

    % Return the distribution of mean differences
    diff_distribution = mean_diffs;
end
