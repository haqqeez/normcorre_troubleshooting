function frame_diffs = calculate_frame_differences(video_file, n, crop, crop_region)
    % Load the video file
    v = VideoReader(video_file);

    % Initialize frame array
    frames = [];

    % Initialize frame counter
    frame_counter = 1;

    % Initialize difference array
    frame_diffs = [];

    % Initialize previous frame
    prev_frame = [];

    while hasFrame(v)
        % Read the next frame
        curr_frame = readFrame(v);
        
        % Convert the frame to grayscale
%         curr_frame = rgb2gray(curr_frame);
        curr_frame = im2gray(curr_frame);
        
        % Check if it's the nth frame
        if mod(frame_counter, n) == 0

            if crop == true
                curr_frame = imcrop(curr_frame, crop_region);
            end
        
            % Add the frame to the array
            frames = cat(3, frames, curr_frame);
            
            % Calculate the difference with the previous frame
            if ~isempty(prev_frame)
                diff = mean2(abs(double(curr_frame) - double(prev_frame)));
                frame_diffs = [frame_diffs, diff];
            end
            
            % Update the previous frame
            prev_frame = curr_frame;
        end
        
        % Increment the frame counter
        frame_counter = frame_counter + 1;
    end

    % Plot the differences between frames over time
%     figure;
%     plot(frame_diffs);
%     title('Differences Between Frames Over Time (uncorrected)');
%     xlabel('Frame');
%     ylabel('Difference');
end
