function frame_diffs = calculate_array_frame_differences(frames,n)
    % Initialize frame counter
    frame_counter = 1;

    % Initialize difference array
    frame_diffs = [];

    % Initialize previous frame
    prev_frame = [];

    % Get the number of frames
    T = size(frames, 3);

    for t = 1:T
        % Get the current frame
        curr_frame = frames(:,:,t);
        
        % Check if it's the nth frame
        if mod(frame_counter, n) == 0
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
    figure;
    plot(frame_diffs);
    title('Differences Between Frames Over Time');
    xlabel('Frame');
    ylabel('Difference');
end
