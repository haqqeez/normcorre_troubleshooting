function plot_cropped_frame(video_file, crop_region, frame_number)
    % Load the video
    v = VideoReader(video_file);

    % Initialize a variable to store the current frame number
    current_frame_number = 0;

    while hasFrame(v)
        % Read the next frame
        frame = readFrame(v);
        
        % Increment the current frame number
        current_frame_number = current_frame_number + 1;
        
        % If this is the frame of choice
        if current_frame_number == frame_number
            % Display the original frame
            figure;
            imshow(frame);
            hold on;

            % Draw a rectangle to indicate the crop region
            rectangle('Position', crop_region, 'EdgeColor', 'r', 'LineWidth', 2);
            
            % Stop the loop
            break;
        end
    end
end