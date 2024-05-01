function crop_region = select_crop_region(video_file, frame_number)
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
            imshow(frame);
            hold on;

            % Draw a rectangle to indicate the crop region
            h = drawrectangle;
            
            % Wait for the user to finish drawing the rectangle
            wait(h);
            
            % Get the position of the rectangle
            crop_region = h.Position;
            
            % Stop the loop
            break;
        end
    end
end