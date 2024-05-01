function plot_mean_of_nth_frames(video_file, n)
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
            % Flatten the frame and store it
            frames = [frames; frame(:)'];
        end
    end

    % Calculate the mean frame
    mean_frame = mean(frames, 1);

    % Reshape the mean frame to its original size
    mean_frame = reshape(mean_frame, [v.Height, v.Width]);

    % Create a new figure
    figure;
    
    % Adjust the aspect ratio to make the figure square
    axis square;

    % Plot the mean frame using imagesc
    imagesc(mean_frame);
    colormap gray;
end
