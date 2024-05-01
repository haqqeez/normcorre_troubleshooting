function output_frame = subtract_frames_from_video(video_file, n)
    % Create a video reader object
    vidObj = VideoReader(video_file);

    % Initialize an empty 3D matrix for the grayscale frames
    frames = zeros(vidObj.Height, vidObj.Width, ceil(vidObj.NumFrames/n));

    % Read each nth frame, convert it to grayscale if necessary, and store it in the 3D matrix
    j = 1;
    for i = 1:n:vidObj.NumFrames
        frame = double(read(vidObj, i));
        if size(frame, 3) == 3  % If the frame is color (3D)
            try
                frame = rgb2gray(frame);  % Try to convert it to grayscale (2D) using rgb2gray
            catch
                frame = im2gray(frame);  % If rgb2gray fails, try to convert it using im2gray
            end
        end
        frames(:,:,j) = frame;
        j = j + 1;
    end

    % Initialize the output frame as the first frame
    output_frame = frames(:,:,1);

    % Subtract each subsequent frame from the current output frame
    for i = 2:size(frames, 3)
        output_frame = output_frame + frames(:,:,i);
    end
end
