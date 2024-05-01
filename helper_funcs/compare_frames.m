function diff_image = compare_frames(video_file, frame1_num, frame2_num)
    % Create a video reader object
    vidObj = VideoReader(video_file);

    % Read the two specified frames
    frame1 = double(read(vidObj, frame1_num));
    frame2 = double(read(vidObj, frame2_num));

    % Convert the frames to grayscale if necessary
    if size(frame1, 3) == 3
        frame1 = rgb2gray(frame1);
    end
    if size(frame2, 3) == 3
        frame2 = rgb2gray(frame2);
    end

    % Compute the difference image
    diff_image = abs(frame1 - frame2);

    % Display the difference image
    imagesc(diff_image);
    clim([0 0.001]);  % Adjust the colorbar range
    colorbar;
end
