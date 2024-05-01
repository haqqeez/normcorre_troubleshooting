function [lower_matrix_values,correlation_matrix] = calculate_correlation_matrix(video_file, n, crop, crop_region)

%     if nargin < 3
%         crop_region = [];
%     end
    % Load the video file
    v = VideoReader(video_file);

    % Initialize frame array
    frames = [];

    % Initialize frame counter
    frame_counter = 1;

    while hasFrame(v)
        % Read the next frame
        curr_frame = readFrame(v);
        
        % Convert the frame to grayscale
%         curr_frame = rgb2gray(curr_frame);
        curr_frame = im2gray(curr_frame);

        if crop == true
%             disp('cropping')
            curr_frame = imcrop(curr_frame, crop_region);
        end
        
        % Check if it's the nth frame
        if mod(frame_counter, n) == 0
            % Add the frame to the array
            frames = cat(3, frames, curr_frame);
        end
        
        % Increment the frame counter
        frame_counter = frame_counter + 1;
    end

    % Initialize correlation matrix
%     imshow(curr_frame)
    num_frames = size(frames, 3);
    disp(num_frames)
    correlation_matrix = zeros(num_frames, num_frames);

    % Calculate the lower triangular matrix
    for i = 1:num_frames
        for j = i:num_frames
            % Compute the correlation
%             corr_value = mean2(abs(double(frames(:,:,i)) - double(frames(:,:,j))));
            corr_value = corr2(double(frames(:,:,i)), double(frames(:,:,j)));
            correlation_matrix(i, j) = corr_value;
        end
    end

    % Duplicate to the upper triangular matrix to get full matrix
    correlation_matrix = correlation_matrix + correlation_matrix' - diag(diag(correlation_matrix));

    % Return the lower triangular matrix, excluding the diagonal
    lower_matrix = tril(correlation_matrix, -1);

    % Get the linear indices of the lower triangular part of the matrix
    [row, col] = find(tril(true(size(correlation_matrix)), -1));
    
    % Use these indices to extract the values from the lower triangular matrix
    lower_matrix_values = lower_matrix(sub2ind(size(lower_matrix), row, col));
    
    % Replace the diagonal with NaN
    correlation_matrix(1:size(correlation_matrix, 1)+1:end) = NaN;
end
