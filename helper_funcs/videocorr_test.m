% % Load the video file
% v = VideoReader('1.avi');
% 
% % Initialize previous frame
% prev_frame = readFrame(v);
% 
% % Initialize correlation array
% correlation = [];
% 
% % Initialize frame counter
% frame_counter = 1;
% 
% while hasFrame(v)
%     % Read the next frame
%     curr_frame = readFrame(v);
%     
%     % Check if it's an even frame
%     if mod(frame_counter, 2) == 0
%         % Calculate the difference
%         diff_frame = double(prev_frame) - double(curr_frame);
%         
%         % Compute the correlation
%         %corr_value = corr2(prev_frame, curr_frame);
%         correlation = [correlation; diff_frame];
%     end
%     
%     % Update the previous frame
%     prev_frame = curr_frame;
%     
%     % Increment the frame counter
%     frame_counter = frame_counter + 1;
% end
% 
% % Display the correlation values
% disp(correlation);


    % Create a VideoWriter object to write the video out to a new file
    v_out = VideoWriter('output.avi');
    open(v_out);
    
    % Write the frames to the new video file
    for i = 1:num_frames
        writeVideo(v_out, frames(:,:,i));
    end
    
    % Close the video file
    close(v_out);
