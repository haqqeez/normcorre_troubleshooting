function plot_correlation_matrix(full_matrix, title_str)
    % Create a new figure
    figure;

    % Plot the full matrix
    imagesc(full_matrix);

    % Set the colorbar range
%     caxis([2 3.5]);

    % Add a colorbar
    colorbar;

    % Set the title
    title(title_str);

    % Adjust the aspect ratio to make the figure square
    axis square;
end
