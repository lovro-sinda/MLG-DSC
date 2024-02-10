function new_matrix = keep_first_kth_largest(matrix, k)
% This function takes a MATLAB matrix and an integer k as input.
% It returns a new matrix where each row has only the k largest numbers by 
% absolute value, and all other numbers in the row are zeroed out.

% Get the number of rows and columns in the matrix.
[num_rows, num_cols] = size(matrix);

% Check for a valid value of k
if k > num_cols || k <= 0
    error('Invalid value of k. It should be between 1 and the number of columns in the matrix.');
end

% Sort each row of the matrix by absolute value in descending order.
[sorted_rows, sorted_indices] = sort(abs(matrix), 2, 'descend');

% Determine the kth largest absolute value for each row
kth_largest_values = sorted_rows(:, k);

% Create a new matrix initialized with zeros.
new_matrix = zeros(num_rows, num_cols);

% Retain only the k largest absolute values in each row.
for i = 1:num_rows
    indices_to_keep = sorted_indices(i, 1:k);
    new_matrix(i, indices_to_keep) = matrix(i, indices_to_keep);
end

end
