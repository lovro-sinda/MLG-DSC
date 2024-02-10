
% Define a matrix A.
% Define a matrix A.
A = [883 4 2; 1 5 6; 8 7 9; 40 2 1];

% Call the keep_topk function with k = 2.
A(logical(eye(size(A)))) = 0;
A_topk = keep_first_kth_largest(A, 2);

% Display the input matrix A and the output matrix A_topk.
disp('Input matrix:');
disp(A);
disp('Output matrix with top 2 elements:');
disp(A_topk);


