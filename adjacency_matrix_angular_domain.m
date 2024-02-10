function [A] = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace)
% Compute adjacency matrix in angular domain
% Inputs: C: NxN data representation matrix
%         delta >=1 discrimination parametere
% Output: A: NxN data adjacency matrix

m=size(C,1);

C = (abs(C) + abs(C'))/2;

if remove_flag
    CT = keep_first_kth_largest(transpose(C),dimSubspace);   
    %transpose is to compensate row-wise operation
    C = transpose(CT);
end
C = (abs(C) + abs(C'))/2;
[Um,Sm,Vm]=svds(C,m);
Z=Um*sqrt(Sm);
zz=diag(Z*Z') + eps;
zz=1./sqrt(zz);
Z=diag(zz)*Z;  % normalize to unit row norm
A = Z*Z';
A = power(A,delta);
end

