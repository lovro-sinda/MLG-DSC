%% MLGDSC_GMC_LRSSC_ORL (Multilayer graph deep subspace clustering)
% I. Kopriva 2023-07

%%
clear
close all

% Set path to all subfolders
addpath(genpath('.'));

robust_flag = 0; % 0: all modes are equally important; 
                 % 1: weights based on norm of SE error
                 % 2: weights based on F1 score
                 
C3_flag = 2;     % 1: C=DSC_NET; 2: C learned by GMC LRSSC 
                  
delta = 1.2;       % post-processing of angular data adjacency matrix ( delta > 1: 4 or 8)      

dimSubspace = 11; % subspace dimension for ORL dataset

% ranges for grid-search based crossvalidation of the hyperparameters for
% GMC LRSSC algortithm
lambda = 3;
alpha = 10;
gamma_gmc = 1;

% number of performance evaluations 
numit=1; 

%%
% data information
i1 = 48; i2 = 42; N=64*38; nc = 38; % 400 face images of 40 persons (each image 32x32 pixels)

% input data
data = load('YaleBCrop025.mat');
load('YaleBCrop025.mat', 'I');
% Reshape the data
[n1, n2, n3, n4] = size(I);
X = reshape(I, [n1*n2, n3*n4])'; % Here, ' denotes transpose
X0 = double(X);
[n1, n2, n3, n4] = size(I);

[n1, n2, n3, n4] = size(I);
Y_labels = repmat(1:n4, [n3, 1]); % Replicate subject numbers 64 times
Y_labels = Y_labels(:); % Convert it to a column vector

labels=Y_labels;
%X0 = mapstd(X0', 0, 1);
X0 = X0'      
% layer 1 output
X1=csvread('yale1.csv');
%X1 = mapstd(X1', 0, 1);
X1 = X1'
% layer 2 output
X2=csvread('yale2.csv');
%X2 = mapstd(X2', 0, 1);
X2 = X2'

% layer 3 output
X3=csvread('yale3.csv');
%X3 = mapstd(X3', 0, 1);
X3 = X3'

%% Cross-validate (tune) GMC_LRSSC algorithm for each "view" independently
wbh_0 = waitbar(0,'Crossvalidation for NUMIT IN-SAMPLE selections. Please wait ...');
for it=1:numit
    waitbar(it/numit,wbh_0);
    
    % random sampling without replacement
    rng('shuffle');
    labels_in = [];
    X0_in = []; X1_in = []; X2_in = []; X3_in = [];
    for l=1:nc
        ind = randperm(64);
        ind_in = (l-1)*64 + ind(1:50);
        labels_in = [labels_in, ones(1,50)*l];
        X0_in = [X0_in, X0(:,ind_in)];
        X1_in = [X1_in, X1(:,ind_in)];
        X2_in = [X2_in, X2(:,ind_in)];
        X3_in = [X3_in, X3(:,ind_in)];
    end
    
    wbh = waitbar(0,'Crossvalidation for lambda, alpha and gamma. Please wait ...');
    for i=1:length(alpha)
        waitbar(i/length(alpha),wbh);
        for j=1:length(lambda)
            for k=1:length(gamma_gmc)
                options = struct('lambda',lambda(j),'alpha',alpha(i),'rank_est',0.6,'gamma',gamma_gmc(k),...
                    'err_thr',1e-4,'iter_max',100, 'affine',false,...
                    'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
                
                alpha = 10; mu2 = 3; gamma = 1;
                options = struct('gamma',gamma);
                [C, error] = GMC_LRSSC(normc(X0_in), alpha, mu2, options);
                A = adjacency_matrix_angular_domain(C, delta);
                A = real(A)
                labels_est = SpectralClusteringL(A,nc);
                CE0(it,i,j,k)  = computeCE(labels_est,labels_in);
                
                alpha = 10; mu2 = 3; gamma = 1;
                options = struct('gamma',gamma);
                [C, error] = GMC_LRSSC(normc(X1_in), alpha, mu2, options);
                A = adjacency_matrix_angular_domain(C, delta);
                A = real(A)
                labels_est = SpectralClusteringL(A,nc);
                CE1(it,i,j,k)  = computeCE(labels_est,labels_in);
                
                alpha = 10; mu2 = 3; gamma = 1;
                options = struct('gamma',gamma);
                [C, error] = GMC_LRSSC(normc(X2_in), alpha, mu2, options);
                A = adjacency_matrix_angular_domain(C, delta);
                A = real(A)
                labels_est = SpectralClusteringL(A,nc);
                CE2(it,i,j,k)  = computeCE(labels_est,labels_in);
                
                if C3_flag == 2
                    alpha = 10; mu2 = 3; gamma = 1;
                    options = struct('gamma',gamma);
                    [C, error] = GMC_LRSSC(normc(X3_in), alpha, mu2, options);
                    A = adjacency_matrix_angular_domain(C, delta);
                    A = real(A)
                    labels_est = SpectralClusteringL(A,nc);
                    CE3(it,i,j,k)  = computeCE(labels_est,labels_in);
                end
            end
        end
    end
    close(wbh)
end
close(wbh_0)

% select optimal hyperparameters for each layer
for i=1:length(alpha)
    for j=1:length(lambda)
        for k=1:length(gamma_gmc)
            CE0_mx(i,j,k) = mean(CE0(:,i,j,k));
            CE1_mx(i,j,k) = mean(CE1(:,i,j,k));
            CE2_mx(i,j,k) = mean(CE2(:,i,j,k));
            CE3_mx(i,j,k) = mean(CE3(:,i,j,k));
        end
    end
end

cemin_0=1; cemin_1=1; cemin_2=1; cemin_3=1;
for i=1:length(alpha)
    for j=1:length(lambda)
        for k=1:length(gamma_gmc)
            if cemin_0 > CE0_mx(i,j,k)
                cemin_0 = CE0_mx(i,j,k);
                alpha_0=alpha(i);
                lambda_0=lambda(j);
                gamma_0 = gamma_gmc(k); 
            end
            
            if cemin_1 > CE1_mx(i,j,k)
                cemin_1 = CE1_mx(i,j,k);
                alpha_1=alpha(i);
                lambda_1=lambda(j);
                gamma_1 = gamma_gmc(k);
            end
            
            if cemin_2 > CE2_mx(i,j,k)
                cemin_2 = CE2_mx(i,j,k);
                alpha_2=alpha(i);
                lambda_2=lambda(j);
                gamma_2 = gamma_gmc(k);
            end
            
            if C3_flag == 2
                if cemin_3 > CE3_mx(i,j,k)
                    cemin_3 = CE3_mx(i,j,k);
                    alpha_3=alpha(i);
                    lambda_3=lambda(j);
                    gamma_3=gamma_gmc(k);
                end
            end
        end
    end
end
disp("Tu sam")
%% crossvalidate gamma
gamma=0.1;
numit=1
wbh = waitbar(0,'Crossvalidation for gamma. Please wait ...');
for it=1:numit
    waitbar(it/numit);
    % random sampling without replacement
    rng('shuffle');
    labels_in = [];
    X0_in = []; X1_in = []; X2_in = []; X3_in = [];
    for l=1:nc
        ind = randperm(64);
        ind_in = (l-1)*64 + ind(1:50);
        labels_in = [labels_in, ones(1,50)*l];
        X0_in = [X0_in, X0(:,ind_in)];
        X1_in = [X1_in, X1(:,ind_in)];
        X2_in = [X2_in, X2(:,ind_in)];
        X3_in = [X3_in, X3(:,ind_in)];
    end
    
    for i=1:length(gamma)
        % Layer 0: input data
        options = struct('lambda',lambda_0,'alpha',alpha_0,'rank_est',0.6,'gamma',gamma_0,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        alpha = 10; mu2 = 3; gamma = 1;
        options = struct('gamma',gamma);
        [C, error] = GMC_LRSSC(normc(X0_in), alpha, mu2, options);
        A = adjacency_matrix_angular_domain(C, delta);
        A = real(A)
        if robust_flag == 1
            % estimate influence of self-expression error
            a_0 = 1/norm(normc(X0_in) - normc(X0_in)*C,'fro');
            a_0 = a_0 * a_0;
          %              a_0 = a_0 * a_0;
        elseif robust_flag == 2           
            labels_est = SpectralClusteringL(A,nc);
            a_0 = compute_f(labels,labels_est);
        elseif robust_flag == 0
            a_0=1/4;
        end     
        N = size(A,1);
        
        % compute shifted Laplacian for input data (layer 0)
        DN = diag( 1./sqrt(sum(A)+eps) );
        L0_shifted = speye(N) + DN*A*DN;
        [U0_s,S0_s,~] = svd(L0_shifted);
        U0_s = U0_s(:,N-nc+1:N);  % nc largest eigenvectors to span the subspace
                       
        % Layer 1
        options = struct('lambda',lambda_1,'alpha',alpha_1,'rank_est',0.6,'gamma',gamma_1,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        alpha = 10; mu2 = 3; gamma = 1;
        options = struct('gamma',gamma);
        [C, error] = GMC_LRSSC(normc(X1_in), alpha, mu2, options);
        A = real(adjacency_matrix_angular_domain(C, delta));   
        A = real(A)
        if robust_flag == 1
            % estimate influence of self-expression error
            a_1 = 1/norm(normc(X1_in) - normc(X1_in)*C,'fro');
            a_1 = a_1 * a_1;
         %               a_1 = a_1 * a_1;
        elseif robust_flag == 2                  
            labels_est = SpectralClusteringL(A,nc);
            a_1 = compute_f(labels,labels_est);
        elseif robust_flag == 0
            a_1=1/4;
        end
        N = size(A,1);
               
        % compute shifted Laplacian for input data (layer 1)
        DN = diag( 1./sqrt(sum(A)+eps) );
        L1_shifted = speye(N) + DN*A*DN;
        [U1_s,S1_s,~] = svd(L1_shifted);
        U1_s = U1_s(:,N-nc+1:N); 
               
        % Layer 2
        options = struct('lambda',lambda_2,'alpha',alpha_2,'rank_est',0.6,'gamma',gamma_2,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        alpha = 10; mu2 = 3; gamma = 1;
        options = struct('gamma',gamma);
        [C, error] = GMC_LRSSC(normc(X2_in), alpha, mu2, options);
        A = real(adjacency_matrix_angular_domain(C, delta));   
        A = real(A)
        if robust_flag == 1
            % estimate influence of self-expression error
            a_2 = 1/norm(normc(X2_in) - normc(X2_in)*C,'fro');
            a_2 = a_2 * a_2;
            %            a_2 = a_2 * a_2;
        elseif robust_flag == 2
            labels_est = SpectralClusteringL(A,nc);
            a_2 = compute_f(labels,labels_est);
        elseif robust_flag == 0
            a_2=1/4;
        end
        N = size(A,1);
        
        % compute shifted Laplacian for input data (layer 2)
        DN = diag( 1./sqrt(sum(A)+eps) );
        L2_shifted = speye(N) + DN*A*DN;
        [U2_s,S2_s,~] = svd(L2_shifted);
        U2_s = U2_s(:,N-nc+1:N);
        
        % Layer 3
        if C3_flag == 1
            C = csvread('yaleb-c.csv');
        elseif C3_flag == 2
            options = struct('lambda',lambda_3,'alpha',alpha_3,'rank_est',0.6,'gamma',gamma_3,...
                'err_thr',1e-4,'iter_max',100, 'affine',false,...
                'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        alpha = 10; mu2 = 3; gamma = 1;
        options = struct('gamma',gamma);
        [C, error] = GMC_LRSSC(normc(X3_in), alpha, mu2, options);
        end
        A = adjacency_matrix_angular_domain(C, delta);
        A = real(A)
        if robust_flag == 1
            % estimate influence of self-expression error
            a_3 = 1/norm(normc(X3_in) - normc(X3_in)*C,'fro');
            a_3 = a_3 * a_3;
            %               a_3 = a_3 * a_3;
        elseif robust_flag == 2
            labels_est = SpectralClusteringL(A,nc);
            a_3 = compute_f(labels,labels_est);
        elseif robust_flag == 0
            a_3=1/4;
        end
        N = size(A,1);
        
        % compute shifted Laplacian for input data (layer 3)
        DN = diag( 1./sqrt(sum(A)+eps) );
        L3_shifted = speye(N) + DN*A*DN;
        [U3_s,S3_s,~] = svd(L3_shifted);
        U3_s = U3_s(:,N-nc+1:N);
        
        % robust multimodal Laplacian
        sum_a = a_0 + a_1 + a_2 + a_3;
        an_0 = a_0/sum_a;
        an_1 = a_1/sum_a;
        an_2 = a_2/sum_a;
        an_3 = a_3/sum_a;
        
        LL_s = an_0*L0_shifted + an_1*L1_shifted + an_2*L2_shifted + ...
            an_3*L3_shifted;
        UU_s = an_0*U0_s*U0_s' + an_1*U1_s*U1_s' + an_2*U2_s*U2_s' + ...
            an_3*U3_s*U3_s';
        
        Lmod_r = LL_s - gamma(i)*UU_s;
        [Umod,Smod,Vmod]=svd(Lmod_r);
        kerN = Vmod(:,1:nc); % nc largest eigenvectors        
        % nornamlize to unit row norm
        for ii = 1:N
            kerNS(ii,:) = kerN(ii,:) ./ norm(kerN(ii,:)+eps);
        end
        %% 
        
        MAXiter = 1000; % Maximum number of iterations for KMeans
        REPlic = 20; % Number of replications for KMeans
        warning off
        labels_est = kmeans(real(kerNS),nc,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
        CE_gamma(it,i)=computeCE(labels_est,labels)        
    end    
end
close(wbh)

% select optimal gamma
CE_mgamma = mean(CE_gamma,1);
cemin_gamma=1;
for i=1:length(gamma)
    if cemin_gamma > CE_mgamma(i)
        cemin_gamma = CE_mgamma(i);
        gamma_opt=gamma(i);
    end
end

%% repeat numit times to estimate clustering quality metrics
numit=10;

wbh = waitbar(0,'Clustering performance evaluation. Please wait ...');
for iter=1:numit
    waitbar(iter/numit);
    % random sampling without replacement
    rng('shuffle');
    labels_in = [];
    X0_in = []; X1_in = []; X2_in = []; X3_in = [];
    X0_out = []; X1_out = []; X2_out = []; X3_out = [];
    labels_out = [];
    for l=1:nc
        ind = randperm(64);
        
        % in-sample data
        ind_in = (l-1)*64 + ind(1:50);
        labels_in = [labels_in, ones(1,50)*l];        
        X0_in = [X0_in, X0(:,ind_in)];        
        X1_in = [X1_in, X1(:,ind_in)];
        X2_in = [X2_in, X2(:,ind_in)];    
        X3_in = [X3_in, X3(:,ind_in)];
        
        % out-of sample data
        ind_out = (l-1)*64 + ind(51:64);
        labels_out = [labels_out, ones(1,64-51+1)*l];
        X0_out = [X0_out, X0(:,ind_out)];
        X1_out = [X1_out, X1(:,ind_out)];
        X2_out = [X2_out, X2(:,ind_out)];
        X3_out = [X3_out, X3(:,ind_out)];       
    end
    
    % Layer 0: input data
    options = struct('lambda',lambda_0,'alpha',alpha_0,'rank_est',0.6,'gamma',gamma_0,...
        'err_thr',1e-4,'iter_max',100, 'affine',false,...
        'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
    alpha = 10; mu2 = 3; gamma = 1;
    options = struct('gamma',gamma);
    [C, error] = GMC_LRSSC(normc(X0_in), alpha, mu2, options);
    A = adjacency_matrix_angular_domain(C, delta);
    
    if robust_flag == 1
        % estimate influence of self-expression error
        a_0 = 1/norm(normc(X0_in) - normc(X0_in)*C,'fro');
        a_0 = a_0 * a_0;
    elseif robust_flag == 2
        labels_est = SpectralClusteringL(A,nc);
        a_0 = compute_f(labels,labels_est);
    elseif robust_flag == 0
        a_0=1/4;
    end
    N = size(A,1);
      
    % compute shifted Laplacian for input data (layer 0)
    DN = diag( 1./sqrt(sum(A)+eps) );
    L0_shifted = speye(N) + DN * A * DN;
    [U0_s,S0_s,~] = svd(L0_shifted);
    
    % Layer 1
     options = struct('lambda',lambda_1,'alpha',alpha_1,'rank_est',0.6,'gamma',gamma_1,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
    alpha = 100; mu2 = 3; gamma = 1;
    options = struct('gamma',gamma);
    [C, error] = GMC_LRSSC(normc(X1_in), alpha, mu2, options);    
    A = adjacency_matrix_angular_domain(C, delta);
    
    if robust_flag == 1
        % estimate influence of self-expression error
        a_1 = 1/norm(normc(X1_in) - normc(X1_in)*C,'fro');
        a_1 = a_1 * a_1;
    elseif robust_flag == 2
        labels_est = SpectralClusteringL(A,nc);
        a_1 = compute_f(labels,labels_est);
    elseif robust_flag == 0
        a_1=1/4;
    end
    N = size(A,1);
      
    % compute shifted Laplacian for input data (layer 1)
    DN = diag( 1./sqrt(sum(A)+eps) );
    L1_shifted = speye(N) + DN*A*DN;
    [U1_s,S1_s,~] = svd(L1_shifted);
    
    % Layer 2
     options = struct('lambda',lambda_2,'alpha',alpha_2,'rank_est',0.6,'gamma',gamma_2,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
    alpha = 10; mu2 = 3; gamma = 1;
    options = struct('gamma',gamma);
    [C, error] = GMC_LRSSC(normc(X2_in), alpha, mu2, options); 
    A = adjacency_matrix_angular_domain(C, delta);
    
    if robust_flag == 1
        % estimate influence of self-expression error
        a_2 = 1/norm(normc(X2) - normc(X2)*C,'fro');
        a_2 = a_2 * a_2;
    elseif robust_flag == 2
        labels_est = SpectralClusteringL(A,nc);
        a_2 = compute_f(labels,labels_est);
    elseif robust_flag == 0
        a_2=1/4;
    end
    N = size(A,1);
      
    % compute shifted Laplacian for input data (layer 2)
    DN = diag( 1./sqrt(sum(A)+eps) );
    L2_shifted = speye(N) + DN*A*DN;
    [U2_s,S2_s,~] = svd(L2_shifted);
    
    % Layer 3
    if C3_flag == 1
        C = csvread('yaleb-c.csv');
    elseif C3_flag == 2
        options = struct('lambda',lambda_3,'alpha',alpha_3,'rank_est',0.6,'gamma',gamma_3,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        alpha = 10; mu2 = 3; gamma = 1;
    options = struct('gamma',gamma);
    [C, error] = GMC_LRSSC(normc(X3_in), alpha, mu2, options);
    end
    A = adjacency_matrix_angular_domain(C, delta);
    
    if robust_flag == 1
        % estimate influence of self-expression error
        a_3 = 1/norm(normc(X3_in) - normc(X3_in)*C,'fro');
        a_3 = a_3 * a_3;
    elseif robust_flag == 2
        labels_est = SpectralClusteringL(A,nc);
        a_3 = compute_f(labels,labels_est);
    elseif robust_flag == 0
        a_3=1/4;
    end
    N = size(A,1);
      
    % compute shifted Laplacian for input data (layer 3)
    DN = diag( 1./sqrt(sum(A)+eps) );
    L3_shifted = speye(N) + DN*A*DN;
    [U3_s,S3_s,~] = svd(L3_shifted);
    %% 
    disp("Počinje")
    % robust multimodal Laplacian
    sum_a = a_0 + a_1 + a_2 + a_3;
    an_0 = a_0/sum_a;
    an_1 = a_1/sum_a;
    an_2 = a_2/sum_a; 
    an_3 = a_3/sum_a;
    
    LL_s = an_0*L0_shifted + an_1*L1_shifted + an_2*L2_shifted + ...
        an_3*L3_shifted;
    UU_s = an_0*U0_s*U0_s' + an_1*U1_s*U1_s' + an_2*U2_s*U2_s' + ...
        an_3*U3_s*U3_s';
    
    Lmod_r = LL_s - gamma_opt*UU_s;
    
    [Umod,Smod,Vmod]=svd(Lmod_r);
    kerN = Vmod(:,1:nc); % nc largest eigenvectors
    % nornamlize to unit row norm
    for i = 1:N
        kerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
    end
    
    MAXiter = 1000; % Maximum number of iterations for KMeans
    REPlic = 20; % Number of replications for KMeans
    warning off
    labels_est = kmeans(real(kerNS),nc,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
        
    % Performance measures for in-sample data 
    ACC_in(iter)  = 1 - computeCE(labels_est,labels_in)
    NMI_in(iter) = compute_nmi(labels_in',labels_est)
    Fscore_in(iter) = compute_f(labels_in',labels_est)
    
    % estimate bases in embeded space using labels from equivalent multilayer space
    [B_x, begB_x, endB_x, mu_X]  = bases_estimation(X3_in, labels_est, dimSubspace);
    A0=labels_out;
    N_out = size(X3_out,2);
    X3_out = normc(X3_out);
    
    for l=1:nc
        X3_outm = X3_out - mu_X(:,l);    % make data zero mean for distance calculation
        BB=B_x(:,begB_x(l):endB_x(l));
        Xproj = (BB*BB')*X3_outm;
        Dproj = X3_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x] = min(D);
    clear D
    
    % Performance measures on out-of-sample data   
    ACC_out(iter)  = 1 - computeCE(A_x,A0)
    NMI_out(iter) = compute_nmi(A0,A_x)
    Fscore_out(iter) = compute_f(A0,A_x)
    
    % compute clustering performance on encoder output (layer 3)
    [U3,S3,V3]=svd(L3_shifted);
    kerN = V3(:,1:nc); % nc largest eigenvectors
    % nornamlize to unit row norm
    for i = 1:N
        kerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
    end
    
    MAXiter = 1000; % Maximum number of iterations for KMeans
    REPlic = 38; % Number of replications for KMeans
    warning off
    labels_est = kmeans(real(kerNS),nc,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');   
    
    % Performance measures on in-sample data
    ACC_3_in(iter)  = 1 - computeCE(labels_est,labels_in)
    NMI_3_in(iter) = compute_nmi(labels_in',labels_est)
    Fscore_3_in(iter) = compute_f(labels_in',labels_est)
    
     % estimate bases in embeded space using labels from equivalent multilayer space
    [B_x, begB_x, endB_x, mu_X]  = bases_estimation(X3_in, labels_est, dimSubspace); 
    A0=labels_out;
    N_out = size(X3_out,2);
    X3_out = normc(X3_out);
    
    for l=1:nc
        X3_outm = X3_out - mu_X(:,l);    % make data zero mean for distance calculation
        BB=B_x(:,begB_x(l):endB_x(l));
        Xproj = (BB*BB')*X3_outm;
        Dproj = X3_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x] = min(D);
    clear D
    
    % Performance measures on out-of-sample data   
    ACC_3_out(iter)  = 1 - computeCE(A_x,A0)
    NMI_3_out(iter) = compute_nmi(A0,A_x)
    Fscore_3_out(iter) = compute_f(A0,A_x)
end

display('IN-SAMPLE DATA !!!!!!!!!!')

display('Multilayer graph')

display('Mean ACC:')
mean(ACC_in)
display('Std ACC:')
std(ACC_in)

display('Mean NMI:')
mean(NMI_in)
display('Std NMI:')
std(NMI_in)

display('Mean Fscore:')
mean(Fscore_in)
display('Std Fscore:')
std(Fscore_in)

display('Encoder output (layer 3):')
display('Mean ACC:')
mean(ACC_3_in)
display('Std ACC:')
std(ACC_3_in)

display('Mean NMI:')
mean(NMI_3_in)
display('Std NMI:')
std(NMI_3_in)

display('Mean Fscore:')
mean(Fscore_3_in)
display('Std Fscore:')
std(Fscore_3_in)

% ranksum two sided Wilcox test of statistical significance
p_acc_in=ranksum(ACC_3_in,ACC_in)
p_nmi_in=ranksum(NMI_3_in,NMI_in)
p_fscore_in=ranksum(Fscore_3_in,Fscore_in)

display('OUT-OF-SAMPLE DATA !!!!!!!!!!')

display('Multilayer graph')

display('Mean ACC:')
mean(ACC_out)
display('Std ACC:')
std(ACC_out)

display('Mean NMI:')
mean(NMI_out)
display('Std NMI:')
std(NMI_out)

display('Mean Fscore:')
mean(Fscore_out)
display('Std Fscore:')
std(Fscore_out)

display('Encoder output (layer 3):')
display('Mean ACC:')
mean(ACC_3_out)
display('Std ACC:')
std(ACC_3_out)

display('Mean NMI:')
mean(NMI_3_out)
display('Std NMI:')
std(NMI_3_out)

display('Mean Fscore:')
mean(Fscore_3_out)
display('Std Fscore:')
std(Fscore_3_out)

% ranksum two sided Wilcox test of statistical significance
p_acc_out=ranksum(ACC_3_out,ACC_out)
p_nmi_out=ranksum(NMI_3_out,NMI_out)
p_fscore_out=ranksum(Fscore_3_out,Fscore_out)


