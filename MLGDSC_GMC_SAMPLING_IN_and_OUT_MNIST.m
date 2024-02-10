%% MLGDSC_GMC_SAMPLING_IN_and_OUT_MNIST (Multilayer graph deep subspace clustering)
% I. Kopriva 2023-07/12

%%
clear
close all

% Set path to all subfolders
addpath(genpath('.'));

%% parameters             
delta = 2;       % post-processing of angular data adjacency matrix ( delta > 1: 4 or 8)   

remove_flag = true; % True - keep only dimSubspace leading coefficients per column of selfreprsentation matrix
dimSubspace = 12;    % a priori known (presumed) dimension of the subspace

% ranges for grid-search based crossvalidation of the hyperparameters for GMC LRSSC algortithm
lambda = 0.5:0.1:2.5;
alpha = 0.2:0.2:2;
gamma_gmc = 0:0.1:1;

%% data information
i1 = 28; i2 = 28;  N=10000; nc = 10; % 10000 images of ten digits (each image 28x28 pixels)

% OVAJ DIO ZAVISI OD TOGA KAKO SU FORMATIRANI PODACI
% ULAZNA MATRICA X0 mora biti 784x10000, A MATRICE X1, X2 i X3
% dimenzijax10000

images = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

[labelssorted,IX] = sort(labels);
X0 = images(:,IX);   %slike sort i tako su rađeni i ostali
labels = kron(0:9, ones(1, 1000));
% X0 = ????
% input data (layer 0)
%

% layer 1 output
X1=csvread('mnist1.csv');



% layer 2 output
X2=csvread('mnist2.csv');
%X2=transpose(X2);
%

% layer 3 output
X3=csvread('mnist3.csv');
%X3=transpose(X3);
%%
% self-representation matrix learned by DSC-L2 net algorithm
C_DSC_L2_net = csvread('mnist.matc.csv'); 
C_DSC_L2_net = C_DSC_L2_net(2:end, :);
%% 

% Sampling in-sample and out-of-sample data nruns times to select hyperparameters of the GMC LRSSC algorithm
nruns = 10;

wbh_0 = waitbar(0,'Number of runs. Please wait ...');
for run = 1:nruns
    waitbar(run/nruns,wbh_0);
    % generate a problem instance
    rng('shuffle');
    labels_in = [];
    X0_in = [];  X1_in = []; X2_in = []; X3_in = []; 
    index_in = [];  
    for l=1:nc
        ind = randperm(1000);
        ind_in = (l-1)*1000 + ind(1:50);   % 50 images per digit for in-sample data
        labels_in = [labels_in, ones(1,50)*l];
        X0_in = [X0_in, X0(:,ind_in)]; X1_in = [X1_in, X1(:,ind_in)];
        X2_in = [X2_in, X2(:,ind_in)]; X3_in = [X3_in, X3(:,ind_in)];
    end
    
    % Crossvalidation of the three hyperparameters of GMC LRSSC algorithm
    wbh_1 = waitbar(0,'Crossvalidation for the whole algorithm. Please wait ...');
    for i=1:length(alpha)
        waitbar(i/length(alpha),wbh_1);
        for j=1:length(lambda)
            for k=1:length(gamma_gmc)
                options = struct('lambda',lambda(j),'alpha',alpha(i),'rank_est',0.6,'gamma',gamma_gmc(k),...
                    'err_thr',1e-4,'iter_max',100, 'affine',false,...
                    'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
                
                [C, error] = ADMM_LRSSC(normc(X0_in),options);
                A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
                labels_est = SpectralClustering_shifted_Laplacian(A,nc);
                CE0(run,i,j,k)  = computeCE(labels_est,labels_in);
                
                [C, error] = ADMM_LRSSC(normc(X1_in),options);
                A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
                labels_est = SpectralClustering_shifted_Laplacian(A,nc);
                CE1(run,i,j,k)  = computeCE(labels_est,labels_in);
                
                [C, error] = ADMM_LRSSC(normc(X2_in),options);
                A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
                labels_est = SpectralClustering_shifted_Laplacian(A,nc);
                CE2(run,i,j,k)  = computeCE(labels_est,labels_in);
                
                [C, error] = ADMM_LRSSC(normc(X3_in),options);
                A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
                labels_est = SpectralClustering_shifted_Laplacian(A,nc);
                CE3(run,i,j,k)  = computeCE(labels_est,labels_in);
            end
        end
    end
    close(wbh_1)
end
close(wbh_0)

% Average performance over multiple runs
for i=1:length(alpha)
    for j=1:length(lambda)
        for k=1:length(gamma_gmc)
            CE0_m(i,j,k)=mean(CE0(:,i,j,k));
            CE1_m(i,j,k)=mean(CE1(:,i,j,k));
            CE2_m(i,j,k)=mean(CE2(:,i,j,k)); 
            CE3_m(i,j,k)=mean(CE3(:,i,j,k));           
        end
    end
end

cemin_0=1; cemin_1=1; cemin_2=1; cemin_3=1;

% Select "optimal" values of hyperparameters
for i=1:length(alpha)
    for j=1:length(lambda)
        for k=1:length(gamma_gmc)
            if cemin_0 > CE0_m(i,j,k)
                cemin_0 = CE0_m(i,j,k);
                alpha_0=alpha(i);
                lambda_0=lambda(j);
                gamma_0=gamma_gmc(k);
            end
            
            if cemin_1 > CE1_m(i,j,k)
                cemin_1 = CE1_m(i,j,k);
                alpha_1=alpha(i);
                lambda_1=lambda(j);
                gamma_1=gamma_gmc(k);
            end
            
            if cemin_2 > CE2_m(i,j,k)
                cemin_2 = CE2_m(i,j,k);
                alpha_2=alpha(i);
                lambda_2=lambda(j);
                gamma_2=gamma_gmc(k);
            end
            
            if cemin_3 > CE3_m(i,j,k)
                cemin_3 = CE3_m(i,j,k);
                alpha_3=alpha(i);
                lambda_3=lambda(j);
                gamma_3=gamma_gmc(k);
            end
        end
    end
end
clear CE0 CE0_m CE1 CE1_m CE2 CE2_m CE3 CE3_m

%% crossvalidate gamma in multimodal Laplacian
gamma=0:0.1:1;

nruns = 10;
wbh_0 = waitbar(0,'Number of runs. Please wait ...');
for run = 1:nruns
    waitbar(run/nruns,wbh_0);
    % generate a problem instance
    rng('shuffle');
    labels_in = [];
    X0_in = [];  X1_in = []; X2_in = []; X3_in = [];
    index_in = [];
    for l=1:nc
        ind = randperm(1000);
        ind_in = (l-1)*1000 + ind(1:50);
        labels_in = [labels_in, ones(1,50)*l];
        X0_in = [X0_in, X0(:,ind_in)]; X1_in = [X1_in, X1(:,ind_in)];
        X2_in = [X2_in, X2(:,ind_in)]; X3_in = [X3_in, X3(:,ind_in)];
    end
    
    wbh = waitbar(0,'Crossvalidation for gamma in multimoda Laplacian. Please wait ...');
    for i=1:length(gamma)
        waitbar(i/length(gamma));
        
        % Layer 0: input data
        options = struct('lambda',lambda_0,'alpha',alpha_0,'rank_est',0.6,'gamma',gamma_0,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        [C, error] = ADMM_LRSSC(normc(X0_in),options);
        A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
        N_in = size(A,1);
        
        % compute shifted Laplacian for input data (layer 0)
        DN = diag( 1./sqrt(sum(A)+eps) );
        L0_shifted = speye(N_in) + DN*A*DN;
        [U0_s,S0_s,~] = svd(L0_shifted);
        U0_s = U0_s(:,N_in-nc+1:N_in);  % nc smallest vectors (largest in normalized Laplacian) eigenvectors to span the subspace
        
        % Layer 1
        options = struct('lambda',lambda_1,'alpha',alpha_1,'rank_est',0.6,'gamma',gamma_1,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        [C, error] = ADMM_LRSSC(normc(X1_in),options);
        A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
        
        % compute shifted Laplacian for input data (layer 1)
        DN = diag( 1./sqrt(sum(A)+eps) );
        L1_shifted = speye(N_in) + DN*A*DN;
        [U1_s,S1_s,~] = svd(L1_shifted);
        U1_s = U1_s(:,N_in-nc+1:N_in); % nc smallest vectors (largest in normalized Laplacian) eigenvectors to span the subspace
        
        % Layer 2
        options = struct('lambda',lambda_2,'alpha',alpha_2,'rank_est',0.6,'gamma',gamma_2,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        [C, error] = ADMM_LRSSC(normc(X2_in),options);
        A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
        
        % compute shifted Laplacian for input data (layer 2)
        DN = diag( 1./sqrt(sum(A)+eps) );
        L2_shifted = speye(N_in) + DN*A*DN;
        [U2_s,S2_s,~] = svd(L2_shifted);
        U2_s = U2_s(:,N_in-nc+1:N_in); % nc smallest vectors (largest in normalized Laplacian) eigenvectors to span the subspace
        
        % Layer 3
        options = struct('lambda',lambda_3,'alpha',alpha_3,'rank_est',0.6,'gamma',gamma_3,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        [C, error] = ADMM_LRSSC(normc(X3_in),options);
        A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
        
        % compute shifted Laplacian for input data (layer 3)
        DN = diag( 1./sqrt(sum(A)+eps) );
        L3_shifted = speye(N_in) + DN*A*DN;
        [U3_s,S3_s,~] = svd(L3_shifted);
        U3_s = U3_s(:,N_in-nc+1:N_in); % nc smallest vectors (largest in normalized Laplacian) eigenvectors to span the subspace
        
        % Multimodal Laplacian       
        LL_s = L0_shifted + L1_shifted + L2_shifted + L3_shifted;
        UU_s = U0_s*U0_s' + U1_s*U1_s' + U2_s*U2_s' + U3_s*U3_s';
        
        Lmod_r = LL_s - gamma(i)*UU_s;
        [Umod,Smod,Vmod]=svd(Lmod_r);
        kerN = Vmod(:,1:nc); % nc largest eigenvectors
        % nornamlize to unit row norm
        for ii = 1:N_in
            kerNS(ii,:) = kerN(ii,:) ./ norm(kerN(ii,:)+eps);
        end
        
        MAXiter = 1000; % Maximum number of iterations for KMeans
        REPlic = 20; % Number of replications for KMeans
        warning off
        labels_est = kmeans(kerNS,nc,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
        CE_gamma(run,i)=computeCE(labels_est,labels_in)
    end
    close(wbh)
end

% select optimal gamma
CE_mgamma = mean(CE_gamma,1);
cemin_gamma=1;
for i=1:length(gamma)
    if cemin_gamma > CE_mgamma(i)
        cemin_gamma = CE_mgamma(i);
        gamma_opt=gamma(i);
    end
end
clear CE_gamma CE_mgamma
    
save MLG_GMC_MNIST_hyperparameters lambda_0 alpha_0 gamma_0 lambda_1 alpha_1 gamma_1...
    lambda_2 alpha_2 gamma_2 lambda_3 alpha_3 gamma_3 gamma_opt

%% Estimate clustering quality metrics
numit = 100;
% Split into in-sample and out-of-sample partitions
for it=1:numit
    fprintf('Iter %d\n',it);
    % generate a problem instance
    rng('shuffle');
    labels_in = [];
    labels_out = [];
    X0_in = []; X0_out = []; X1_in = []; X1_out = [];
    X2_in = []; X2_out = []; X3_in = []; X3_out = [];    
    index_in = []; index_out = [];
    for l=1:nc
        ind = randperm(1000);
        ind_in = (l-1)*1000 + ind(1:50);  % 50 images per digit for in-sample
        index_in = [index_in, ind_in];
        ind_out = (l-1)*1000 + ind(51:100); % 50 images per digiat for out-of-sample
        index_out = [index_out, ind_out];
        labels_in = [labels_in, ones(1,50)*l];
        labels_out = [labels_out, ones(1,50)*l];
        X0_in = [X0_in, X0(:,ind_in)]; X0_out = [X0_out, X0(:,ind_out)];
        X1_in = [X1_in, X1(:,ind_in)]; X1_out = [X1_out, X1(:,ind_out)];  
        X2_in = [X2_in, X2(:,ind_in)]; X2_out = [X2_out, X2(:,ind_out)];        
        X3_in = [X3_in, X3(:,ind_in)]; X3_out = [X3_out, X3(:,ind_out)];
    end

    A0=labels_out;
    N_out = size(X3_out,2);
    X0_out = normc(X0_out); X1_out = normc(X1_out);
    X2_out = normc(X2_out); X3_out = normc(X3_out);
    
    % Layer 0: input data
    options = struct('lambda',lambda_0,'alpha',alpha_0,'rank_est',0.6,'gamma',gamma_0,...
        'err_thr',1e-4,'iter_max',100, 'affine',false,...
        'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
    [C, error] = ADMM_LRSSC(normc(X0_in),options);
    A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
    N_in = size(A,1);
      
    % compute shifted Laplacian for input data (layer 0)
    DN = diag( 1./sqrt(sum(A)+eps) );
    L0_shifted = speye(N_in) + DN*A*DN;
    [U0_s,S0_s,~] = svd(L0_shifted);
    U0_s = U0_s(:,N_in-nc+1:N_in);  % nc smallest vectors (largest in normalized Laplacian) eigenvectors to span the subspace
    
    % Layer 1 data
    options = struct('lambda',lambda_1,'alpha',alpha_1,'rank_est',0.6,'gamma',gamma_1,...
        'err_thr',1e-4,'iter_max',100, 'affine',false,...
        'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
    [C, error] = ADMM_LRSSC(normc(X1_in),options);
    A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
    N_in = size(A,1);
    
    % compute shifted Laplacian for layer 1 output
    DN = diag( 1./sqrt(sum(A)+eps) );
    L1_shifted = speye(N_in) + DN*A*DN;
    [U1_s,S1_s,~] = svd(L1_shifted);
    U1_s = U1_s(:,N_in-nc+1:N_in);  % nc smallest vectors (largest in normalized Laplacian) eigenvectors to span the subspace
        
    % Layer 2 data
    options = struct('lambda',lambda_2,'alpha',alpha_2,'rank_est',0.6,'gamma',gamma_2,...
        'err_thr',1e-4,'iter_max',100, 'affine',false,...
        'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
    [C, error] = ADMM_LRSSC(normc(X2_in),options);
    A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
    N_in = size(A,1);
    
    % compute shifted Laplacian for layer 2 output
    DN = diag( 1./sqrt(sum(A)+eps) );
    L2_shifted = speye(N_in) + DN*A*DN;
    [U2_s,S2_s,~] = svd(L2_shifted);
    U2_s = U2_s(:,N_in-nc+1:N_in);  % nc smallest vectors (largest in normalized Laplacian) eigenvectors to span the subspace
   
    % Layer 3 data
    options = struct('lambda',lambda_3,'alpha',alpha_3,'rank_est',0.6,'gamma',gamma_3,...
        'err_thr',1e-4,'iter_max',100, 'affine',false,...
        'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
    [C, error] = ADMM_LRSSC(normc(X3_in),options);
    A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
    N_in = size(A,1);
       
    % compute shifted Laplacian for layer 3 output
    DN = diag( 1./sqrt(sum(A)+eps) );
    L3_shifted = speye(N_in) + DN*A*DN;
    [U3_s,S3_s,~] = svd(L3_shifted);
    U3_s = U3_s(:,N_in-nc+1:N_in);  % nc smallest vectors (largest in normalized Laplacian) eigenvectors to span the subspace
    
    % Multimodal Laplacian   
    LL_s = L0_shifted + L1_shifted + L2_shifted + L3_shifted;
    UU_s = U0_s*U0_s' + U1_s*U1_s' + U2_s*U2_s' + U3_s*U3_s';
    
    Lmod_r = LL_s - gamma_opt*UU_s;
    [Umod,Smod,Vmod]=svd(Lmod_r);
    kerN = Vmod(:,1:nc); % nc largest eigenvectors
    % nornamlize to unit row norm
    for ii = 1:N_in
        kerNS(ii,:) = kerN(ii,:) ./ norm(kerN(ii,:)+eps);
    end
    
    MAXiter = 1000; % Maximum number of iterations for KMeans
    REPlic = 20; % Number of replications for KMeans
    warning off
    labels_est_MLG(1,:) = kmeans(kerNS,nc,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
  
    % Performance on in-sample data
    ACC_MLG_in(it)  = 1 - computeCE(labels_est_MLG,labels_in);
    NMI_MLG_in(it) = compute_nmi(labels_in,labels_est_MLG);
    Fscore_MLG_in(it) = compute_f(labels_in,labels_est_MLG);
    Rand_MLG_in(it) = RandIndex(labels_in,labels_est_MLG);
    Purity_MLG_in(it) = purFuc(labels_in,labels_est_MLG);
   
    % MLG: estimate bases using in-sample data
    [B_MLG, begB_MLG, enddB_MLG, mu_MLG]  = bases_estimation(X3_in,labels_est_MLG,dimSubspace);
    for l=1:nc
        X_outm = X3_out - mu_MLG(:,l);    % make data zero mean for distance calculation
        BB=B_MLG(:,begB_MLG(l):enddB_MLG(l));
        Xproj = (BB*BB')*X_outm;
        Dproj = X_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_MLG] = min(D); % out-of-sample labels
    clear D
    
    % Performance on out-of-sample data
    ACC_MLG_out(it)  = 1 - computeCE(A_MLG,labels_out);
    NMI_MLG_out(it) = compute_nmi(labels_out,A_MLG);
    Fscore_MLG_out(it) = compute_f(labels_out,A_MLG);
    Rand_MLG_out(it) = RandIndex(labels_out,A_MLG);
    Purity_MLG_out(it) = purFuc(labels_out,A_MLG);
            
    % Compute clustering performance of GMC LRSSC on layer 3 output
    [U3,S3,V3]=svd(L3_shifted);
    kerN = V3(:,1:nc); % nc largest eigenvectors
    % nornamlize to unit row norm
    for i = 1:N_in
        kerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
    end
    
    MAXiter = 1000; % Maximum number of iterations for KMeans
    REPlic = 20; % Number of replications for KMeans
    warning off
    labels_est_GMC_layer_3(1,:) = kmeans(kerNS,nc,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    
    % Performance on in-sample data
    ACC_GMC_in(it)  = 1 - computeCE(labels_est_GMC_layer_3,labels_in);
    NMI_GMC_in(it) = compute_nmi(labels_in,labels_est_GMC_layer_3);
    Fscore_GMC_in(it) = compute_f(labels_in,labels_est_GMC_layer_3);
    Rand_GMC_in(it) = RandIndex(labels_in,labels_est_GMC_layer_3);
    Purity_GMC_in(it) = purFuc(labels_in,labels_est_GMC_layer_3);
    
    % Estimate bases for out-of-sample data clustering
    [B_GMC_X3, begB_GMC_X3, enddB_GMC_X3, mu_GMC_X3]  = bases_estimation(X3_in,labels_est_GMC_layer_3,dimSubspace);
    for l=1:nc
        X_outm = X3_out - mu_GMC_X3(:,l);    % make data zero mean for distance calculation
        BB=B_GMC_X3(:,begB_GMC_X3(l):enddB_GMC_X3(l));
        Xproj = (BB*BB')*X_outm;
        Dproj = X_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_GMC] = min(D);
    clear D
    
    % Performance on out-of-sample data
    ACC_GMC_out(it)  = 1 - computeCE(A_GMC,labels_out);
    NMI_GMC_out(it) = compute_nmi(labels_out,A_GMC);
    Fscore_GMC_out(it) = compute_f(labels_out,A_GMC);
    Rand_GMC_out(it) = RandIndex(labels_out,A_GMC);
    Purity_GMC_out(it) = purFuc(labels_out,A_GMC);
       
    % DSC_L2 C matrix
    % Estimate performance on in-sample-data
    C = C_DSC_L2_net(index_in,index_in); % take in-sample submatrix
    A = adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace);
    N_in = size(A,1);
    
    % compute shifted Laplacian for DSC-L2 
    DN = diag( 1./sqrt(sum(A)+eps) );
    LDSC_shifted = speye(N_in) + DN*A*DN;
    [UDSC,SDSC,VDSC]=svd(LDSC_shifted);
    % compute clustering performance
    kerN = VDSC(:,1:nc); % nc largest eigenvectors
    % nornamlize to unit row norm
    for i = 1:N_in
        kerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
    end
    
    MAXiter = 1000; % Maximum number of iterations for KMeans
    REPlic = 20; % Number of replications for KMeans
    warning off
    labels_est_DSC_L2(1,:) = kmeans(kerNS,nc,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');

    % Performance on in-sample data
    ACC_DSC_in(it)  = 1 - computeCE(labels_est_DSC_L2,labels_in);
    NMI_DSC_in(it) = compute_nmi(labels_in,labels_est_DSC_L2);
    Fscore_DSC_in(it) = compute_f(labels_in,labels_est_DSC_L2);
    Rand_DSC_in(it) = RandIndex(labels_in,labels_est_DSC_L2);
    Purity_DSC_in(it) = purFuc(labels_in,labels_est_DSC_L2);
        
    % estimate bases using in-sample data    
    [B_DSC, begB_DSC, enddB_DSC, mu_DSC]  = bases_estimation(X3_in,labels_est_DSC_L2,dimSubspace);
    for l=1:nc
        X_outm = X3_out - mu_DSC(:,l);    % make data zero mean for distance calculation
        BB=B_DSC(:,begB_DSC(l):enddB_DSC(l));
        Xproj = (BB*BB')*X_outm;
        Dproj = X_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_DSC] = min(D);
    clear D    
    
    % Performance on out-of-sample data
    ACC_DSC_out(it)  = 1 - computeCE(A_DSC,labels_out);
    NMI_DSC_out(it) = compute_nmi(labels_out,A_DSC);
    Fscore_DSC_out(it) = compute_f(labels_out,A_DSC);
    Rand_DSC_out(it) = RandIndex(labels_out,A_DSC);
    Purity_DSC_out(it) = purFuc(labels_out,A_DSC);
end

save MLG_MNIST_Results ACC_MLG_in NMI_MLG_in Fscore_MLG_in Rand_MLG_in Purity_MLG_in ...
    ACC_MLG_out NMI_MLG_out Fscore_MLG_out Rand_MLG_out Purity_MLG_out...
    ACC_GMC_in NMI_GMC_in Fscore_GMC_in Rand_GMC_in Purity_GMC_in ...
    ACC_GMC_out NMI_GMC_out Fscore_GMC_out Rand_GMC_out Purity_GMC_out...
    ACC_DSC_in NMI_DSC_in Fscore_DSC_in Rand_DSC_in Purity_DSC_in ...
    ACC_DSC_out NMI_DSC_out Fscore_DSC_out Rand_DSC_out Purity_DSC_out...
 
   
display('MULTILAYER GRAPH CLUSTERING:')

display('In sample data:')

display('Mean ACC:')
mean(ACC_MLG_in)
display('Std ACC:')
std(ACC_MLG_in)

display('Mean NMI:')
mean(NMI_MLG_in)
display('Std NMI:')
std(NMI_MLG_in)

display('Mean Fscore:')
mean(Fscore_MLG_in)
display('Std Fscore:')
std(Fscore_MLG_in)

display('Mean Rand:')
mean(Rand_MLG_in)
display('Std Rand:')
std(Rand_MLG_in)

display('Mean Purity:')
mean(Purity_MLG_in)
display('Std Purity:')
std(Purity_MLG_in)

display('Out-of-sample data:')

display('Mean ACC:')
mean(ACC_MLG_out)
display('Std ACC:')
std(ACC_MLG_out)

display('Mean NMI:')
mean(NMI_MLG_out)
display('Std NMI:')
std(NMI_MLG_out)

display('Mean Fscore:')
mean(Fscore_MLG_out)
display('Std Fscore:')
std(Fscore_MLG_out)

display('Mean Rand:')
mean(Rand_MLG_out)
display('Std Rand:')
std(Rand_MLG_out)

display('Mean Purity:')
mean(Purity_MLG_out)
display('Std Purity:')
std(Purity_MLG_out)

display('GMC LRSSC ENCODER OUTPUT (LAYER 3):')

display('In sample data:')

display('Mean ACC:')
mean(ACC_GMC_in)
display('Std ACC:')
std(ACC_GMC_in)

display('Mean NMI:')
mean(NMI_GMC_in)
display('Std NMI:')
std(NMI_GMC_in)

display('Mean Fscore:')
mean(Fscore_GMC_in)
display('Std Fscore:')
std(Fscore_GMC_in)

display('Mean Rand:')
mean(Rand_GMC_in)
display('Std Rand:')
std(Rand_GMC_in)

display('Mean Purity:')
mean(Purity_GMC_in)
display('Std Purity:')
std(Purity_GMC_in)

display('Out-of-sample data:')

display('Mean ACC:')
mean(ACC_GMC_out)
display('Std ACC:')
std(ACC_GMC_out)

display('Mean NMI:')
mean(NMI_GMC_out)
display('Std NMI:')
std(NMI_GMC_out)

display('Mean Fscore:')
mean(Fscore_GMC_out)
display('Std Fscore:')
std(Fscore_GMC_out)

display('Mean Rand:')
mean(Rand_GMC_out)
display('Std Rand:')
std(Rand_GMC_out)

display('Mean Purity:')
mean(Purity_GMC_out)
display('Std Purity:')
std(Purity_GMC_out)

display('DSCL2 NET:')

display('In sample data:')

display('Mean ACC:')
mean(ACC_DSC_in)
display('Std ACC:')
std(ACC_DSC_in)

display('Mean NMI:')
mean(NMI_DSC_in)
display('Std NMI:')
std(NMI_DSC_in)

display('Mean Fscore:')
mean(Fscore_DSC_in)
display('Std Fscore:')
std(Fscore_DSC_in)

display('Mean Rand:')
mean(Rand_DSC_in)
display('Std Rand:')
std(Rand_DSC_in)

display('Mean Purity:')
mean(Purity_DSC_in)
display('Std Purity:')
std(Purity_DSC_in)

display('Out-of-sample data:')

display('Mean ACC:')
mean(ACC_DSC_out)
display('Std ACC:')
std(ACC_DSC_out)

display('Mean NMI:')
mean(NMI_DSC_out)
display('Std NMI:')
std(NMI_DSC_out)
display('Mean Fscore:')
mean(Fscore_DSC_out)
display('Std Fscore:')
std(Fscore_DSC_out)

display('Mean Rand:')
mean(Rand_DSC_out)
display('Std Rand:')
std(Rand_DSC_out)

display('Mean Purity:')
mean(Purity_DSC_out)
display('Std Purity:')
std(Purity_DSC_out)

% ranksum two sided Wilcox test of statistical significance
display("p-values MLG vs. DSC:")

display('In sample data:')
p_acc=ranksum(ACC_MLG_in,ACC_DSC_in)
p_nmi=ranksum(NMI_MLG_in,NMI_DSC_in)
p_fscore=ranksum(Fscore_MLG_in,Fscore_DSC_in)
p_rand=ranksum(Rand_MLG_in,Rand_DSC_in)
p_purity=ranksum(Purity_MLG_in,Purity_DSC_in)

display('Out-of-sample data:')
p_acc=ranksum(ACC_MLG_out,ACC_DSC_out)
p_nmi=ranksum(NMI_MLG_out,NMI_DSC_out)
p_fscore=ranksum(Fscore_MLG_out,Fscore_DSC_out)
p_rand=ranksum(Rand_MLG_out,Rand_DSC_out)
p_purity=ranksum(Purity_MLG_out,Purity_DSC_out)


display("p-values GMC vs. DSC:")

display('In sample data:')
p_acc=ranksum(ACC_GMC_in,ACC_DSC_in)
p_nmi=ranksum(NMI_GMC_in,NMI_DSC_in)
p_fscore=ranksum(Fscore_GMC_in,Fscore_DSC_in)
p_rand=ranksum(Rand_GMC_in,Rand_DSC_in)
p_purity=ranksum(Purity_GMC_in,Purity_DSC_in)

display('Out-of-sample data:')
p_acc=ranksum(ACC_GMC_out,ACC_DSC_out)
p_nmi=ranksum(NMI_GMC_out,NMI_DSC_out)
p_fscore=ranksum(Fscore_GMC_out,Fscore_DSC_out)
p_rand=ranksum(Rand_GMC_out,Rand_DSC_out)
p_purity=ranksum(Purity_GMC_out,Purity_DSC_out)


display("p-values MLG vs. GMC:")

display('In sample data:')
p_acc=ranksum(ACC_GMC_in,ACC_MLG_in)
p_nmi=ranksum(NMI_GMC_in,NMI_MLG_in)
p_fscore=ranksum(Fscore_GMC_in,Fscore_MLG_in)
p_rand=ranksum(Rand_GMC_in,Rand_MLG_in)
p_purity=ranksum(Purity_GMC_in,Purity_MLG_in)

display('Out-of-sample data:')
p_acc=ranksum(ACC_GMC_out,ACC_MLG_out)
p_nmi=ranksum(NMI_GMC_out,NMI_MLG_out)
p_fscore=ranksum(Fscore_GMC_out,Fscore_MLG_out)
p_rand=ranksum(Rand_GMC_out,Rand_MLG_out)
p_purity=ranksum(Purity_GMC_out,Purity_MLG_out)
