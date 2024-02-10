%% MLGDSC_GMC_WHOLE_SET_COIL20 (Multilayer graph deep subspace clustering)
% I. Kopriva 2023-07

%%
clear
close all

%% parameters
robust_flag = 1; % 0: all modes are equally important; 
                 % 1: weights based on norm of SE error
                 % 2: weights based on F1 score

C1_flag = 2;     % 1: C=DSC_NET; 2: C learned by GMC LRSSC 
              
adj_matrix_flag=2; % 1: symmetric matrix and shifted Laplacian domain
                   % 2: adjacency matrix in angular domain with delta=4 and shifted Laplacian  

delta = 4;       % post-processing of angular data adjacency matrix ( delta > 1: 4 or 8)     

% ranges for grid-search based crossvalidation of the hyperparameters for
% GMC LRSSC algortithm
lambda = 0.7%0.0:0.1:1.0;
alpha = 1%0.2:0.2:1.6;
gamma_gmc = 0.6%0.3:0.1:1;

% Number of evaluations
numit = 10;

%%
cd  .\COIL20_1

% inpup data
load COIL20.mat
X0=transpose(fea); % columns of X represent vectorized data of squared images
i1=32; i2=32; N=1440; nc=20; % 1440 images of 20 objects (72 images per object) (each image is 32x32 pixels)
clear fea;

X1=csvread('coil1.csv');
X1 = transpose(X1);

labels=gnd;   % to be used for oracle based validation
clear gnd
cd ..

wbh_0 = waitbar(0,'Crossvalidation for the whole algorithm. Please wait ...');
    
for iter =1:numit
     waitbar(iter/numit,wbh_0);
    
    %% Cross-validate (tune) GMC_LRSSC algorithm for each "view" independently        
    wbh = waitbar(0,'Crossvalidation for lambda, alpha and gamma. Please wait ...');   
    for i=1:length(alpha)
        waitbar(i/length(alpha),wbh);
        for j=1:length(lambda)
            for k=1:length(gamma_gmc)
                options = struct('lambda',lambda(j),'alpha',alpha(i),'rank_est',0.6,'gamma',gamma_gmc(k),...
                    'err_thr',1e-4,'iter_max',100, 'affine',false,...
                    'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
                
                [C, error] = ADMM_LRSSC(normc(X0),options);
                if adj_matrix_flag == 1
                    A = BuildAdjacency(C);
                elseif adj_matrix_flag == 2
                    A = adjacency_matrix_angular_domain(C, delta);
                end
                labels_est = SpectralClustering_shifted_Laplacian(A,nc);
                CE0(i,j,k)  = computeCE(labels_est,labels);
                disp("To je !!!!!!!!!!!!!")
                disp(CE0(i,j,k))
                if C1_flag == 1
                    C=csvread('coil20_matrica_c.csv');
                elseif C1_flag == 2
                    [C, error] = ADMM_LRSSC(normc(X0),options);
                end
                if adj_matrix_flag == 1
                    A = BuildAdjacency(C);
                elseif adj_matrix_flag == 2
                    A = adjacency_matrix_angular_domain(C, delta);
                end
                labels_est = SpectralClustering_shifted_Laplacian(A,nc);
                CE1(i,j,k)  = computeCE(labels_est,labels);
            end
        end
    end
    close(wbh)
    
    cemin_0=1; cemin_1=1;
    for i=1:length(alpha)
        for j=1:length(lambda)
            for k=1:length(gamma_gmc)
                if cemin_0 > CE0(i,j,k)
                    cemin_0 = CE0(i,j,k);
                    alpha_0=alpha(i);
                    lambda_0=lambda(j);
                    gamma_0=gamma_gmc(k);
                end
                
                if C1_flag == 2
                    if cemin_1 > CE1(i,j,k)
                        cemin_1 = CE1(i,j,k);
                        alpha_1=alpha(i);
                        lambda_1=lambda(j);
                        gamma_1=gamma_gmc(k);
                    end
                end
            end
        end
    end
    
%     alpha_0
%     lambda_0
%     gamma_0
%     
%     alpha_1
%     lambda_1
%     gamma_1
%     pause
    
    %% crossvalidate gamma
    gamma=0:0.1:1;
    
    wbh = waitbar(0,'Crossvalidation for gamma. Please wait ...');  
    for i=1:length(gamma)
        waitbar(i/length(gamma));
        
        % Layer 0: input data
        options = struct('lambda',lambda_0,'alpha',alpha_0,'rank_est',0.6,'gamma',gamma_0,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        [C, error] = ADMM_LRSSC(normc(X0),options);
        if adj_matrix_flag == 1
            A = BuildAdjacency(C);
        elseif adj_matrix_flag == 2
            A = adjacency_matrix_angular_domain(C, delta);
        end
        
        if robust_flag == 1
            % estimate influence of self-expression error
            a_0 = 1/norm(normc(X0) - normc(X0)*C,'fro');
          %  a_0 = a_0 * a_0;
          %              a_0 = a_0 * a_0;
        elseif robust_flag == 2           
            labels_est = SpectralClustering_shifted_Laplacian(A,nc);
            a_0 = compute_f(labels,labels_est);
        elseif robust_flag == 0
            a_0=1/4;
        end     
        N = size(A,1);
        
        % compute shifted Laplacian for input data (layer 0)
        DN = diag( 1./sqrt(sum(A)+eps) );
        L0_shifted = speye(N) + DN*A*DN;
        [U0_s,S0_s,~] = svd(L0_shifted);
        U0_s = U0_s(:,1:nc);  % nc largest eigenvectors to span the subspace
        
        % Layer 1       
        if C1_flag == 1
           C=csvread('coil20_matrica_c.csv');
        elseif C1_flag == 2
            options = struct('lambda',lambda_1,'alpha',alpha_1,'rank_est',0.6,'gamma',gamma_1,...
                'err_thr',1e-4,'iter_max',100, 'affine',false,...
                'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
            [C, error] = ADMM_LRSSC(normc(X1),options);
        end
        if adj_matrix_flag == 1
            A = BuildAdjacency(C);
        elseif adj_matrix_flag == 2
            A = adjacency_matrix_angular_domain(C, delta);
        end  
        
         if robust_flag == 1
             % estimate influence of self-expression error
             a_1 = 1/norm(normc(X1) - normc(X1)*C,'fro');
           %  a_1 = a_1 * a_1;
          %               a_1 = a_1 * a_1;
         elseif robust_flag == 2
             labels_est = SpectralClustering_shifted_Laplacian(A,nc);
             a_1 = compute_f(labels,labels_est);
         elseif robust_flag == 0
             a_1=1/4;
         end
         N = size(A,1);
        
        % compute shifted Laplacian for input data (layer 1)
        DN = diag( 1./sqrt(sum(A)+eps) );
        L1_shifted = speye(N) + DN*A*DN;
        [U1_s,S1_s,~] = svd(L1_shifted);
        U1_s = U1_s(:,1:nc);    
        
        % robust multimodal Laplacian
        sum_a = a_0 + a_1;
        an_0 = a_0/sum_a;
        an_1 = a_1/sum_a;
        
        LL_s = an_0*L0_shifted + an_1*L1_shifted;
        UU_s = an_0*U0_s*U0_s' + an_1*U1_s*U1_s';
        
        Lmod_r = LL_s - gamma(i)*UU_s;
        [Umod,Smod,Vmod]=svd(Lmod_r);
        kerN = Vmod(:,1:nc); % nc largest eigenvectors        
        % nornamlize to unit row norm
        for ii = 1:N
            kerNS(ii,:) = kerN(ii,:) ./ norm(kerN(ii,:)+eps);
        end
        
        MAXiter = 1000; % Maximum number of iterations for KMeans
        REPlic = 20; % Number of replications for KMeans
        warning off
        labels_est = kmeans(kerNS,nc,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
        CE_gamma(i)=computeCE(labels_est,labels)
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
    gamma_iter(iter)=gamma_opt
    %% Estimate clustering quality metrics
    
    % Layer 0: input data
    options = struct('lambda',lambda_0,'alpha',alpha_0,'rank_est',0.6,'gamma',gamma_0,...
        'err_thr',1e-4,'iter_max',100, 'affine',false,...
        'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
    [C, error] = ADMM_LRSSC(normc(X0),options);
    if adj_matrix_flag == 1
        A = BuildAdjacency(C);
    elseif adj_matrix_flag == 2
        A = adjacency_matrix_angular_domain(C, delta);
    end
    
    if robust_flag == 1
        % estimate influence of self-expression error
        a_0 = 1/norm(normc(X0) - normc(X0)*C,'fro');
      %  a_0 = a_0 * a_0;
       %           a_0 = a_0 * a_0;  
    elseif robust_flag == 2
        labels_est = SpectralClustering_shifted_Laplacian(A,nc);
        a_0 = compute_f(labels,labels_est);
    elseif robust_flag == 0
        a_0=1/4;
    end
    N = size(A,1);
    
    % compute shifted Laplacian for input data (layer 0)
    DN = diag( 1./sqrt(sum(A)+eps) );
    L0_shifted = speye(N) + DN * A * DN;
    [U0_s,S0_s,~] = svd(L0_shifted);
    U0_s = U0_s(:,N-nc+1:N);
    
    % Layer 1
    if C1_flag == 1
       C=csvread('coil20_matrica_c.csv')
    elseif C1_flag == 2
        options = struct('lambda',lambda_1,'alpha',alpha_1,'rank_est',0.6,'gamma',gamma_1,...
            'err_thr',1e-4,'iter_max',100, 'affine',false,...
            'l1_nucl_norm',false,'l0norm',false,'elra',false, 'gmc',true);
        [C, error] = ADMM_LRSSC(normc(X1),options);
    end
    if adj_matrix_flag == 1
        A = BuildAdjacency(C);
    elseif adj_matrix_flag == 2
        A = adjacency_matrix_angular_domain(C, delta);
    end
    
    if robust_flag == 1
        % estimate influence of self-expression error
        a_1 = 1/norm(normc(X1) - normc(X1)*C,'fro');
       % a_1 = a_1 * a_1;
       % a_1 = a_1 * a_1;
    elseif robust_flag == 2
        labels_est = SpectralClustering_shifted_Laplacian(A,nc);
        a_1 = compute_f(labels,labels_est);
    elseif robust_flag == 0
        a_1=1/4;
    end
    N = size(A,1);

    % compute shifted Laplacian for input data (layer 1)
    DN = diag( 1./sqrt(sum(A)+eps) );
    L1_shifted = speye(N) + DN*A*DN;
    [U1_s,S1_s,~] = svd(L1_shifted);
    U1_s = U1_s(:,1:nc);
      
    % robust multimodal Laplacian
    sum_a = a_0 + a_1;
    an_0 = a_0/sum_a;
    an_1 = a_1/sum_a;
    
    LL_s = an_0*L0_shifted + an_1*L1_shifted;
    UU_s = an_0*U0_s*U0_s' + an_1*U1_s*U1_s';
    
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
    labels_est = kmeans(kerNS,nc,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    
    % Performance measures
    ACC(iter)  = 1 - computeCE(labels_est,labels)
    NMI(iter) = compute_nmi(labels,labels_est)
    Fscore(iter) = compute_f(labels,labels_est)
    
    % compute clustering performance on encoder output (layer 1)
    %[U1,S1,V1]=svd(L1_shifted);
    %kerN = V1(:,1:nc); % nc largest eigenvectors
    % nornamlize to unit row norm
    %for i = 1:N
    %    kerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
    %end
    
    MAXiter = 1000; % Maximum number of iterations for KMeans
    REPlic = 20; % Number of replications for KMeans
    warning off
    %labels_est = kmeans(kerNS,nc,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    
    % Performance measures
    %ACC_1(iter)  = 1 - computeCE(labels_est,labels)
    %NMI_1(iter) = compute_nmi(labels,labels_est)
    %Fscore_1(iter) = compute_f(labels,labels_est)
end
close(wbh_0)

display('Multilayer graph:')
display('Mean ACC:')
mean(ACC)
display('Std ACC:')
std(ACC)

display('Mean NMI:')
mean(NMI)
display('Std NMI:')
std(NMI)

display('Mean Fscore:')
mean(Fscore)
display('Std Fscore:')
std(Fscore)
%% 

ACC_1 = [ 70.83, 68.13, 65.90, 68.13, 69.17, 70.83, 70.90, 70.83, 68.13, 64.38]
NMI_1 = [85.06, 83.54, 83.77, 83.54, 83.51, 85.06, 85.13, 85.06, 83.54, 82.29]
Fscore_1 = [67.16, 63.61, 64.48, 63.61, 63.21, 67.16, 67.23, 67.16, 63.61, 61.00]
display('Encoder output (layer 1):')
display('Mean ACC:')
mean(ACC_1)
display('Std ACC:')
std(ACC_1)

display('Mean NMI:')
mean(NMI_1)
display('Std NMI:')
std(NMI_1)

display('Mean Fscore:')
mean(Fscore_1)
display('Std Fscore:')
std(Fscore_1)

% ranksum two sided Wilcox test of statistical significance
p_acc=ranksum(ACC_1,ACC)
p_nmi=ranksum(NMI_1,NMI)
p_fscore=ranksum(Fscore_1,Fscore)
