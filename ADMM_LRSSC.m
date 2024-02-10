% Non-convex Low Rank Sparse Subbspace Clustering for estimation of the
% matrix of coefficients:
% min\|X-XC\|_F^2 + \tau\|S\|_1 + \lambda\sum(fi(sig(L),a)
%
% INPUTS:
%   X: PxN data matrix with n samples and p features
%   lambda: regularization constant related to sparsity induced regualrizer of singular values
%   tau: weight of L1 norm regularization for entries of S
%   affine: 1 - affine subspace model; 0 - independent subspace model
%   opts:  Structure value with following fields:
%          opts.lambda:    coefficients for low-rank constraint
%          opts.rank_est:    parametar for nonconvex rank estimator, if <1
%          convex problem
%          opts.mu1:  penalty parameter for auxiliary variable C1 in augmented Lagrangian
%          opts.mu2:  penalty parameter for auxiliary variable C2 in augmented Lagrangian
%          opts.mu3:  penalty parameter for affine subspaces constraint in augmented Lagrangian
%          opts.max_mu1:  maximum  penalty parameter for mu1 parameter
%          opts.max_mu2:  maximum  penalty parameter for mu2 parameter
%          opts.max_mu3:  maximum  penalty parameter for mu3 parameter (used only for affine subspaces)
%          opts.rho1: step size for adaptively changing mu1, if 1 fixed mu1 is used
%          opts.rho2: step size for adaptively changing mu2, if 1 fixed mu2 is used
%          opts.rho3: step size for adaptively changing mu3, if 1 fixed mu3
%          is used (used only for affine subspaces)
%          opts.error_bound: error bound for convergence
%          opts.iter_max:  maximal number of iterations
%          opts.affine: true for the affine subspaces (default: false)
%          opts.soft_thr:  true for soft thresholding (minimization of L1
%          norm, convex problem), false for hard threshold (minimization of L0 norm, non-convex problem)
%          (default: false)
%
% OUTPUTS:
%   C: NxN matrix of coefficients
%   RMSE: error
%   error: ||X-XC||/||X||
%
% Ivica Kopriva and Maria Brbic , January, 2017.

function [C, error] = ADMM_LRSSC (X,opts)

if ~exist('opts', 'var')
    opts = [];
end

if isfield(opts, 'lambda');      lambda = opts.lambda;      end
if isfield(opts, 'alpha');      alpha = opts.alpha;      end % elra parameter
if isfield(opts, 'gamma');      gamma = opts.gamma;      end % gmc parameter
%if isfield(opts, 'rank_est');      rank_est = opts.rank_est;      end
if isfield(opts, 'err_thr');    err_thr = opts.err_thr;    end
if isfield(opts, 'iter_max');    iter_max = opts.iter_max;    end
if isfield(opts, 'affine');      affine = opts.affine;      end
if isfield(opts, 'l1_nucl_norm');      l1_nucl_norm = opts.l1_nucl_norm;      end
if isfield(opts, 'l0norm');      l0norm = opts.l0norm;      end
if isfield(opts, 'elra');      elra = opts.elra;      end
if isfield(opts, 'gmc');      gmc = opts.gmc;      end

%% initialization

[M,N]=size(X);

lambda1 = alpha/(1+lambda);
lambda2 = alpha*lambda/(1+lambda);

% setting penalty parameters for the ADMM
%mu1 = 10;
mu2 = alpha;

mu1 = 0.1;
%mu2 = alpha * 1;
%mu3 = alpha * 10;

max_mu1 = 1e6;
max_mu2 = 1e6;
max_mu3 = 1e6;

rho = 3;

J = zeros(N,N);  % auxiliary variable
C1 = J; C2 = J;

% Lagrange multpliers
LAM_1 = zeros(N,N);
LAM_2 = LAM_1;
lam_3 = zeros(1,N);

% Fixed precomputed term for J
tic;

XT = X'*X;

if affine==0
    Jf = inv(XT + (mu1 + mu2)*eye(N));
    J = Jf*(XT + mu1*C1 + mu2*C2 - LAM_1 - LAM_2);
    J = normc(J);  % necessary if X is column normalized (YaleB dataset !!!!!!!)
elseif affine==1
    Jf = inv(XT + (mu1 + mu2)*eye(N) + mu3*ones(N,N));
    lam_3 = zeros(1,N);
end

not_converged = 1;
iter=1;

while not_converged
    
    J_prev = J;
    % Update of J
    if ~affine
        J = Jf*(XT + mu1*C1 + mu2*C2 - LAM_1 - LAM_2);
    else
        J = Jf*(XT + mu1*C1 + mu2*C2 + mu3*ones(N,N) - LAM_1 - LAM_2 - ones(N,1)*lam_3);
    end
    J = normc(J);  % necessary if X is column normalized (YaleB dataset !!!!!!!)
    
    % Update of C1
    %[U,Sig,V]=svdsecon(J+LAM_1/mu1,k);
    [U Sig V]=svd(J+LAM_1/mu1,'econ');
    sig=diag(Sig)';
    
    thr = lambda1/mu1;
    if l1_nucl_norm   % nuclear norm - soft threshodling of singular values
        thr_rank=thr;
        indic=(sign(abs(sig)-thr_rank)+1)/2;
        sig_thr =max(0,abs(sig)-thr_rank).*sign(sig);
        
        [is inds]=sort(indic,'descend');
        ind = inds(1:sum(is));
        sig=sig_thr(ind);
        V=V(:,ind);
        U=U(:,ind);
        Sig=diag(sig);
        C1=U*Sig*V';
        
        %  C1=sigma_soft_thresh(J_new+Lambda3/mu3,lambda1/mu3);
        
    elseif l0norm   % hard threshodling of signular values
        thr_rank=sqrt(2*thr);
        sig_thr=sig.*((sign(abs(sig)-thr_rank)+1)/2);
        
        [is inds]=sort(sig_thr,'descend');
        ind = inds(1:sum(sign(is)));
        sig = sig_thr(ind);
        V=V(:,ind);
        U=U(:,ind);
        Sig=diag(sig);
        C1=U*Sig*V';
        %       %      pause
        %
    elseif gmc    % Apply enhanced low-rank approximation
        a = gamma/thr; % nonconvex rank estimator
        tmp = (sig-lambda1/mu1)/(1-a*(lambda1/mu1));
        sigm = max([tmp; zeros(1,length(sig))]);
        tmp = [sig; sigm];
        Sig = diag(min(tmp).*sign(sig));
        C1 = U*Sig*V';
 %   elseif gmc
  %      thr = lambda1/mu1;
   %     tmp = arrayfun(@(y) firm_thresh(y, thr, thr/gamma), sig);
    %    C1 = U*diag(tmp)*V';
    end
    %   C1 = normc(C1);
    
    % Update of C2
    tmp = J+LAM_2/mu2;
    thr = lambda2/mu2;
   
    if l1_nucl_norm    % soft thresholding
        C2=sign(tmp).*max(abs(tmp)-thr,0);
    elseif l0norm %|| elra % hard thresholding (l0 norm for sparsity)
        thr = sqrt(2*thr);
        C2 = tmp.*((sign(abs(tmp)-thr)+1)/2);
    elseif gmc
        a = gamma/thr;
        tmp2 = (abs(tmp)-thr)/(1-a*thr);
        tmp2 = max(tmp2, zeros(size(tmp2)));
        C2 = min(abs(tmp),tmp2).*sign(tmp);
  %  elseif gmc
   %     C2 = arrayfun(@(y) firm_thresh(y, thr, thr/gamma), tmp);
    end
    C2 = C2 - diag(diag(C2));
    
    % compute Lagrangian
    if l1_nucl_norm
        sparse_term = sum(sum(abs(C2)));
        low_rank_term = sum(sig);
%     elseif l0norm || elra
%        if elra
%             idx = find(abs(sig)<=1/a);
%             low_rank_term = sum(sig(idx)-a/2*sig(idx).^2);
%             low_rank_term = low_rank_term + 1/(2*a)*(length(sig) - length(idx)); 
%             
%             idx = find(abs(C2)<=1/a2);
%             sparse_term = sum(abs(C2(idx)) - a2/2*C2(idx).^2);
%             sparse_term = sparse_term + 1/(2*a2)*(N^2 - length(idx));
%         elseif l0norm
%             low_rank_term = sum(sign(sig));
%              sparse_term = sum(C2(:)~=0);
%         end
    end
    
%     if l0norm
%         L(iter) = 1/2*norm(X-X*J,'fro')^2 + lambda1*low_rank_term...
%             + lambda2*sparse_term;
%         
%         if iter>1
%             Lgrad = abs(L(iter)-L(iter-1));
%         else
%             Lgrad = L(iter);
%         end
%     end
    
    % Update of Lagrange multipliers
    LAM_1 = LAM_1 + mu1*(J - C1);
    LAM_2 = LAM_2 + mu2*(J - C2);
    
    mu1 = min(rho*mu1, max_mu1);
    mu2 = min(rho*mu2, max_mu2);

    if ~affine
        if rho~=1 || iter==1
            Jf = inv(XT + (mu1 + mu2)*eye(N));
        end
    else
        lam_3 = lam_3 + mu3*(ones(1,N)*J - ones(1,N));
        mu3 = min(rho*mu3, max_mu3);
        if rho~=1 || iter==1
            Jf = inv(XT + (mu1 + mu2)*eye(N) + mu3*ones(N,N));
        end
    end
    
    err1 = max(max(abs(J-C1)));
    err2 = max(max(abs(J-C2)));
    err3 = max(max(abs(J-J_prev)));
    
    % check convergence
    if iter>=iter_max
        not_converged = 0;
    end
    
    if gmc || elra || l0norm
        if err1<err_thr && err2<err_thr && err3<err_thr
            not_converged = 0;
        end
    else
        if Lgrad < err_thr
            not_converged = 0;
        end
    end
    
    iter = iter+1;
    
end
%iter
%C = normc(C1);
C=C1;

error = norm(X-X*J)/norm(X);

