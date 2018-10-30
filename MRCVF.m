function CVF = MRCVF(X, Y, gamma, beta, lambda, lambda2, theta, a, MaxIter, ecr, minP)
% Learning Coherent Vector Fields for Robust Point Matching under Manifold Regularization. Neurocomputing, 2016, Vol 216, pp. 393-401
% Gang Wang
% //2016

[N, D]=size(X);
K=con_K(X,X,beta);
V=X; iter=1;  tecr=1; C=zeros(N,D); E=1;
sigma2=sum(sum((Y-X).^2))/(N*D);
newE = [];
newE2 = [];
while (iter<=MaxIter) && (tecr > ecr) && (sigma2 > 1e-8)
    % E-step
    E_old=E;
    [P, E]=get_P(Y,V, sigma2 ,gamma, a);
    newE = [newE,E];
    % graph-laplacian
    options=ml_options('Kernel','rbf', 'KernelParam', beta, ...
        'NN',6,'gamma_A',lambda,'gamma_I',lambda2);
    options.GraphWeights= 'heat';
    options.GraphWeightParam=sqrt(sigma2);
    L=laplacian(V,'nn',options);
    E=E+lambda/2*trace(C'*K*C) + lambda2 * trace(C'*K'*L*K*C);
    tecr=abs((E-E_old)/E);
    newE2 = [newE2,tecr];
    % M-step
    P = max(P, minP);
    inv_P = diag(1./P);
    C=(K+(lambda*eye(size(P,1))+lambda2*L*K)*sigma2*inv_P)\Y;
    V=K*C;
    Sp=sum(P);
    sigma2=sum(P'*sum((Y-V).^2, 2))/(Sp*D);
    numcorr = length(find(P > theta));
    gamma=numcorr/size(X, 1);
    if gamma > 0.95, gamma = 0.95; end
    if gamma < 0.05, gamma = 0.05; end
    iter=iter+1;
end
%
V=X; iter=1;  tecr=1; C=zeros(N,D); E=1;
sigma2=sum(sum((Y-X).^2))/(N*D);
while (iter<=MaxIter) && (tecr > ecr) && (sigma2 > 1e-8)
    % E-step
    E_old=E;
    [P, E]=get_P(Y,V, sigma2 ,gamma, a);
    % graph-laplacian
    options=ml_options('Kernel','rbf', 'KernelParam', beta, ...
        'NN',6,'gamma_A',lambda,'gamma_I',lambda2);
    options.GraphWeights= 'heat';
    options.GraphWeightParam= sqrt(sigma2);
    L=laplacian(X,'nn',options);
    E=E+lambda/2*trace(C'*K*C) + lambda2 * trace(C'*K'*L*K*C);
    tecr=abs((E-E_old)/E);
    % M-step
    P = max(P, minP);
    inv_P = diag(1./P);
    C=(K+(lambda*eye(size(P,1))+lambda2*L*K)*sigma2*inv_P)\Y;
    V=K*C;
    Sp=sum(P);
    sigma2=sum(P'*sum((Y-V).^2, 2))/(Sp*D);
    iter=iter+1;
end
%
CVF.X = X;
CVF.Y = Y;
CVF.beta = beta;
CVF.V=V;
CVF.C=C;
CVF.P = P;
CVF.Index = find(P > theta);
CVF.gamma=gamma;
CVF.sigma2=sigma2;
%
function K=con_K(x,y,beta)
[n, d]=size(x); [m, d]=size(y);
K=repmat(x,[1 1 m])-permute(repmat(y,[1 1 n]),[3 2 1]);
K=squeeze(sum(K.^2,2));
K=-beta * K;
K=exp(K);

%
function [P, E]=get_P(Y,V, sigma2 ,gamma, a)
D = size(Y, 2);
temp1 = exp(-sum((Y-V).^2,2)/(2*sigma2));
temp2 = (2*pi*sigma2)^(D/2)*(1-gamma)/(gamma*a);
P=temp1./(temp1+temp2);
E=P'*sum((Y-V).^2,2)/(2*sigma2)+sum(P)*log(sigma2)*D/2;
