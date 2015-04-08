%% Multimodal reterival by Share-Private modal on WIKI
% Sep 24, 2014
% Tang Xu
% tangxu AT stu.xidian.edu.cn
% modified: Nov 27, 2014

clear all;
%% Access directories
addpath(genpath('SPAMS'));
addpath(genpath('Data'));
%addpath(genpath('DomainAdaptDict'));
%load raw_features;


%% Load parmeters and dataset

load train_cat_name; % train label
load test_cat_name; %test label

load Q;   %label matrix

 load BOW_sift_new_1000;
 load bow_5000d;
 Xa_tr = BOW(:,1:2173)';
 Xb_tr = bow_5000d(:,1:2173)';

%  Xa_tr = I_tr;
%  Xb_tr = T_tr;

%%%%%%%%%%%%%%%%
% Xa_tr = BOW(:,1:2173);
% Xb_tr = train_bow_pca_1000;%(:,1:2173);
% train_bovw = BOW(:,1:2173);
% s = zeros(11,1);
% for ii = 1:10
%     cat_ind = find(train_cat_name==ii);
%     ind = size(cat_ind,1);
%     Xa_tr(:, s(ii)+1:s(ii)+ind) = I_tr(:,cat_ind);
%     Xb_tr(:, s(ii)+1:s(ii)+ind) = T_tr(:,cat_ind);
%     s(ii+1) = s(ii)+ind;
% end
% par.list_groups = s(1:10,1);
%%%%%%%%%%%%%%%%


%% Parameters setting
% par.mu = par.mu*1;
% % par.K 	= 50;
% % param.K = 50;
% % par.L	= 50;
% % param.L = 50;
%   par.K 	= 100;
%   param.K = 100;
%   par.L	= 100;
%   param.L = 100;
%   par.nIter = 200;
%   param.iter =200;
% % par.lambda1 = 0.2;
% % par.lambda2 =0.05;
% %  %par.mu = 1;
% % param.lambda        = par.lambda1; % not more than 20 non-zeros coefficients
% % param.lambda2       = par.lambda2;
% param.mode          = 2;       % penalized formulation
% param.approx=0;


% par.rho = 0.0500;
% par.lambda1 = 1;
% par.lambda2 = 0;%0.001;
% par.mu = 1;
% par.sqrtmu = 1;%0.1000;
% par.nu = 0.1; %0.1000;
% par.nIter = 50;
% par.epsilon = 0.005;%%%%%%%%
% %par.t0 = 5;
% par.K = 200;


param.K= 200;
param.lambda= 1;
param.iter= 50;
%param.L= 200;
param.lambda2= 0;%0.001;
param.mode= 2;
param.approx= 0;

%% normalize

Xa_tr = (Xa_tr - repmat(mean(Xa_tr),size(Xa_tr,1),1))./repmat(sqrt(var(Xa_tr,1)),size(Xa_tr,1),1);
Xb_tr = (Xb_tr - repmat(mean(Xb_tr),size(Xb_tr,1),1))./repmat(sqrt(var(Xb_tr,1)),size(Xb_tr,1),1);

Xa_tr(find(isnan(Xa_tr)))=0;
Xb_tr(find(isnan(Xb_tr)))=0;

% 
%  opt_pca.PCAthresh =0.95;
%  [Xatr_pca, xa_ev] = pcaIn(Xa_tr,opt_pca);% 
% [Xbtr_pca, xb_ev] = pcaIn(Xb_tr,opt_pca);
%  Xa_tr = Xatr_pca';
%  Xb_tr = Xbtr_pca';


 Xa_tr = Xa_tr';
 Xb_tr = Xb_tr';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Start our proposed algorithm on training samples %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Intialize D,A, and W(U)



%% Dictionary Learning

D = mexTrainDL([Xa_tr;Xb_tr], param);
Da = D(1:size(Xa_tr,1),:);
Db = D(size(Xa_tr,1)+1:end,:);



%%%%%%%%%%%%%%%%%%%
param.numThreads = -1; % all cores (-1 by default)
%param3.verbose=true; %verbosity, false by default
%param_d.lambda = 0.1; % regularization parameter
param.it0 = 5;% frequency for duality gap computations
param.max_it=20; % maximum number of iterations
param.L0=0.1;
param.tol=1e-3;
param.intercept=false;
param.pos=false;
param.loss = 'square';
%param_d.regul = 'l1linf';
param.regul = 'l1';
%A_a = zeros(size(D,2), size([Xa_tr;Xb_tr],2));
%A_b = A_a;


energy = 0;
maxiter = 20;

lambda1 = 1;
lambda2 = 0.1;
lambda3 = 0.1;
%lambda4 = 0.1;

param_a = param;
param_a.lambda = lambda2; 
%param_d.lambda = lambda2; 
para =  lambda3/lambda1;

A_a = full(mexLasso(Xa_tr, Da, param_a));
A_b = full(mexLasso(Xb_tr, Db, param_a));
Wb = Q * A_b' * inv(A_b * A_b' +  para * eye(size(A_b, 1)));
Wa = Q * A_a' * inv(A_a * A_a' +  para * eye(size(A_a, 1)));

for ii = 1:maxiter 
%% update A
[A_a ~] = mexFistaFlat([Xa_tr; sqrt(lambda1) * Q], [Da;sqrt(lambda1) * Wa], A_a, param_a);
[A_b ~] = mexFistaFlat([Xb_tr; sqrt(lambda1) * Q], [Db;sqrt(lambda1) * Wb], A_b, param_a);
%[W ~] = mexFistaFlat([Xa_tr;Xb_tr],D,W0,param4);
%A_a0 = A;

 %% Updating Ds and Dp
    
            for i=1:param.K
               ai        =    A_a(i,:);
               Y         =    Xa_tr-Da*A_a+Da(:,i)*ai;
               di        =    Y*ai';
               di        =    di./(norm(di,2) + eps);
               Da(:,i)    =    di;
            end
    
            for i=1:param.K
               ai        =    A_b(i,:);
               Y         =    Xb_tr-Db*A_b+Db(:,i)*ai;
               di        =    Y*ai';
               di        =    di./(norm(di,2) + eps);
               Db(:,i)    =    di;
            end
    
%% update D
%D = D';


%[D ~] = mexFistaFlat([Xa_tr;Xb_tr]',W',D,param3);
%%
%D_a = Da';
%[D_a ~] = mexFistaFlat(Xa_tr',A_a',D_a,param_d);
%%
%  for n =1:200
%  if(norm(D_a(n,:),2)<0.01)
%      D_a(n,:)=0;
%  end
%  end

%%
% D_b = Db';
% [D_b ~] = mexFistaFlat(Xb_tr',A_b',D_b,param_d); 

 % Da = D_a';
 % Db = D_b';
%%
%  for n =1:200
%  if(norm(D_b(n,:),2)<0.01)
%      D_b(n,:)=0;
%  end
%  end

%D= D';
%Da = D(1:size(Xa_tr,1),:);
%Db = D(size(Xa_tr,1)+1:end,:);


%% update W
%Wb_new = (1 - rho) * Wb  + rho * Wa * A_a * A_b' * inv(A_b * A_b' +  * eye(size(A_b, 1)));
%Wa_new = (1 - rho) * Wa  + rho * Wb * A_b * A_a' * inv(A_a * A_a' +  * eye(size(A_a, 1)));
Wb_new = Q * A_b' * inv(A_b * A_b' +  para * eye(size(A_b, 1)));
Wa_new = Q * A_a' * inv(A_a * A_a' +  para * eye(size(A_a, 1)));

Wb = Wb_new;
Wa = Wa_new;

%lamada = param_d.lambda;
%gamma = param4.lambda;
energy0 =energy;
% energy = norm(Xa_tr-Da*A_a,'fro') + norm(Xb_tr-Db*A_b,'fro')... 
%          + lambda1 * ( norm(Xa_tr-Da*A_a,'fro') + norm(Xb_tr-Db*A_b,'fro')) ...
%          + lambda2 * ( mixed1infnorm(Da') + mixed1infnorm(Db') )...
%          + lambda3 * ( mixed1infnorm(A_a) + mixed1infnorm(A_b) )...
%          + lambda4 * ( norm(Wa, Inf) + norm(Wa, Inf));
energy = norm(Xa_tr-Da*A_a,'fro') + norm(Xb_tr-Db*A_b,'fro')... 
         + lambda1 * ( norm(Q-Wa*A_a,'fro') + norm(Q-Wb*A_b,'fro')) ...
         + lambda2 * ( norm(A_a,1) + norm(A_b,1) )...
         + lambda3 * ( norm(Wa, 'fro') + norm(Wa, 'fro'));
     %clc;
fprintf('%f\n',energy);
fprintf('%i\n',ii);
if (abs(energy-energy0)<0.05)
    fprintf('%f\n',energy);
    break;
end
end
%%%%%%%%%%%%%%%%%%%

save('Da','Da');
save('Db','Db');
save('Wa','Wa');
save('Wb','Wb');
%save('xa_ev','xa_ev');
%save('xb_ev','xb_ev');


Xa_te = BOW(:,2174:end)';%IMAGE

Xb_te = bow_5000d(:,2174:end)';

Xa_te = (Xa_te - repmat(mean(Xa_te),size(Xa_te,1),1))./repmat(sqrt(var(Xa_te,1)),size(Xa_te,1),1);
Xb_te = (Xb_te - repmat(mean(Xb_te),size(Xb_te,1),1))./repmat(sqrt(var(Xb_te,1)),size(Xb_te,1),1);
Xa_te=Xa_te';
Xb_te=Xb_te';
Xa_te(find(isnan(Xa_te)))=0;
Xb_te(find(isnan(Xb_te)))=0;

%% Project the testing samples on the common feature space
Alpha_a = full(mexLasso(Xa_te, Da, param_a));
Alpha_b = full(mexLasso(Xb_te, Db, param_a));

a_coeffs = Wa * Alpha_a;
b_coeffs = Wb * Alpha_b;
%%
%load test_cat_name;
gt = test_cat_name';

fprintf('\n text-->image');
fprintf('%d\n', calculateMAP_C( b_coeffs', a_coeffs' , gt, gt, 50));
fprintf('%d\n', calculateMAP( b_coeffs', a_coeffs' , gt, gt, 50));
fprintf('\n image-->text');
fprintf('%d\n', calculateMAP_C(  a_coeffs' , b_coeffs', gt, gt, 50));
fprintf('%d\n', calculateMAP(  a_coeffs' , b_coeffs', gt, gt, 50));


%{


%% initialized coefficient

param2=param;
param2.lambda = 0.005;
param2.lambda2 = 0.0005;

Alpha_a = mexLasso(Xa_tr, Da, param);
Alpha_b = mexLasso(Xb_tr, Db, param);


clear D;

%par.groups = train_cat_name;
%par.Drls = Drls;
%par.nClass = 10;

%% Initialize Us, Up as I

Ua = eye(size(Da, 2));
Ub = eye(size(Db, 2));

% Iteratively solve D,A, and U

%[Alpha_a, Alpha_b, Xa_tr, Xb_tr, Da, Db, Wa, Wb, f] = coupled_DL(Alpha_a, Alpha_b, Xa_tr, Xb_tr, Da, Db, Ua, Ub, par);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Finish our proposed alorithm on training samples %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Xa_te = BOW(:,2174:end)';%IMAGE

Xb_te = bow_1000d(:,2174:end)';

Xa_te = (Xa_te - repmat(mean(Xa_te),size(Xa_te,1),1))./repmat(sqrt(var(Xa_te,1)),size(Xa_te,1),1);
Xb_te = (Xb_te - repmat(mean(Xb_te),size(Xb_te,1),1))./repmat(sqrt(var(Xb_te,1)),size(Xb_te,1),1);
Xa_te=Xa_te';
Xb_te=Xb_te';
Xa_te(find(isnan(Xa_te)))=0;
Xb_te(find(isnan(Xb_te)))=0;


%% Project the testing samples on the common feature space
Alpha_a = full(mexLasso(Xa_te, Da, param));
Alpha_b = full(mexLasso(Xb_te, Db, param));

a_coeffs = Wa * Alpha_a;
b_coeffs = Wb * Alpha_b;
%%
%load test_cat_name;
gt = test_cat_name';

fprintf('\n text-->image');
fprintf('%d\n', calculateMAP_C( b_coeffs', Alpha_a' , gt, gt, 50));
fprintf('%d\n', calculateMAP( b_coeffs', Alpha_a' , gt, gt, 50));
fprintf('\n image-->text');
fprintf('%d\n', calculateMAP_C(  a_coeffs' , Alpha_b', gt, gt, 50));
fprintf('%d\n', calculateMAP(  a_coeffs' , Alpha_b', gt, gt, 50));


%}

