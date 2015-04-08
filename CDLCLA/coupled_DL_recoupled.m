% % Coupled Dictionary and Feature Space Learning with Application to Cross-Domain Image Synthesis and Recognition.  
% % De-An Huang and Yu-Chiang Frank Wang
% % IEEE International Conference on Computer Vision (ICCV), 2013.
% %
% % Contact: Yu-Chiang Frank Wang (ycwang@citi.sinica.edu.tw)
% %
% % Main Function of Coupled Dictionary Learning
% % Input:
% % Alphap,Alphas: Initial sparse coefficient of two domains
% % Xp    ,Xs    : Image Data Pairs of two domains
% % Dp    ,Ds    : Initial Dictionaries
% % Wp    ,Ws    : Initial Projection Matrix
% % par          : Parameters 
% %
% %
% % Output
% % Alphap,Alphas: Output sparse coefficient of two domains
% % Dp    ,Ds    : Output Coupled Dictionaries
% % Up    ,Us    : Output Projection Matrix for Alpha
% % 
% 
% function [Alphap, Alphas, Xp, Xs, Dp, Ds, Wp, Ws, Up, Us, f] = coupled_DL_recoupled(Alphap, Alphas, Xp, Xs, Dp, Ds, Wp, Ws, par)
% 
% %% parameter setting
% 
% [dimX, numX]        =       size(Xp);
% dimY                =       size(Alphap, 1);
% numD                =       size(Dp, 2);
% rho                 =       par.rho;
% lambda1             =       par.lambda1;
% lambda2             =       par.lambda2;
% mu                  =       par.mu;
% sqrtmu              =       sqrt(mu);
% nu                  =       par.nu;
% nIter               =       par.nIter;
% t0                  =       par.t0;
% epsilon             =       par.epsilon;
% param.lambda        = 	    lambda1; % not more than 20 non-zeros coefficients
% param.lambda2       =       lambda2;
% param.mode          = 	    2;       % penalized formulation
% param.approx=0;
% param.K = par.K;
% param.L = par.L;
% f = 0;
% 
% 
% 
% %% Initialize Us, Up as I
% 
% Us = Ws; 
% Up = Wp; 
% 
% %% Iteratively solve D A U
%  param2=param;
%  param2.lambda = 0.01;
%  param2.lambda2 = 0.001;
%  %param2.regul='l1l2';
%  param2.regul='group-lasso-l2';
% %param2.groups = par.groups;
%  param.regul='group-lasso-l2';
%  param.groups = par.groups(1:200);
% for t = 1 : nIter
% 
%    % Ds=[Ds; sqrtmu * Ws] ./(repmat(sqrt(sum([Ds; sqrtmu * Ws ].^2)),[size([Ds; sqrtmu * Ws ],1) 1]));
%    % Dp=[Dp; sqrtmu * Wp]./(repmat(sqrt(sum([Dp; sqrtmu * Wp].^2)),[size([Dp; sqrtmu * Wp],1) 1]));
%     %% Updating Alphas and Alphap
%     f_prev = f;
%      %Alphas = mexL1L2BCD([Xs;sqrtmu * full(Alphap)]./(repmat(sqrt(sum([Xs;sqrtmu * full(Alphap) ].^2)),[size([Xs;sqrtmu * full(Alphap) ],1) 1])),...
%      %    [Ds; sqrtmu * Ws] ./(repmat(sqrt(sum([Ds; sqrtmu * Ws ].^2)),[size([Ds; sqrtmu * Ws ],1) 1])), full(Alphas), int32(par.list_groups) , param2);
%      %Alphap = mexL1L2BCD([Xp;sqrtmu * full(Alphas)]./(repmat(sqrt(sum([Xp;sqrtmu * full(Alphas) ].^2)),[size([Xp;sqrtmu * full(Alphas) ],1) 1])), ...
%      %    [Dp; sqrtmu * Wp]./(repmat(sqrt(sum([Dp; sqrtmu * Wp].^2)),[size([Dp; sqrtmu * Wp],1) 1])), full(Alphap), int32(par.list_groups) , param);
% %      tDs = [Ds; sqrtmu * Ws]';
% %      tXs = [Xs;sqrtmu * full(Alphap)]';
% %      tDp = [Dp; sqrtmu * Wp]';
% %      tXp = [Xp;sqrtmu * full(Alphas)]';
% %      
% %      Alphas = mexProximalFlat( tXs/tDs ,param2);
% %      Alphap = mexProximalFlat( tXp/tDp ,param2);
% %      Alphas = Alphas';
% %      Alphap = Alphap';
%      Alphas = mexLasso([Xs;sqrtmu * full(Alphap)], [Ds; sqrtmu * Ws],param);
%      Alphap = mexLasso([Xp;sqrtmu * full(Alphas)], [Dp; sqrtmu * Wp],param);
% %       Alphas = mexProximalFlat([Ds; sqrtmu * Ws]\[Xs;sqrtmu * full(Alphap)],param2);
% %       Alphap = mexProximalFlat([Dp; sqrtmu * Wp]\[Xp;sqrtmu * full(Alphas)],param2);
% %     Alphas = mexLasso([Xs;sqrtmu * full(Alphas)], [Ds; sqrtmu * Ws],param);
% %     Alphap = mexLasso([Xp;sqrtmu * full(Alphap)], [Dp; sqrtmu * Wp],param);
%     dictSize = par.K;
% 
% %     %% Updating Ds and Dp 
% % 
% %     for i=1:dictSize
% %        ai        =    Alphas(i,:);
% %        Y         =    Xs-Ds*Alphas+Ds(:,i)*ai;
% %        di        =    Y*ai';
% %        di        =    di./(norm(di,2) + eps);
% %        Ds(:,i)    =    di;
% %     end
% % 
% %     for i=1:dictSize
% %        ai        =    Alphap(i,:);
% %        Y         =    Xp-Dp*Alphap+Dp(:,i)*ai;
% %        di        =    Y*ai';
% %        di        =    di./(norm(di,2) + eps);
% %        Dp(:,i)    =    di;
% %     end
% 
%     %% Updating Ws and Wp => Updating Us and Up
%     ts = inv(Up)*Alphap;
%     tp = inv(Us)*Alphas;
% %     tp = inv(Up)*Alphap;
% %     ts = inv(Us)*Alphas;
%     
%     Us = (1 - rho) * Us  + rho * Alphas * ts' * inv( ts * ts' + par.nu * eye(size(Alphas, 1)));
%     Up = (1 - rho) * Up  + rho * Alphap * tp' * inv( tp * tp' + par.nu * eye(size(Alphap, 1)));
%     Ws = Up * inv(Us);
%     Wp = Us * inv(Up);
% 
%     %% Find if converge
% 
% %     P1 = Xp - Dp * Alphap;
% %     P1 = P1(:)'*P1(:) / 2;
%     P1 = 0;
%     P2 = lambda1 *  norm(Alphap, 1);    
%     P3 = Alphas - Wp * Alphap;
%     P3 = P3(:)'*P3(:) / 2;
%     P4 = nu * norm(Up, 'fro');
%     fp = 1 / 2 * P1 + P2 + mu * (P3 + P4);
%     
% %     P1 = Xs - Ds * Alphas;
% %     P1 = P1(:)'*P1(:) / 2;
%     P1 = 0;
%     P2 = lambda1 *  norm(Alphas, 1);    
%     P3 = Alphap - Ws * Alphas;
%     P3 = P3(:)'*P3(:) / 2;
%     P4 = nu * norm(Us, 'fro');  %%
%     fs = 1 / 2 * P1 + P2 + mu * (P3 + P4);
%     
%     f = fp + fs;
% 	
%         %% if converge then break
%     if (abs(f_prev - f) / f < epsilon)
%         fprintf('%d ',t);
%         break;
%     end
% 
% end
%     

% Coupled Dictionary and Feature Space Learning with Application to Cross-Domain Image Synthesis and Recognition.  
% De-An Huang and Yu-Chiang Frank Wang
% IEEE International Conference on Computer Vision (ICCV), 2013.
%
% Contact: Yu-Chiang Frank Wang (ycwang@citi.sinica.edu.tw)
%
% Main Function of Coupled Dictionary Learning
% Input:
% Alphap,Alphas: Initial sparse coefficient of two domains
% Xp    ,Xs    : Image Data Pairs of two domains
% Dp    ,Ds    : Initial Dictionaries
% Wp    ,Ws    : Initial Projection Matrix
% par          : Parameters 
%
%
% Output
% Alphap,Alphas: Output sparse coefficient of two domains
% Dp    ,Ds    : Output Coupled Dictionaries
% Up    ,Us    : Output Projection Matrix for Alpha
% 

function [Alpha_a, Alpha_b, Xa, Xb, Da, Db, Ua, Ub, f] = coupled_DL_recoupled(Alpha_a, Alpha_b, Xa, Xb, Da, Db, Ua, Ub, par)

%% parameter setting


rho                 =       par.rho;
lambda1             =       par.lambda1;
lambda2             =       par.lambda2;
mu                  =       par.mu;
sqrtmu              =       sqrt(mu);
nu                  =       par.nu;
nIter               =       par.nIter;
epsilon             =       par.epsilon;
param.lambda        = 	    lambda1; % not more than 20 non-zeros coefficients
param.lambda2       =       lambda2;
param.mode          = 	    2;       % penalized formulation
param.approx=0;
param.K = par.K;
%param.L = par.L;
nClass              =        par.nClass;
dictSize            =        par.K;
labels              =        par.groups;
f = 0;

param2=param;
param2.lambda = 0.005;
param2.lambda2 = 0.0005;



%% Iteratively solve D A U

for t = 1 : nIter
    
    f_prev = f;
 
    %% Updating Alphas and Alphap
    
    Alpha_a = mexLasso([Xa;sqrtmu * Ub * full(Alpha_b)], [Da; sqrtmu * Ua],param);
    Alpha_b = mexLasso([Xb;sqrtmu * Ua * full(Alpha_a)], [Db; sqrtmu * Ub],param);
    
    % sparse-group-lasso 
    param3 = param;
    param3.regul='sparse-group-lasso-l2';
    param3.groups = par.groups;
    
    param4 = param;
    param4.regul='sparse-group-lasso-l2';
    param4.groups = par.groups;
    
    Alpha_a = Alpha_a';
    Alpha_b = Alpha_b';
    Alpha_a = mexProximalFlat( full(Alpha_a) ,param4);
    Alpha_b = mexProximalFlat( full(Alpha_b) ,param3);
    Alpha_a = Alpha_a';
    Alpha_b = Alpha_b';    
    
    %% Updating Ds and Dp
    
    %     for i=1:dictSize
    %        ai        =    Alphas(i,:);
    %        Y         =    Xs-Ds*Alphas+Ds(:,i)*ai;
    %        di        =    Y*ai';
    %        di        =    di./(norm(di,2) + eps);
    %        Ds(:,i)    =    di;
    %     end
    %
    %     for i=1:dictSize
    %        ai        =    Alphap(i,:);
    %        Y         =    Xp-Dp*Alphap+Dp(:,i)*ai;
    %        di        =    Y*ai';
    %        di        =    di./(norm(di,2) + eps);
    %        Dp(:,i)    =    di;
    %     end
    
    %sub-dictionary update

    for ci = 1:nClass
        
        sub_dictSize = dictSize/nClass ;
        
        for i = 1 + sub_dictSize * (ci-1) : sub_dictSize * ci
            ai        =    Alpha_b(i,labels==ci);
            Y         =    Xb(:,labels==ci) - ...
                           Db(:,1+sub_dictSize*(ci-1):sub_dictSize*ci) * ...
                           Alpha_b(1+sub_dictSize*(ci-1):sub_dictSize*ci,labels==ci) + Db(:,i)*ai;
            di        =    Y*ai';
            di        =    di./(norm(di,2) + eps);
            Db(:,i)    =    di;
        end
        
        for i = 1 + sub_dictSize * (ci-1) : sub_dictSize * ci
            ai        =    Alpha_a(i,labels==ci);
            Y         =    Xa(:,labels==ci) - ...
                           Da(:,1+sub_dictSize*(ci-1):sub_dictSize*ci) * ...
                           Alpha_a(1+sub_dictSize*(ci-1):sub_dictSize*ci,labels==ci) + Da(:,i)*ai;
            di        =    Y*ai';
            di        =    di./(norm(di,2) + eps);
            Da(:,i)    =    di;
        end
        
    end
    % nClass =10;
    % new_Ds = zeros(size(Ds));
    % new_Dp = zeros(size(Dp));
    % Fish_ipts.D= Ds;%  the dictionary in the last interation
    % Fish_ipts.trls=par.groups;%  the labels of training data
    % Fish_par.fish_tau3=30;
    % Fish_par.fish_tau2=4;
    %     Fish_par.dls = par.Drls;
    %     for ci = 1:nClass
    % %         fprintf(['Updating dictionary, class: ' num2str(ci) '\n']);
    %         [new_Ds(:,par.Drls==ci),Delt{ci}]= FDDL_UpdateDi(Xs, Alphas,...
    %             ci,nClass,Fish_ipts,Fish_par);
    %     end
    %
    % Fish_ipts.D = Dp;
    %     for ci = 1:nClass
    % %         fprintf(['Updating dictionary, class: ' num2str(ci) '\n']);
    %         [new_Dp(:,par.Drls==ci),Delt{ci}]= FDDL_UpdateDi(Xp, Alphap,...
    %             ci,nClass,Fish_ipts,Fish_par);
    %     end
    %     Ds = new_Ds;
    %     Dp = new_Dp;
    
    %% Updating Us and Up
    
    Ua = (1 - rho) * Ua  + rho * Ub * Alpha_b * Alpha_a' * inv(Alpha_a * Alpha_a' + par.nu * eye(size(Alpha_a, 1)));
    Ub = (1 - rho) * Ub  + rho * Ua * Alpha_a * Alpha_b' * inv(Alpha_b * Alpha_b' + par.nu * eye(size(Alpha_b, 1)));
    
    %     ts = Up*Alphap;
    %     tp = Us*Alphas;
    %     Us = (1 - rho) * Us  + rho * Alphas * ts' * inv( ts * ts' + par.nu * eye(size(Alphas, 1)));
    %     Up = (1 - rho) * Up  + rho * Alphap * tp' * inv( tp * tp' + par.nu * eye(size(Alphap, 1)));
    %     Ws = Up * inv(Us);
    %     Wp = Us * inv(Up);
    
    
    
    %Ws = inv(Up) * Us;
    %Wp = inv(Us) * Up;
    %% Find if converge
    %P1 = 0;
    P1 = Xa - Da * Alpha_a;
    P1 = P1(:)'*P1(:) / 2;
    P2 = lambda1 *  norm(Alpha_a, 1);
    P3 = Ub * Alpha_b - Ua * Alpha_a;
    %P3 = Alphas - Wp * Alphap;
    P3 = P3(:)'*P3(:) / 2;
    P4 = nu * norm(Ua, 'fro');
    fp = 1 / 2 * P1 + P2 + mu * (P3 + P4);
    
    P1 = Xb - Db * Alpha_b;
    P1 = P1(:)'*P1(:) / 2;
    P2 = lambda1 *  norm(Alpha_b, 1);
    P3 = Ua * Alpha_a - Ub *Alpha_b;
    %P3 = Alphap - Ws * Alphas;
    P3 = P3(:)'*P3(:) / 2;
    P4 = nu * norm(Ub, 'fro');  %%
    fs = 1 / 2 * P1 + P2 + mu * (P3 + P4);
    
    f = fp + fs;
    
    %% if converge then break
    if (abs(f_prev - f) / f < epsilon)
        break;
    end
    
end
    
