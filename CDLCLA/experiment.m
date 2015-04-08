load Da;
load Db;

%param.K= 200;
param.lambda= 1;
param.iter= 100;
%param.L= 200;
param.lambda2= 0;%0.001;
param.mode= 2;
param.approx= 0;

 Xa_te = I_te;
 Xb_te = T_te;
% Xa_te = BOW(:,2174:end)';%IMAGE
% Xb_te = bow_1000d(:,2174:end)';
%load xa_ev;
%load xb_ev;
Xa_te = (Xa_te - repmat(mean(Xa_te),size(Xa_te,1),1))./repmat(sqrt(var(Xa_te,1)),size(Xa_te,1),1);
Xb_te = (Xb_te - repmat(mean(Xb_te),size(Xb_te,1),1))./repmat(sqrt(var(Xb_te,1)),size(Xb_te,1),1);

Xa_te(find(isnan(Xa_te)))=0;
Xb_te(find(isnan(Xb_te)))=0;
%Xa_te = Xa_te * xa_ev;
%Xb_te = Xb_te * xb_ev;
Xa_te = Xa_te';
Xb_te = Xb_te';

%% Project the testing samples on the common feature space
Alpha_a = full(mexLasso(Xa_te, Da, param));
Alpha_b = full(mexLasso(Xb_te, Db, param));

a_coff =  Wa * Alpha_a;
b_coff =  Wb * Alpha_b;
%%
%load test_cat_name;
gt = test_cat_name';

fprintf('\n text-->image');
fprintf('%d\n', calculateMAP_C( b_coff', a_coff' , gt, gt, 50));
fprintf('%d\n', calculateMAP( b_coff', a_coff' , gt, gt, 50));
fprintf('\n image-->text');
fprintf('%d\n', calculateMAP_C(  a_coff' , b_coff', gt, gt, 50));
fprintf('%d\n', calculateMAP(  a_coff' , b_coff', gt, gt, 50));