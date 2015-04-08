%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description
%  compute MAP

%Input
%  queryset     n*dim_a data matrix 
%  targetset     n*dim_b data matrix
%  test_Y       n*1 label vector

%Output
%  map   MAP score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function map = calculateMAP_2( queryset, targetset, gt_que,gt_ret,R )

    
    %Dist = distMat(queryset,targetset);
    
    Dist = CosineDist(queryset, targetset);
    % Dist = norm(queryset - targetset);
    [asDist, index] = sort(Dist, 2, 'ascend');
    classIndex = gt_ret(index);
    ntest = size(unique(gt_ret));
    AP = [];
    
    [num c] = size( queryset );
    
    for k = 1:num
        reClassIndex = find(classIndex( k, 1:R) == gt_que(k));
        relength = length(reClassIndex);
        counts = [1:relength];
        AP =[AP sum(counts./reClassIndex+eps)/(relength+eps)];
       % AP =[AP sum(precision(reClassIndex))/11];
    end
    
    map = mean (AP);
end

function D = distMat(P1, P2)
%
% Euclidian distances between vectors
% each vector is one row
  
if nargin == 2
    P1 = double(P1);
    P2 = double(P2);
    
    X1=repmat(sum(P1.^2,2),[1 size(P2,1)]);
    X2=repmat(sum(P2.^2,2),[1 size(P1,1)]);
    R=P1*P2';
    D=real(sqrt(X1+X2'-2*R));
else
    P1 = double(P1);

    % each vector is one row
    X1=repmat(sum(P1.^2,2),[1 size(P1,1)]);
    R=P1*P1';
    D=X1+X1'-2*R;
    D = real(sqrt(D));
end
end