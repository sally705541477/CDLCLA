function [ out ] = mixed1infnorm( M )
%MAXED1INFNORM Summary of this function goes here
%   Detailed explanation goes here

%%% L1-Linf norm%%%%%%%

t = size(M,1);
tp = zeros(t,1);
for j = 1:t
    tp(j)=norm(M(j,:),Inf);
end
out = norm(tp,1);
end

