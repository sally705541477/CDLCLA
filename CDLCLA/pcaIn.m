function  [X ev meanPCA] = pcaIn(Xin,opts)
%PCAIN Summary of this function goes here
%   Detailed explanation goes here


oTmp.disp = 0;
numSamples = size(Xin,1); % Xin is in the form where one sample is in one row

if isfield(opts,'meanMinus') && opts.meanMinus
    A = Xin*Xin';
else
%    Xin = Xin - repmat(mean(Xin,1),numSamples,1);
    A = Xin*Xin';
end

dim = size(Xin,2);
[ev ed] = eigs(A,numSamples-1,'LA',oTmp);

if (~isfield(opts,'PCAthresh'))
    opts.PCAthresh = 0.95; % default
end

if opts.PCAthresh > 1
    ev = ev(:,1:opts.PCAthresh);
else
    counter = 0;
    ed1 = diag(ed);
    tot = sum(ed1);
    rat = 0;
    runSum = 0;
    while rat < opts.PCAthresh
        counter = counter +1;
        runSum = runSum + ed1(counter);
        rat = runSum/tot;
    end
    ev = ev(:,1:counter);
    ed = ed(1:counter,1:counter);
    ev = Xin'*ev;
    ev = ev*ed^(-0.5);
end
X = Xin*ev;
meanPCA = mean(X);
X = X - repmat(mean(X,1),numSamples,1);
end


