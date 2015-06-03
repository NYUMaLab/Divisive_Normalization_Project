clear;

tic

% network parameters
L       = 3;
nneuron = 41;
nnode   = [nneuron 8 1];
ftype   = 'relu';

% training parameters
eta   = 0.001;
nepch = 10;
bsize = 2; 

% generate data
sig1_sq      = 3^2;
sig2_sq      = 12^2;
sigtc_sq     = 10^2;
gains        = [1 3 15];
ndatapergain = 3000;
% [R,P,S]      = generate_popcode_noisy_data(ndatapergain, nneuron, sig1_sq, sig2_sq, sigtc_sq, gains);
[R,P,S]      = generate_popcode_noisy_data_allgains(3*ndatapergain, nneuron, sig1_sq, sig2_sq, sigtc_sq);
ndata        = length(gains) * ndatapergain;
Xdata        = R';
Ydata        = P';

% initialize network parameters
W_init = cell(L,1);
b_init = cell(L,1);
for l=2:L
    W_init{l} = .1*rand(nnode(l),nnode(l-1));
    b_init{l} = .1*randn(nnode(l),1);
end


%% Train network with SGD
for e = 1:nepch
    
    pp = randperm(ndata);
    
    for bi = 1:(ndata/bsize)
    
        bbegin = (bi-1)*bsize+1;
        bend   = bi*bsize;
        X      = Xdata(:,pp(bbegin:bend));
        Y      = Ydata(:,pp(bbegin:bend));
        
        if (e == 1) && (bi == 1)
            W = W_init;
            b = b_init;
        end
        
        [W, b] = do_backprop_on_batch(X, Y, W, b, eta, L, ftype, 0);
    
    end
    
    fprintf('Epoch: %i done \n', e);

end


%% Evaluate trained network
ntestdatapergain = 3000;
testgains = [1 3 15];
[Rtest,Ptest,Stest] = generate_popcode_noisy_data(ntestdatapergain, nneuron, sig1_sq, sig2_sq, sigtc_sq, testgains);
ntestdata = length(testgains) * ntestdatapergain;
Xtestdata = Rtest';
Ytestdata = Ptest';
Yhat      = zeros(size(Ytestdata));
% ALLResps1 = zeros(16,ntestdata);
% ALLResps2 = zeros(1,ntestdata);

for ti = 1:ntestdata
    [a, ~]     = fwd_pass(Xtestdata(:,ti),W,b,L,ftype);
    Yhat(1,ti) = a{end};
%     ALLResps1(:,ti) = a{2};
%     ALLResps2(:,ti) = a{3};
end


%% Analyze and plot
nbins = 20;
edges = linspace(0,1,nbins);

Y1      = Ytestdata(1,1:ntestdatapergain);
Yhat1   = Yhat(1,1:ntestdatapergain);
[~,BIN] = histc(Y1,edges);
XX1 = zeros(nbins,1);
MM1 = zeros(nbins,1);
SS1 = zeros(nbins,1);
for i = 1:nbins
    indx     = find(BIN==(i-1));
    XX1(i)   = mean(Y1(indx));
    MM1(i)   = mean(Yhat1(indx));
    SS1(i)   = std(Yhat1(indx));
end

Y2      = Ytestdata(1,(ntestdatapergain+1):2*ntestdatapergain);
Yhat2   = Yhat(1,(ntestdatapergain+1):2*ntestdatapergain);
[~,BIN] = histc(Y2,edges);
XX2 = zeros(nbins,1);
MM2 = zeros(nbins,1);
SS2 = zeros(nbins,1);
for i = 1:nbins
    indx     = find(BIN==(i-1));
    XX2(i)   = mean(Y2(indx));
    MM2(i)   = mean(Yhat2(indx));
    SS2(i)   = std(Yhat2(indx));
end

Y3      = Ytestdata(1,(2*ntestdatapergain+1):3*ntestdatapergain);
Yhat3   = Yhat(1,(2*ntestdatapergain+1):3*ntestdatapergain);
[~,BIN] = histc(Y3,edges);
XX3 = zeros(nbins,1);
MM3 = zeros(nbins,1);
SS3 = zeros(nbins,1);
for i = 1:nbins
    indx     = find(BIN==(i-1));
    XX3(i)   = mean(Y3(indx));
    MM3(i)   = mean(Yhat3(indx));
    SS3(i)   = std(Yhat3(indx));
end
figure(5);
subplot(2,2,3); 
errorbar(real(XX1),MM1,SS1,'b-o'); hold on; 
errorbar(real(XX2),MM2,SS2,'g-o'); 
errorbar(real(XX3),MM3,SS3,'r-o'); 
% legend('gain=1','gain=3','gain=15');
% xlabel('Optimal posterior','FontSize',15);
% ylabel('Network posterior','FontSize',15);

hold on; axis square; box off; 
plot([0 1],[0 1],'k--','LineWidth',1.5); 
xlim([0 0.85]); ylim([0 0.85]);

figure(6);
[~,sortidx] = sort(W{3});
sortedW1 = real(W{2});
sortedW1 = sortedW1(sortidx,:);
sortedW2 = W{3};
sortedW2 = sortedW2(sortidx);

subplot(2,1,1); imagesc(sortedW1'); xlim([0.5 16.5]); 
xlabel('Hidden unit','FontSize',15);
ylabel('Input unit (deg.)','FontSize',15);
title('1-to-2 Weight matrix','FontSize',15,'Color','r');
set(gca,'YTick',[1,21,41]);
set(gca,'YTickLabel',{'-60','0','60'});

subplot(2,1,2); plot(sortedW2,'b-o','LineWidth',1.5); xlim([0.5 16.5])
xlabel('Hidden unit','FontSize',15);
title('2-to-3 Weight vector','FontSize',15,'Color','r');
ylim([-1 1]); hold on; 
plot([0.5 16.5],[0 0],'k--')
toc

