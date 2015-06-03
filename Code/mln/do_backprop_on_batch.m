function [W, b] = do_backprop_on_batch(X, Y, W, b, eta, L, ftype, plt)

m     = size(X,2); % batch size
sum_w = cell(L,1);
sum_b = cell(L,1);
delta = cell(L,1);

% initialize update matrices
for i = 2:L
    sum_w{i} = zeros(size(W{i}));
    sum_b{i} = zeros(size(b{i}));
end

for mm = 1:m

    y        = Y(:,mm);
    h        = X(:,mm);
    [a, z]   = fwd_pass(h, W, b, L, ftype);
    delta{L} = (a{L} - y) .* sigma_deriv(z{L}, ftype);
    sum_w{L} = sum_w{L} + delta{L} * a{L-1}';
    sum_b{L} = sum_b{L} + delta{L};

    for l = (L-1):-1:2
        delta{l} = (W{l+1}' * delta{l+1}) .* sigma_deriv(z{l}, ftype);
        sum_w{l} = sum_w{l} + delta{l} * a{l-1}';
        sum_b{l} = sum_b{l} + delta{l};
    end

end

for l = 2:L
    W{l} = W{l} - (eta/m) * real(sum_w{l});
    b{l} = b{l} - (eta/m) * real(sum_b{l});
end

if plt; 

figure(1);
cla;
[~,sortidx] = sort(W{3});
sortedW1 = real(W{2});
sortedW1 = sortedW1(sortidx,:);
sortedW2 = real(W{3});
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
plot([0.5 16.5],[0 0],'k--');
getframe;

end