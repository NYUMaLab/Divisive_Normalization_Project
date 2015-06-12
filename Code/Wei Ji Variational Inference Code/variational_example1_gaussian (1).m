% Bishop Section 10.1.3
% Implemented by Weiji Ma 20140508

clear all; close all;

variational = 1; % Set to 1 if you want the variational approximation

la0 = 0;
mu0 = 0;
a0 = 0;
b0 = 0;

x = randn(1,10);
xm = mean(x);
N = length(x);

mu = linspace(-1,1,100);
tau = linspace(0,2,101);
[M,T] = meshgrid(mu, tau);

% True posterior
aN = a0 + N/2;
muN = (N*xm+la0*mu0)/(N+la0);
b = b0 + sum((x-xm).^2)/2 + N*la0/2/(N+la0) * (xm-mu0)^2;

factor1 = sqrt((N+la0)*T) .* exp(-(N+la0)*T/2.*(M-muN).^2);
factor2 = T.^(aN-1) .* exp(-T * b);

posterior = factor1 .* factor2;
posterior = posterior/sum(sum(posterior));

if ~variational
    figure;
    imagesc(mu, tau, posterior); axis xy;
    xlabel('\mu'); ylabel('\tau')
    h = colorbar; set(get(h,'title'),'string','posterior');
end

% Variational approximation
if variational
    maxiter = 10;
    
    KL = NaN(1,maxiter);
    lavec = NaN(1,maxiter);
    bvec = NaN(1,maxiter);
    
    Etau= 10;
    aN = a0 + (N+1)/2;
    
    figure;
    for i = 1:maxiter
        i
        subplot(2,11,1:5);
        imagesc(mu, tau, posterior); axis xy; title('True posterior')
        xlabel('\mu'); ylabel('\tau')
        
        laN = (la0+N) * Etau;
        Emu2 = muN^2 + 1/laN;
        bN = b0 + sum(x.^2)/2 + la0*mu0^2/2 + (N+la0)*Emu2/2 - muN *(N*xm + la0*mu0);
        Etau = aN/bN;
        
        q_mu = sqrt(laN) .* exp(-laN/2.*(M-muN).^2);
        q_tau = T.^(aN-1) .* exp(-T * bN);
        q = q_mu .* q_tau;
        q = q/sum(sum(q));
        
        subplot(2,11,7:11);
        imagesc(mu, tau, q); axis xy; title(['Variational approximation iteration ' num2str(i)])
        xlabel('\mu'); set(gca,'ytick',[]')
        pause(0.5)
        
        bvec(i) = bN;
        lavec(i) = laN;
        KL(i) = sum(sum(q .* log((q+realmin)./(posterior+realmin)))); % realmin to avoid divergences; KL is just for visualization
        
        subplot(2,11,12:14);
        plot(1:maxiter, KL); ylabel('KL(q||p)'); xlabel('iteration')
        xlim([1 maxiter])
        subplot(2,11,16:18);
        plot(1:maxiter, 1./sqrt(lavec)); ylabel('std of q_{\mu}'); xlabel('iteration')
        axis([1 maxiter 0 1])
        subplot(2,11,20:22);
        plot(1:maxiter, aN./bvec); ylabel('mean of q_{\tau}'); xlabel('iteration')
        axis([1 maxiter 0 2]);
    end
    
end