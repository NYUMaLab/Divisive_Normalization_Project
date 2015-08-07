function [pC, qC] = variational_example2_classification(sig, Ntrials, plots)

% Task of Qamar et al. PNAS 2012
% Implemented by Weiji Ma 2014058

set(0,'DefaultLineLineWidth',2)

variational = 1;

sig0 = 3;
sig1 = 12;

tau = 1/sig^2;
tau0 = 1/sig0^2;
tau1 = 1/sig1^2;
dtau = tau0 - tau1;

% Observation
C = rand>0;
x = randn*sig + randn*sig0 * (C==0) + randn*sig1 * (C==1);

% True posterior
s = linspace(-25,25,1000);
c = [0 1];
[S,C] = meshgrid(s,c);

tauC = (1-C)/sig0^2 + C/sig1^2;
mus = x*tau./(tau+tauC);
factor1 = sqrt(tau + tauC) .* exp(-(tau+tauC)/2.*(S-mus).^2);
factor2 = 1./sqrt(sig^2+1./tauC) .* exp(-x^2/2./(sig^2+1./tauC));

posterior = factor1 .* factor2;
posterior = posterior/sum(sum(posterior));

if ~variational
    figure;
    subplot(2,1,1);
    imagesc(s, c, posterior); axis xy; set(gca,'ytick',[0 1]); xlabel('orientation'); ylabel('class');
    h = colorbar; set(get(h,'title'),'string','posterior');
    title(['x = ' num2str(x,3)])
    h = subplot(2,1,2);
    plot(s, posterior); legend('C=0', 'C=1'); xlabel('orientation'); ylabel('posterior'); xlim([-25 25])
    ax=get(h,'Position');
    ax(4)=ax(4)*0.84;
    ax(3)=ax(3)*0.84;
    set(h,'Position',ax);
    
    
end
% Variational approximation

if variational
    
    maxiter = 10;
    lavec = NaN(1,maxiter);
    bvec = NaN(1,maxiter);
    EC= 0.5;
    
    KL = NaN(1,maxiter);
    Esvec = NaN(1,maxiter);
    ECvec = NaN(1,maxiter);
    
    if plots==1
        figure;
    end
    
    for i = 1:maxiter
        
        Es = tau*x/(tau + tau0 - dtau * EC);
        Vars = 1/(tau + tau0 - dtau * EC);
        Es2 = Es^2 + Vars;
        
        factor1 = normpdf(S, Es, sqrt(Vars));
        factor2 = sqrt(tau0-C*dtau) .* exp(Es2/2 * dtau * C);
        EC = 1/(1+ sqrt(tau0/tau1) * exp(-Es2*dtau/2));
        
        q = factor1 .* factor2;
        q = q/sum(sum(q));
        
        Esvec(i) = Es;
        ECvec(i) = EC;
        
        KL(i) = sum(sum(q .* log(q./posterior)));
                
        if plots ==1
            subplot(2,3,1) 
            plot(s, posterior); hold on;
            plot(s, q,'--'); hold off;
            legend('C=0', 'C=1','Location','Northwest'); xlabel('orientation'); ylabel('posterior');xlim([-25 25]); 
            title(['x = ' num2str(x,3) '    Variational approximation iteration ' num2str(i)]); set(gca,'ytick',[]);
                        
            subplot(2,3,2) 
            plot(s, sum(posterior),'k', s, sum(q),'k--'); 
            title('Marginal over s')            
            
            subplot(2,3,3) 
            bar([0, 1],[sum(posterior,2) sum(q,2)]); xlim([-.5 1.5]); title('Marginal over C')
            
            pause(0.5)
            
            subplot(2,3,4); 
            plot(1:maxiter, KL); ylabel('KL(q||p)'); xlabel('iteration'); xlim([1 maxiter]) 
            
            subplot(2,3,5);
            plot(1:maxiter, Esvec); ylabel('mean of q_s'); xlabel('iteration'); xlim([1 maxiter])
            subplot(2,3,6); 
            plot(1:maxiter, ECvec); ylabel('mean of q_C'); xlabel('iteration'); axis([1 maxiter 0 1]);
        end
    end
end
pC = sum(posterior(1,:));
qC = sum(q(1,:));

