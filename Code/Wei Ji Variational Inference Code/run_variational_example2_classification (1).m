clear all; close all;
sigvec = [3];
Ntrials = 1;
plots = 1;

for k = 1:length(sigvec)
    sig = sigvec(k)
    
    for j=1:Ntrials
        [pC(j,k) qC(j,k)] = variational_example2_classification(sig, Ntrials, 1);
    end
end

h = [0 0 1; 1 0 0 ; 0 0 0];

if Ntrials > 1
    figure;
    for k=1:length(sigvec)
        scatter(pC(:,k), qC(:,k),[],h(k,:)); hold on;
    end
    plot([0 1], [0 1],'k--');xlabel('p(C=1)'); ylabel('q(C=1)')
    legend('\sigma = 1', '\sigma = 2', '\sigma = 10','Location','Best')
end