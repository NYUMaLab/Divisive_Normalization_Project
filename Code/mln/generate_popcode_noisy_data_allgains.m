function [R,P,S] = generate_popcode_noisy_data_allgains(ndatapergain, nneuron, sig1_sq, sig2_sq, sigtc_sq)

sprefs = linspace(-60,60,nneuron);
gains  = 1 + 49 * rand(ndatapergain,1);

% S1  = zeros(ndatapergain,1); % 
% S1  = 30 * rand(ndatapergain,1) - 15;
S1  = [sqrt(sig1_sq) * randn(ndatapergain/2,1); sqrt(sig2_sq) * randn(ndatapergain/2,1)];
R1  = repmat(gains,1,nneuron) .* exp(-(repmat(S1,1,nneuron) - repmat(sprefs,ndatapergain,1)).^2 / (2*sigtc_sq));
R1  = R1 + sqrt(R1).*randn(size(R1));
AR1 = sum(R1,2) / sigtc_sq;
BR1 = sum(R1.*repmat(sprefs,ndatapergain,1),2) / sigtc_sq;
P1  = 1 ./ (1 + sqrt((1+sig1_sq*AR1)./(1+sig2_sq*AR1)) .* exp(-0.5 * ((sig1_sq - sig2_sq) .* BR1.^2) ./ ((1+sig1_sq*AR1).*(1+sig2_sq*AR1))));

R = R1;
S = S1;
P = P1;
