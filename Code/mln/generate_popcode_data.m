function [R,P,S] = generate_popcode_data(ndatapergain, nneuron, sig1_sq, sig2_sq, sigtc_sq, gains)

sprefs = linspace(-60,60,nneuron);

S1  = 30 * rand(ndatapergain,1) - 15;
R1  = gains(1) * exp(-(repmat(S1,1,nneuron) - repmat(sprefs,ndatapergain,1)).^2 / (2*sigtc_sq));
AR1 = sum(R1,2) / sigtc_sq;
BR1 = sum(R1.*repmat(sprefs,ndatapergain,1),2) / sigtc_sq;
P1  = 1 ./ (1 + sqrt((1+sig1_sq*AR1)./(1+sig2_sq*AR1)) .* exp(-0.5 * ((sig1_sq - sig2_sq) .* BR1.^2) ./ ((1+sig1_sq*AR1).*(1+sig2_sq*AR1))));

S2  = 30 * rand(ndatapergain,1) - 15;
R2  = gains(2) * exp(-(repmat(S2,1,nneuron) - repmat(sprefs,ndatapergain,1)).^2 / (2*sigtc_sq));
AR2 = sum(R2,2) / sigtc_sq;
BR2 = sum(R2.*repmat(sprefs,ndatapergain,1),2) / sigtc_sq;
P2  = 1 ./ (1 + sqrt((1+sig1_sq*AR2)./(1+sig2_sq*AR2)) .* exp(-0.5 * ((sig1_sq - sig2_sq) .* BR2.^2) ./ ((1+sig1_sq*AR2).*(1+sig2_sq*AR2))));

S3  = 30 * rand(ndatapergain,1) - 15;
R3  = gains(3) * exp(-(repmat(S3,1,nneuron) - repmat(sprefs,ndatapergain,1)).^2 / (2*sigtc_sq));
AR3 = sum(R3,2) / sigtc_sq;
BR3 = sum(R3.*repmat(sprefs,ndatapergain,1),2) / sigtc_sq;
P3  = 1 ./ (1 + sqrt((1+sig1_sq*AR3)./(1+sig2_sq*AR3)) .* exp(-0.5 * ((sig1_sq - sig2_sq) .* BR3.^2) ./ ((1+sig1_sq*AR3).*(1+sig2_sq*AR3))));

R = [R1; R2; R3];
S = [S1; S2; S3];
P = [P1; P2; P3];
