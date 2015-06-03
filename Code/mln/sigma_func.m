function s = sigma_func(h,ftype)

if strcmp(ftype,'tanh')
    s = tanh(h);
elseif strcmp(ftype,'relu')
    s = max(0,h);
end