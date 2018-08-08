function D = average_gaussian(A, B)

assert(size(A,1)==size(B,1));
assert(size(A,3)==size(B,3));

mu = bsxfun(@minus, A(:,:,1)', reshape(B(:,:,1), 1, size(B, 1), []));
sig = permute(A(:,:,2), [2, 1]);

D = squeeze(exp(-0.5*dot(mu./(sig.^2), mu, 2)))./(squeeze(prod(sig, 2)*sqrt(2*pi)));
