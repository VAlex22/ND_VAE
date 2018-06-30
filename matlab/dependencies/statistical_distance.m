function D = statistical_distance(A, B)
% Compute square Bhattacharyya distances between all pair of vectors.
%   A: d x n1 x 2 data matrix of d-dimensional normal distributions
%   B: d x n2 x 2 data matrix of d-dimensional normal distributions
%   D: n1 x n2 pairwise Bhattacharyya distance
% Implementation is based on the Michael Chen implementation of Euclidean 
% distance matrix computation (see sqdistance.m)

assert(size(A,1)==size(B,1));
assert(size(A,3)==size(B,3));
sig = bsxfun(@plus, A(:,:,2)'/2, reshape(B(:,:,2), 1, size(B, 1), [])/2);
mu = bsxfun(@minus, A(:,:,1)', reshape(B(:,:,1), 1, size(B, 1), []));
denom = bsxfun(@times, full(realsqrt(prod(A(:,:,2), 1)))', full(realsqrt(prod(B(:,:,2), 1))));
D = squeeze(0.125*dot(mu./sig, mu, 2)) + 0.5*log(squeeze(prod(sig, 2))./denom);
