% Copyright (c) 2009, Michael Chen
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

function D = sqdistance(A, B, M)
% Compute square Euclidean or Mahalanobis distances between all pair of vectors.
   A: d x n1 data matrix
   B: d x n2 data matrix
   M: d x d  Mahalanobis matrix
   D: n1 x n2 pairwise square distance matrix
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1
    A = bsxfun(@minus,A,mean(A,2));
    S = full(dot(A,A,1));
    D = bsxfun(@plus,S,S')-full(2*(A'*A));
elseif nargin == 2
    assert(size(A,1)==size(B,1));
    D = bsxfun(@plus,full(dot(B,B,1)),full(dot(A,A,1))')-full(2*(A'*B));
elseif nargin == 3
    assert(size(A,1)==size(B,1));
    R = chol(M);
    RA = R*A;
    RB = R*B;
    D = bsxfun(@plus,full(dot(RB,RB,1)),full(dot(RA,RA,1))')-full(2*(RA'*RB));
