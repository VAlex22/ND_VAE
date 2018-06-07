function dist = dist_multi(data, neighbors)
sqdistfun = @(x,y)sqdistance(x',y'); % file exchage Michael Chen

sqdistmat = sqdistfun(data, neighbors); % pairwise squared-distance matrix

dist_sq = min(sqdistmat,[],2);
dist = sqrt(max(dist_sq,0)); % max because sqdistance.m sometimes yields negative values such as -2e-06
