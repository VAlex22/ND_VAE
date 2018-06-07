function dist = stat_dist_multi(data, neighbors)
stdistfun = @(x,y)statistical_distance(permute(x,[2,1,3]),permute(y,[2,1,3]));

stdistmat = stdistfun(data, neighbors); % pairwise statistical-distance matrix

dist_sq = min(stdistmat,[],2);
dist = sqrt(max(dist_sq,0));