function dens = dens_multi(data, neighbors)
densfun = @(x,y)average_gaussian(permute(x,[2,1,3]),permute(y,[2,1,3]));

densmat = densfun(data, neighbors); % pairwise density matrix

dens = mean(densmat,2);


