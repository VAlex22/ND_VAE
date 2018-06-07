function novelty_score = compute_novelty_score(normal_data, test_data, metric, use_gpu)
% normal_data: n1 x d x 2 normal data matrix of d-dimensional normal distributions (mean and std_dev)
% test_data: n2 x d x 2 test data matrix of d-dimensional normal distributions (mean and std_dev)
% metric: novelty score metric, can be euclidean, statistical or density
% use_gpu: specify whether to use gpu or not

if use_gpu
    normal_data = gpuArray(normal_data);
    test_data = gpuArray(test_data);
end

swithc metric
    case 'euclidean'
        novelty_score = bigmatrix_rowfun(@(x)dist_multi(x,normal_data), test_data);
    case 'statistical'
        novelty_score = bigmatrix_rowfun(@(x)stat_dist_multi(x,normal_data), test_data);
    case 'density'
        novelty_score = bigmatrix_rowfun(@(x)dens_multi(x,normal_data), test_data);
    otherwise
        disp('Unknown novelty metric, only euclidean, statistical and density supported');
if use_gpu
    novelty_score = gather(novelty_score);
end