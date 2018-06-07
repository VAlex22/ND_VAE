function result = bigmatrix_rowfun(fcn, matrix, progress)
% Apply a row-wise function to a big matrix. The cool thing: If the calculation
% doesn't fit into memory then the calculation is split recursively in two.
% Author: Vladimir Golkov

N = size(matrix,1);
maxrows = 5; 

if ~exist('progress','var')
	progress = [1 N];
	for_progress(progress); % initial call without \b
end

try
	if N>maxrows
 		% disp(['Assuming ' num2str(N) ' rows are too many.'])
		error('MATLAB:nomem', 'Pre-emptive nomem error.')
    end
	result = fcn(matrix);
	for_progress(progress + [N-1 0]) % display progress in percent
catch e
	if N>1
        result1 = bigmatrix_rowfun(fcn, matrix(1:ceil(N/2), :, :), [progress(1) progress(2)]);
        result2 = bigmatrix_rowfun(fcn, matrix(ceil(N/2)+1 : end, :, :), [progress(1)+ceil(N/2) progress(2)]);
        result = [result1; result2];
    end
end
