function for_progress(iIjJkK)
% Display progress of (nested) for loop.
% Author: Vladimir Golkov
%
% Example:
% for i = 1:I, for j = 1:J, for k = 1:K
%	for_progress([i I j J k K])
%	your code here
% end
% 

ijk = iIjJkK(1:2:end);
IJK = iIjJkK(2:2:end);
IJK = [IJK(:); 1];

progress = 0;
for n = 1:numel(ijk)
	progress = progress + ijk(n)-1;
	progress = progress * IJK(n+1);
end
progress = (progress+1) / prod(IJK);

if all(ijk==1)
	fprintf('% 6.2f%%',100*progress)
else
	fprintf('\b\b\b\b\b\b\b% 6.2f%%',100*progress)
end
if progress==1
	fprintf('\n')
end

