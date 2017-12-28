function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


% File through centroids
for i = 1:K
  % select for ith centroid in idx vector, making all other centroid values zero 
  i_idx = any(idx== i,2);
  % Find and sum all non zero examlpes, i.e. corresponding to ith centroid
  non_zero = sum(X.*i_idx);
  % Devide by # of examples and assign as new centroid location
  centroids(i,:) = non_zero/ sum(any(X.*i_idx,2));
end
  





% =============================================================


end

