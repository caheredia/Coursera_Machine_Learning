function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% grab the cost function from ex1, recall h = theta_transpose*X
J =  sum((X*theta-y).^2)/(2*m);
% Now we need to add the regularization term, copied structure from ex4, needed to be transposed
reg_term = lambda/(2*m)*( sum(theta(2:end,:).^2) );
% For complete cost function add reg_term
J = J + reg_term;

grad = sum((X*theta-y).*X)'/m + 7/m*[0 theta(2:end,:)']';


% =========================================================================

%grad = grad(:);

end
