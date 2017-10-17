function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%solve for z2
%add ones column to X 
X = [ones(m, 1) X];
%Solve for z2 which yields a 25 by 5000 matrix
z2 = Theta1*X';
%solve for a2, this yields a 25 by 5000 matrix
a2 = sigmoid(z2);
%add bias column to a2, first transpose 
a2 = [ones(size(a2', 1), 1) a2'];
%solving for z3
z3 = Theta2*a2';
%solve for a3
a3 = sigmoid(z3);

%finally, grabbing the max values for the predictions
[pval,p] = max(a3', [], 2);

% =========================================================================


end
