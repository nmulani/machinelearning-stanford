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


% FIRST LAYER COMPUTATIONS

% Add a column of ones for our 'bias units'
a1 = [ones(m, 1) X];

% Multiply inputs by our first set of parameters
z2 = a1 * Theta1';

% Compute sigmoid values of products
z2v2 = sigmoid(z2);

% SECOND LAYER computations

% Add a column of ones for our 'bias units'
a2 = [ones(size(z2, 1), 1) z2v2];

% Compute sigmoid values of products of a2 values and our second set of parameters
a3 = sigmoid(a2 * Theta2');

% RETURN predictions

% Find the values and indices(classes) of maximum probabilities for each training example
[Y, I] = max(a3, [], 2);

% The indices correspond to the predicted class values for each training example
p = I;




% =========================================================================


end
