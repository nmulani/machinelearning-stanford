function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

multipliers = [0.01 0.03 0.1 0.3 1 3 10 30];

nummult = length(multipliers);

errors = zeros(nummult, nummult);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

for a=1:nummult
      for b=1:nummult

        % Iterate over different options for C and sigma values
        % and determine errors for trained models
         C = multipliers(a);
         sigma = multipliers(b);

         model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

         yValPredict = svmPredict(model, Xval);

         errors(a, b) = mean(double(yValPredict ~= yval));

      end
end

% Return the C and sigma values where we had a minimum error
[errors, l] = find(errors==min(errors(:)));
C = multipliers(errors);
sigma = multipliers(l);




% =========================================================================

end
