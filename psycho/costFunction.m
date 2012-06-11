function [J, grad] = costFunction(X, y, theta, lambda)
m = length(y);
grad = zeros(size(theta));
h = X * theta;
J = (1/(2*m)) * sum((h-y) .^ 2) + (lambda/(2*m)) * (sum(theta(2:end) .^ 2));
grad(1) = (1/m) * sum((h-y)'*X);
grad(2:end) = ((1/m) * sum((h-y)' * X)) + (lambda/m)*theta(2:end);
