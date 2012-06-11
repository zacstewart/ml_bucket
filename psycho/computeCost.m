function [J, grad] = computeCost(X, y, theta)
m = length(y);
J = 0;
h = X * theta;
sqrErrors = (h - y) .^2;
J = 1/ (2 * m) * sum(sqrErrors);
grad = 1/m * ((h-y)'*X);
end
