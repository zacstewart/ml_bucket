data = csvread('train.csv');
ids = data(2:end, 1);
data = data(2:end, 2:end); % dump headers and id col
tlen = floor(size(data, 1) / 100 * 60);
rand_indices = randperm(size(data,1));
train = data(rand_indices(1:tlen), :);
cv = data(rand_indices(tlen+1:end), :);

% training set
y = train(:, 1); X = [ones(size(train, 1), 1) train(:, 2:end)];
y_cv = cv(:, 1); X_cv = [ones(size(cv, 1), 1) cv(:, 2:end)];
m = length(y); n = size(X, 2);

%plot(X(:, 1), y);

fprintf('Running gradient descent...\n');
theta = 0.1 * ones(n, 1);
iterations = 1500;
alpha = 0.01;
lambda = 1;

% initial cost
[J, grad] = costFunction(X, y, theta, lambda);
fprintf('\nInitial cost:');
disp(J);

options = optimset('GradObj', 'on', 'MaxIter', iterations);
[theta, J, exit_flag] = fminunc(@(t) costFunction(X, y, t, lambda), theta, options);

fprintf('\nOptimized cost:');
disp(J);


