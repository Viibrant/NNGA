% Load the iris dataset
load fisheriris
inputs = zscore(meas); % standardize input features
targets = categorical(species); % target labels

% split into training and validation sets
cvp = cvpartition(size(inputs,1),'Holdout',0.3);
idxTrain = training(cvp);
idxValidation = test(cvp);

inputsTrain = inputs(idxTrain,:);
targetsTrain = targets(idxTrain);

inputsValidation = inputs(idxValidation,:);
targetsValidation = targets(idxValidation);

% Define the architecture of the network
layers = [ ...
    featureInputLayer(4)
    
      fullyConnectedLayer(8)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
      fullyConnectedLayer(16)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
    fullyConnectedLayer(32)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
    fullyConnectedLayer(32)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
    fullyConnectedLayer(64)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
    fullyConnectedLayer(64)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
    fullyConnectedLayer(32)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
    fullyConnectedLayer(32)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
    fullyConnectedLayer(16)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
      fullyConnectedLayer(8)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
      fullyConnectedLayer(4)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.2)
    
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];


% Specify the training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',500, ...
    'InitialLearnRate',0.25, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{inputsValidation, targetsValidation}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network
net = trainNetwork(inputsTrain, targetsTrain, layers, options);

% Generate predictions on the validation data
YPred = classify(net, inputsValidation);


% Apply PCA
[coeff,score,latent] = pca(inputs);

% Plot the first two principal components
figure;
scatter(score(:,1), score(:,2));
title('PCA of Iris Dataset');
xlabel('First Principal Component');
ylabel('Second Principal Component');

% Create a confusion matrix
cm = confusionchart(targetsValidation, YPred);
title('Confusion Matrix for Validation Data');

% Generate a grid of points within the range of the PCA scores
x1_range = min(inputsPCA(:,1)):0.01:max(inputsPCA(:,1));
x2_range = min(inputsPCA(:,2)):0.01:max(inputsPCA(:,2));
[X1, X2] = meshgrid(x1_range, x2_range);
XGrid = [X1(:), X2(:)];

% Make predictions for each point on the grid
YGrid = classify(netPCA, XGrid);

% Convert categorical predictions to numeric for plotting
YGridNumeric = grp2idx(YGrid);

% Reshape the predicted classes into the shape of the grid
YGridNumeric = reshape(YGridNumeric, size(X1));

% Plot the decision boundary
figure;
hold on;
contourf(X1, X2, YGridNumeric);
scatter(inputsPCA(:,1), inputsPCA(:,2), 30, grp2idx(targets), 'filled');
title('Decision Boundaries');
xlabel('First Principal Component');
ylabel('Second Principal Component');
hold off;

function [meanAccuracy, stdAccuracy] = crossValNN(inputs, targets, layers, options, k)
    % inputs: matrix of input features
    % targets: array of target labels
    % layers: array of layers
    % options: training options
    % k: number of folds for cross-validation
    
    cvp = cvpartition(targets, 'KFold', k); % create cvpartition object
    accuracies = zeros(cvp.NumTestSets, 1); % preallocate accuracies array
    
    for i = 1:cvp.NumTestSets
        % Get the training and validation indices
        trainingIdx = cvp.training(i);
        validationIdx = cvp.test(i);
        
        % Train the network on the training data
        net = trainNetwork(inputs(trainingIdx,:), targets(trainingIdx), layers, options);
        
        % Evaluate the network on the validation data
        YPred = classify(net, inputs(validationIdx,:));
        accuracies(i) = sum(YPred == targets(validationIdx)) / numel(targets(validationIdx));
    end
    
    % Compute mean and standard deviation of accuracies
    meanAccuracy = mean(accuracies);
    stdAccuracy = std(accuracies);
end
