% Load the iris dataset
load fisheriris
inputs = zscore(meas); % standardize input features
targets = categorical(species); % target labels

TEST_SIZE = 45;

[y1, idx] = datasample(inputs, TEST_SIZE);

y2 = targets(idx);

inputs = removerows(inputs, "ind", idx);
targets = removerows(targets, "ind", idx);

% split into training and validation sets
cvp = cvpartition(targets,'KFold',3,'Stratify',true);

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
[net, info] = trainNetwork(inputsTrain, targetsTrain, layers, options);


for i = 1:cvp.NumTestSets
    idxTrain = training(cvp, i);
    idxValidation = test(cvp, i);

    inputsTrain = inputs(idxTrain,:);
    targetsTrain = targets(idxTrain);

    inputsValidation = inputs(idxValidation,:);
    targetsValidation = targets(idxValidation);
    
    options.ValidationData = {inputsValidation, targetsValidation};
    
    % Train the network
    [net, info] = trainNetwork(inputsTrain, targetsTrain, layers, options);

% Generate predictions on the validation data
YPred = classify(net, inputsValidation);

% Obtain the predicted scores (probabilities) from the trained network
scores = predict(net, inputsValidation);


% Convert the categorical target labels to a binary matrix
binaryTargetsValidation = full(ind2vec(double(targetsValidation)'))';

% For each class
for i = 1:3
    % Compute the ROC curve
    [X,Y,T,AUC] = perfcurve(binaryTargetsValidation(:,i), scores(:,i), 1);

    % Plot the ROC curve
    plot(X, Y)
    hold on
end
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC curve')
legend('Class 1','Class 2','Class 3','Location','Best')
hold off

% Apply PCA
[coeff, score] = pca(inputsTrain);

% Transform inputsValidation to PCA space
inputsValidationPCA = inputsValidation * coeff;

% Generate a grid of points within the range of the PCA scores
x1_range = min(score(:,1)):0.01:max(score(:,1));
x2_range = min(score(:,2)):0.01:max(score(:,2));
[X1, X2] = meshgrid(x1_range, x2_range);
XGridPCA = [X1(:), X2(:)];

% Make predictions for each point on the grid in the PCA space
XGrid = XGridPCA * coeff(1:2,:); % transform to original space before prediction
YGrid = classify(net, XGrid);

% Convert categorical predictions to numeric for plotting
YGridNumeric = grp2idx(YGrid);

% Reshape the predicted classes into the shape of the grid
YGridNumeric = reshape(YGridNumeric, size(X1));
% PCA-transform the validation data
scoreValidation = inputsValidation * coeff;

% Plot the decision boundary
figure;
hold on;
contourf(X1, X2, YGridNumeric);
scatter(scoreValidation(:,1), scoreValidation(:,2), 30, grp2idx(targetsValidation), 'filled');
title('Decision Boundaries');
xlabel('First Principal Component');
ylabel('Second Principal Component');
hold off;

% Create a confusion matrix
cm = confusionchart(targetsValidation, YPred);
title('Confusion Matrix for Validation Data');
end