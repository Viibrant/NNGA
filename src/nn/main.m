%% loading data
load fisheriris
inputs = zscore(meas); % standardize input features
targets = categorical(species); % target labels

TEST_SIZE = 45;
VAL_SIZE = 30;
numFolds = 3;

[x_test, idx] = datasample(inputs, TEST_SIZE);
y_test = targets(idx);
inputs = removerows(inputs, "ind", idx);
targets = removerows(targets, "ind", idx);


[x_val, idx] = datasample(inputs, VAL_SIZE);
y_val = targets(idx);
inputs = removerows(inputs, "ind", idx);
targets = removerows(targets, "ind", idx);


% split into training and validation sets
cvp = cvpartition(targets,'KFold',numFolds,'Stratify',true);

% Initialize arrays to store results
bestPerformance = Inf;
bestModel = [];
bestOptions = [];
bestInfo = [];

worstPerformance = -Inf;
worstModel = [];
worstOptions = [];
worstInfo = [];
worstHyper = [];

% Define grid search parameters
learningRates = [0.1, 0.25];
dropoutRates = [0.2, 0.5];
% activationFns = {@reluLayer, @tanhLayer};


for lr = learningRates
    for dr = dropoutRates
        options = trainingOptions('sgdm', ...
            'MaxEpochs', 250, ...ana
            'InitialLearnRate',lr, ...
            'Shuffle','every-epoch', ...
            'ValidationData',{x_val, y_val}, ...
            'ValidationFrequency',30, ...
            'Verbose',false, ...
            'Plots','training-progress' ...
         );

        layers = [ ...
            featureInputLayer(4)

            fullyConnectedLayer(8)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
              fullyConnectedLayer(16)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
            fullyConnectedLayer(32)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
            fullyConnectedLayer(32)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
            fullyConnectedLayer(64)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
            fullyConnectedLayer(64)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
            fullyConnectedLayer(32)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
            fullyConnectedLayer(32)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
            fullyConnectedLayer(16)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
              fullyConnectedLayer(8)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
            fullyConnectedLayer(4)
            reluLayer
            batchNormalizationLayer
            dropoutLayer(dr)
            
            fullyConnectedLayer(3)
            softmaxLayer
            classificationLayer
           ];
        for i = 1:cvp.NumTestSets
            idxTrain = training(cvp, i);
            idxValidation = test(cvp, i);

            inputsTrain = inputs(idxTrain,:);
            targetsTrain = targets(idxTrain);

            inputsValidation = inputs(idxValidation,:);
            targetsValidation = targets(idxValidation);
            
            options.ValidationData = {inputsValidation, targetsValidation};
            
            % Train the network
            [model, info] = trainNetwork(inputsTrain, targetsTrain, layers, options);

            if info.FinalValidationLoss < bestPerformance
                bestPerformance = info.FinalValidationLoss;
                bestModel = model;
                bestOptions = options;
                bestInfo = info;
                bestHyper = {dr, lr};
            end
           
            if info.FinalValidationLoss > worstPerformance
                worstPerformance = info.FinalValidationLoss;
                worstModel = model;
                worstOptions = options;
                worstInfo = info;
                worstHyper = {dr, lr};
            end
        end
    end
end

% Display the best performance
disp(['Best performance: ', num2str(bestPerformance), "Hyperparameters : ", bestHyper]);
disp(['Worst performance: ', num2str(worstPerformance), "Hyperparameters : ", worstHyper]);

model = bestModel;
info = bestInfo;

%% predict labels
% score store likelihood of each sample
% being of each class: nSample by nClass
% Test classifier
[YPred, scores] = classify(model, x_test);

% Compute confusion matrix
cm = confusionmat(y_test, YPred);
plotconfusion(y_test, YPred)

% Compute and display accuracy
accuracy = sum(YPred == y_test) / numel(y_test);
disp(['Accuracy: ', num2str(accuracy)])

plot(rocmetrics(YPred, scores, categories(y_test)))



% Apply PCA
[coeff, score] = pca(x_test);

% Transform inputsValidation to PCA space
inputsValidationPCA = x_test * coeff;

% Generate a grid of points within the range of the PCA scores
x1_range = min(score(:,1)):0.01:max(score(:,1));
x2_range = min(score(:,2)):0.01:max(score(:,2));
[X1, X2] = meshgrid(x1_range, x2_range);
XGridPCA = [X1(:), X2(:)];

% Make predictions for each point on the grid in the PCA space
XGrid = XGridPCA * coeff(1:2,:); % transform to original space before prediction
YGrid = classify(model, XGrid);

% Convert categorical predictions to numeric for plotting
YGridNumeric = grp2idx(YGrid);

% Reshape the predicted classes into the shape of the grid
YGridNumeric = reshape(YGridNumeric, size(X1));
% PCA-transform the validation data
scoreValidation = x_test * coeff;

% Convert categorical predictions to numeric for plotting
YValidationNumeric = grp2idx(y_test);

% Plot the decision boundary
figure;
hold on;
contourf(X1, X2, YGridNumeric);
scatter(scoreValidation(:,1), scoreValidation(:,2), 35, 'k', 'filled'); % black color
scatter(scoreValidation(:,1), scoreValidation(:,2), 30, YValidationNumeric, 'filled');

title('Decision Boundaries');
xlabel('First Principal Component');
ylabel('Second Principal Component');
hold off;

