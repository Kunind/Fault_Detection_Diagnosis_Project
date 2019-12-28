%% Fault Detection Project
% This script loads AHU and VAV data from the given directory,
% pre-processes the dataset for Nan's and uses Supervised and Unsupervised
% learning models to predicts faults.

%% Load Data
% Data Directory
FolderName = 'C:\Users\kunin\Google Drive\Desktop\School\A.I. in Energy Systems\Project FDD\Data';

% Load file
Data = LoadData(FolderName,'MZVAV-1');

% To make sure the requested file is loaded
DataTable_1 = Data.MZVAV_1;

% Variables with logical values
DataTable_1.AHU_SupplyAirFanStatus = logical(DataTable_1.AHU_SupplyAirFanStatus);
DataTable_1.AHU_ReturnAirFanStatus = logical(DataTable_1.AHU_ReturnAirFanStatus); 
DataTable_1.OccupancyModeIndicator = logical(DataTable_1.OccupancyModeIndicator); 
DataTable_1.FaultDetectionGroundTruth = logical(DataTable_1.FaultDetectionGroundTruth); 
%% Datetime Analysis
% Check for datetime effect on model predictions
DataTable_1.Day = day(DataTable_1.Datetime);
DataTable_1.Month = month(DataTable_1.Datetime);
DataTable_1.Year = year(DataTable_1.Datetime);
DataTable_1.Hour = hour(DataTable_1.Datetime);
DataTable_1.Minute = minute(DataTable_1.Datetime);

% Data before October
DataTable_1_b4_Oct = DataTable_1(month(DataTable_1.Datetime)<10,:);

% Data before Jun
DataTable_1_b4_Jun = DataTable_1(month(DataTable_1.Datetime)<6,:);

% Fine Tree model
[Mdl_TreeDatetime_b4_oct, Accuracy_TreeDatetime_b4_oct] = TrainTreeDatetime(DataTable_1_b4_Oct);
[Mdl_TreeDatetime_b4_jun, Accuracy_TreeDatetime_b4_jun] = TrainTreeDatetime(DataTable_1_b4_Jun);

% Remove datetime from analysis if overfitting exists
if round(Accuracy_TreeDatetime_b4_oct) == 1 && round(Accuracy_TreeDatetime_b4_jun) == 1 
    warning('Datetime leads to overfitting, check data!')
    warning('Removing Datetime as a feature')
    fprintf('\nCheck figure for confirmation')
    view(Mdl_TreeDatetime_b4_oct.ClassificationTree,'mode','graph')
    DataTable_1(:,{'Day' 'Month' 'Year' 'Hour' 'Minute'}) = [];    
else
    fprintf('\nDatetime does not lead to overfitiing, not discarded!\n')
end
%% Feature Reduction
% Normalize data
numData = DataTable_1{:,2:end-1};
normalizedData = normalize(numData,'range',[0 1]);

% Principal Component Analysis
[coeff_pca,score_pca,latent_pca,tsquared_pca,Var_Exp] = ...
    pca(normalizedData);
figure('WindowStyle','docked')
pareto(Var_Exp);
ylabel('Variance Explained (%)')
xticklabels({'Supply Air Temperature', 'Supply Air Temperature Setpoint',...
    'Outdoor Temperature', 'Mixed Air Temperature'})
xlabel('Predictors')
title('Pareto Chart (PCA)')
%% Hierarchical Clustering
figure('WindowStyle','docked')
z = linkage(normalizedData(1:20000,:),'centroid','euclidean');
dendrogram(z)
ylabel('Distance from centroids')
xlabel('Clusters')
title('Hierarchical Clustering')
T = cluster(z,'maxclust',3);
% Plot clusters
figure('WindowStyle','docked')
scatter3(score_pca(T==1,1),score_pca(T==1,2),score_pca(T==1,3),'r.')
hold on
scatter3(score_pca(T==2,1),score_pca(T==2,2),score_pca(T==2,3),'b.')
scatter3(score_pca(T==3,1),score_pca(T==3,2),score_pca(T==3,3),'k.')
legend('Cluster 1','Cluster 2','Cluster 3',...
       'Location','NW')
xlabel('Supply Air Temperature')
ylabel('Supply Air Temperature Setpoint')
zlabel('Outside Air temperature')
title('Hierarchical Cluster Assignments and Centroids')
hold off

%% Neural network
NeuralNetData = DataTable_1;

% Convert Table format to numerical formatting
Num_dataTrain = NeuralNetData{:,2:end-1};
Num_dataTrain_Response = NeuralNetData{:,end};

% Run Neural Net function, Last Input = "All" for comparing error of
% different algorithms and different number of neurons
NeuralNet(Num_dataTrain, Num_dataTrain_Response,"NotAll")
%% Split Data into Training, Validation and Testing
% Cross varidation (train: 60%, test: 40%)
cv = cvpartition(size(DataTable_1,1),'HoldOut',0.4);
test_idx = cv.test;

% Separate to training and test data
dataTrain = DataTable_1(~test_idx,:);
dataTest  = DataTable_1(test_idx,:);

% Fit Decision tree models
% Fine tree
[mdl_FineTree, AccuracyTrain_FineTree] = trainFineTreeClassifier(dataTrain);
% Medium tree
[mdl_MediumTree, AccuracyTrain_MediumTree] = trainMediumTreeClassifier(dataTrain);
% Coarse tree
[mdl_CoarseTree, AccuracyTrain_CoarseTree] = trainCoarseTreeClassifier(dataTrain);

% Predictions
GroundTruth_Test = dataTest{:,end};
dataTest.FaultDetectionGroundTruth = [];
Predictions_FineTree = mdl_FineTree.predictFcn(dataTest);
Predictions_MediumTree = mdl_MediumTree.predictFcn(dataTest);
Predictions_CoarseTree = mdl_CoarseTree.predictFcn(dataTest);

% Check accuracy
Correct_Predictions_FineTree = nnz(GroundTruth_Test == Predictions_FineTree);
Accuracy_FineTree = Correct_Predictions_FineTree/length(GroundTruth_Test)*100;

Correct_Predictions_MediumTree = nnz(GroundTruth_Test == Predictions_MediumTree);
Accuracy_MediumTree = Correct_Predictions_MediumTree/length(GroundTruth_Test)*100;

Correct_Predictions_CoarseTree = nnz(GroundTruth_Test == Predictions_CoarseTree);
Accuracy_CoarseTree = Correct_Predictions_CoarseTree/length(GroundTruth_Test)*100;

disp('Decision Tree Accuracy')
fprintf('\nFine Tree: %.2f, Medium Tree: %.2f and Coarse Tree: %.2f\n',...
    Accuracy_FineTree, Accuracy_MediumTree, Accuracy_CoarseTree)

%% Plot Decision Trees
view(mdl_FineTree.ClassificationTree,'mode','graph')
view(mdl_MediumTree.ClassificationTree,'mode','graph')
view(mdl_CoarseTree.ClassificationTree,'mode','graph')

figure('WindowStyle','docked')
cm = confusionchart(GroundTruth_Test,Predictions_FineTree);
cm.Normalization = 'column-normalized';
title('Decision Tree (Fine): Confusion Chart')

figure('WindowStyle','docked')
cm = confusionchart(GroundTruth_Test,Predictions_MediumTree);
cm.Normalization = 'column-normalized';
title('Decision Tree (Medium): Confusion Chart')

figure('WindowStyle','docked')
cm = confusionchart(GroundTruth_Test,Predictions_CoarseTree);
cm.Normalization = 'column-normalized';
title('Decision Tree (Coarse): Confusion Chart')

%% SVM testing 
% Train SVM model Quadratic
[mdl_SVM, AccuracyTrain_SVM] = trainSVMmodel(dataTrain);
Predictions_SVM = mdl_SVM.predictFcn(dataTest);
ActualResponse_dataTest = dataTest.FaultDetectionGroundTruth;
Correct_Predictions_SVM = nnz(ActualResponse_dataTest == Predictions_SVM);
Accuracy_SVM = Correct_Predictions_SVM/length(ActualResponse_dataTest)*100;
disp('SVM Accuracy')
fprintf('\n Quadratic SVM: %.2f\n',Accuracy_SVM)

%% Plots Confusion chart for SVM
figure('WindowStyle','docked')
cm = confusionchart(ActualResponse_dataTest,Predictions_SVM);
cm.Normalization = 'column-normalized';
title('Test data SVM (Quadratic): Confusion Chart')