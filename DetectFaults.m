%% Fault Detection Project
% This script loads AHU and VAV data from the given directory,
% pre-processes the dataset for Nan's and uses Supervised and Unsupervised
% learning models to predicts faults.

%% Load Data
% Data Directory
FolderName = 'C:\Users\kunin\Google Drive\Desktop\School\A.I. in Energy Systems\Project FDD\Data\';
Data = LoadData(FolderName);
DataTable_1 = Data.MZVAV_1;
%%
CatTable = table();
% Variables with categorical values
CatTable.AHU_SupplyAirFanStatus = logical(DataTable_1.AHU_SupplyAirFanStatus);
CatTable.AHU_ReturnAirFanStatus = logical(DataTable_1.AHU_ReturnAirFanStatus); 
CatTable.FaultDetectionGroundTruth = logical(DataTable_1.FaultDetectionGroundTruth); 
CatTable.OccupancyModeIndicator = logical(DataTable_1.OccupancyModeIndicator); 

DataTable_1.AHU_SupplyAirFanStatus = [];
DataTable_1.AHU_ReturnAirFanStatus = [];
DataTable_1.OccupancyModeIndicator = [];
DataTable_1.FaultDetectionGroundTruth = [];

%% Normalizing Data
numData = DataTable_1{:,2:end-1};
ResponseData = DataTable_1{:,end};
normalizedData = normalize(numData,'range',[0 1]);
%% Feature Reduction
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
% figure('WindowStyle','docked')
% z = linkage(normalizedData,'centroid','euclidean');
% dendrogram(z)
% ylabel('Distance from centroids')
% xlabel('Clusters')
% title('Hierarchical Clustering')
%% K-means clustering
[idx,C] = kmeans(normalizedData,3);

figure('WindowStyle','docked')
scatter3(score_pca(idx==1,1),score_pca(idx==1,2),score_pca(idx==1,3),'r.')
hold on
scatter3(score_pca(idx==2,1),score_pca(idx==2,2),score_pca(idx==2,3),'b.')
scatter3(score_pca(idx==3,1),score_pca(idx==3,2),score_pca(idx==3,3),'k.')
scatter3(C(:,1),C(:,2),C(:,3),'kx') 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','NW')
xlabel('Supply Air Temperature')
ylabel('Supply Air Temperature Setpoint')
zlabel('Outside Air temperature')
title 'Cluster Assignments and Centroids'
hold off
%% Dataset after Feature Selection

ClassificationData = DataTable_1(:,1:4);
ClassificationData.AHU_SupplyAirFanStatus = CatTable.AHU_SupplyAirFanStatus;
ClassificationData.AHU_ReturnAirFanStatus = CatTable.AHU_ReturnAirFanStatus; 
ClassificationData.OccupancyModeIndicator = CatTable.OccupancyModeIndicator; 
ClassificationData.FaultDetectionGroundTruth = CatTable.FaultDetectionGroundTruth; 


%%

NeuralNetData = Data.MZVAV_1;
NeuralNetData.AHU_SupplyAirFanStatus = logical(NeuralNetData.AHU_SupplyAirFanStatus);
NeuralNetData.AHU_ReturnAirFanStatus = logical(NeuralNetData.AHU_ReturnAirFanStatus);
NeuralNetData.OccupancyModeIndicator = logical(NeuralNetData.OccupancyModeIndicator);
NeuralNetData.FaultDetectionGroundTruth = logical(NeuralNetData.FaultDetectionGroundTruth); 

% Amrita's Dataset 
Num_dataTrain = NeuralNetData{:,2:end-1};
Num_dataTrain_Response = NeuralNetData{:,end};

%% Split Data into Training, Validation and Testing
% Cross varidation (train: 60%, test: 40%)
cv = cvpartition(size(ClassificationData,1),'HoldOut',0.4);
idx = cv.test;

% Separate to training and test data
% Mohit and Kunind's Dataset
dataTrain = ClassificationData(~idx,:);
dataTest  = ClassificationData(idx,:);

% Fit Decision tree models

[mdl_FineTree, AccuracyTrain_FineTree] = trainFineTreeClassifier(dataTrain);
[mdl_MediumTree, AccuracyTrain_MediumTree] = trainMediumTreeClassifier(dataTrain);
[mdl_CoarseTree, AccuracyTrain_CoarseTree] = trainCoarseTreeClassifier(dataTrain);

% Predictions
GroundTruth_Test = dataTest{:,end};
dataTest.FaultDetectionGroundTruth = [];
Predictions_FineTree = mdl_FineTree.predictFcn(dataTest);
Predictions_MediumTree = mdl_MediumTree.predictFcn(dataTest);
Predictions_CoarseTree = mdl_CoarseTree.predictFcn(dataTest);
%
Correct_Predictions_FineTree = nnz(GroundTruth_Test == Predictions_FineTree);
Accuracy_FineTree = Correct_Predictions_FineTree/length(GroundTruth_Test)*100;

Correct_Predictions_MediumTree = nnz(GroundTruth_Test == Predictions_MediumTree);
Accuracy_MediumTree = Correct_Predictions_MediumTree/length(GroundTruth_Test)*100;

Correct_Predictions_CoarseTree = nnz(GroundTruth_Test == Predictions_CoarseTree);
Accuracy_CoarseTree = Correct_Predictions_CoarseTree/length(GroundTruth_Test)*100;


%% Plots Decision Tree
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
% ######## Uncomment if the file exits
% load('Quadratic SVM model.mat')
Predictions_SVM = QuadraticSVM.predictFcn(dataTest);
ActualResponse_dataTest = dataTest.FaultDetectionGroundTruth;
Correct_Predictions_SVM = nnz(ActualResponse_dataTest == Predictions_SVM);
Accuracy_SVM = Correct_Predictions_SVM/length(ActualResponse_dataTest)*100;

%% Plots Confusion chart for SVM
figure('WindowStyle','docked')
cm = confusionchart(ActualResponse_dataTest,Predictions_SVM);
cm.Normalization = 'column-normalized';
title('Test data SVM (Quadratic): Confusion Chart')