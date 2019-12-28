%% Neural net
% This function trains and tests for various training functions and
% neurons.
function NeuralNet(InputArray, ResponseArray,Analysis)
%% Check for best fit Algorithm
% Algorithms
% 'trainlm' Levenberg-Marquardt
% 'trainbr' Bayesian Regularization
% 'trainrp' Resilient Backpropagation
if Analysis == "All"
    Algo = ["trainlm";"trainbr"; "trainrp"];
    Network_Size = [20;25;30;35;40;45;50;55;60;65;70];
    alg1_trainPerformance = zeros(1,1);
    alg1_valPerformance = zeros(1,1);
    alg1_testPerformance = zeros(1,1);
    alg2_trainPerformance = zeros(1,1);
    alg2_valPerformance = zeros(1,1);
    alg2_testPerformance = zeros(1,1);
    alg3_trainPerformance = zeros(1,1);
    alg3_valPerformance = zeros(1,1);
    alg3_testPerformance = zeros(1,1);
    
    for AlgorithmN = 1:length(Algo)
        Algorithm = Algo(AlgorithmN);
        for NeuronN = 1:length(Network_Size)
            Num_Neurons = Network_Size(NeuronN);
            [trainPerform, valPerform, testPerform] = trainANN(...
                InputArray',ResponseArray',Algorithm,Num_Neurons);
            if AlgorithmN ==1
                alg1_trainPerformance(NeuronN,1) = trainPerform;
                alg1_valPerformance(NeuronN,1) = valPerform;
                alg1_testPerformance(NeuronN,1) = testPerform;
            end
            if AlgorithmN ==2
                alg2_trainPerformance(NeuronN,1) = trainPerform;
                alg2_valPerformance(NeuronN,1) = valPerform;
                alg2_testPerformance(NeuronN,1) = testPerform;
            end
            if AlgorithmN ==3
                alg3_trainPerformance(NeuronN,1) = trainPerform;
                alg3_valPerformance(NeuronN,1) = valPerform;
                alg3_testPerformance(NeuronN,1) = testPerform;
            end
        end
    end
    % Save performance metrics
    save('Algorithm_performance.mat',...
        'alg1_testPerformance','alg1_trainPerformance','alg1_valPerformance',...
        'alg2_testPerformance','alg2_trainPerformance','alg2_valPerformance',...
        'alg3_testPerformance','alg3_trainPerformance','alg3_valPerformance')
    
    %% Plot performance
    figure()
    plot(Network_Size,alg1_testPerformance)
    hold on
    plot(Network_Size,alg2_testPerformance)
    plot(Network_Size,alg3_testPerformance)
else
    Algo = "trainlm";
    Neurons = 45;
    [~, ~, ~] = trainANN(InputArray',ResponseArray',Algo,Neurons);
end