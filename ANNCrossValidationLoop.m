%% This ANN trains using BRB along with LMA validation. Note

%Note that you have to provide the file with the desired number of
%10-neuron layers (or even modify the number of neurons accordingly) and
%the indices of the training, validation, and testing samples besides
%passing the actual X and y dataset

function [net, tr] = ANNCrossValidationLoop(X,y, tr, val, ts, layers) % %Tr is training indices %Ts is testing indices
%ANN takes in X and y matrices and runs the training based on some neural
%network structure. Takes indices for training and testing set, each index
%begins the a simulation run within X. Returns the %network architecture 
%and information on training/testing results. 

%Create Network structure
hiddenlayersize = 10*ones(1,layers);
%net = fitnet(hiddenlayersize, 'trainlm');
net = fitnet(hiddenlayersize, 'trainbr'); %This is the bayesian trainer, more complex 
net.layers{1}.transferFcn='logsig'; net.layers{2}.transferFcn='logsig';
net.divideFcn = 'divideind';
net.divideParam.trainInd = tr;
net.divideParam.valInd = val;
net.divideParam.testInd = ts;

% Termination criteria 
net.trainParam.max_fail = 15;
% net.trainParam.mu_max = 1e12; 
% Modify hyperparameters
%net.trainParam.lr = 1e20; 
% net.trainParam.lr_inc = 1e9;
% net.trainParam.lr_dec = 0;
%net.trainParam.mu = 10; 
%net.trainParam.mu_dec = 0.00001; 
%net.trainParam.mu_inc = 1; 
%net.trainParam.epochs = 500; 
[net, tr] = train(net, X, y);
end

