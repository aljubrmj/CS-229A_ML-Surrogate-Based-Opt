
%% Partition data into training, validation, and testing sets

load('CS229A_Dataset.mat'); 
total = 1:1330*502; %Set of all of the indices
z = 1;
ValidationPercentage = 0.2;
TestingPercentage = 90;

PercentTrainVal = (100-TestingPercentage)/100;
PercentTest = TestingPercentage/100;
NumberTrainValidationRuns = round(PercentTrainVal*1330); %Number of Training Runs 
NumberTestRuns = 1330-NumberTrainValidationRuns; %Number of runs that will be tested 
startTrVal = Partition(NumberTrainValidationRuns, ICVrecord); %Latin Hypercube Sampling
startVal = randsample(startTrVal, round(ValidationPercentage*length(startTrVal)));
startTr = setdiff(startTrVal,startVal);
tr = [];
val = [];

for j=1:length(startTr)
  indices = (502*startTr(j)-501):(502*startTr(j)); %Creates vector of indices corresponding to WHOLE run in X
  tr = [tr indices];

end

for j=1:length(startVal)
  indices = (502*startVal(j)-501):(502*startVal(j)); %Creates vector of indices corresponding to WHOLE run in X
  val = [val indices]; 

end
ts = setdiff(total,[tr val]); %Finds the testing indices by subtracting total indices by training indices

%% Train ANN (Current setup trains 10-neuron layers starting with 1 to 4

% note that each archeticture is trained several times to account for
% problems associated with optimization with different weight
% initializations as some might give better chances of finding a more
% superior weights values compared to others
%Loop over to train with different neurons or layers
Layers = 1:4; 
repeat = 4; 
mse_train_matrix = zeros(length(Layers), repeat);
mse_val_matrix = zeros(length(Layers), repeat);
mse_test_matrix = zeros(length(Layers), repeat);
R2_matrix = zeros(length(Layers), repeat); 

for i = Layers
    i
    parfor j = 1:repeat
        

        [netOut, trOut] = ANNCrossValidationLoop(X', y(:, [1,2])', tr, val, ts, i); %Rows=features, Cols=dataPoints 
        m_matrix{i,j} = netOut; 
        tr_matrix{i,j} = trOut; 
        
        mse_train_matrix(i,j) = trOut.best_perf;
        mse_val_matrix(i,j) = trOut.best_vperf; 
        mse_test_matrix(i,j) = trOut.best_tperf;
        
    end
    
end

mse_train = min(mse_train_matrix');
mse_test = min(mse_test_matrix');
mse_val = min(mse_val_matrix'); 
R2 = min(R2_matrix');

