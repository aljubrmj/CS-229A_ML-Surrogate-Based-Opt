close all
load ('CS229A_Dataset.mat')
%% Partition Data for Training, Validation, and Testing using Latin Hypercube Sampling

total = 1:1330*502; %Set of all of the indices
z = 1;
mse_trainRBF = zeros(100);
mse_valRBF = zeros(100);
mse_testRBF = zeros(100);
R2RBF = zeros(100); 
ValidationPercentage = 0.2;

PercentTrainVal = 0.1;
PercentTest = 1 - PercentTrainVal;
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
%setdiff separates out the testing indices from training indices

%% Train SVM with RBF Gaussian Kernel

%Create SVM model with Verbose to show progress in the command window
%Create hyperparameters to loop over
KernelScalevec = [0.1 1 10 100 1000 10000]; 
BoxConstraintvec = [1e1 1e2 1e3 1e4 1e5 1e6 1e7 1e8 1e9];

for i = 1:length(BoxConstraintvec)
    i
    parfor j = 1:length(KernelScalevec)

        Mdl = fitrsvm(X(tr,:), y(tr,[1]), ...
            'Verbose',0 , 'Standardize', 1, ...
            'KernelFunction', 'gaussian', ...
            'BoxConstraint', BoxConstraintvec(i), 'KernelScale', KernelScalevec(j));   
        MdlRBF_matrix{i,j} = Mdl; 
        
        mse_trainRBF(i,j) = resubLoss(Mdl); %Training error
        
        p_y = predict(Mdl, X(val, :)); %Predict on validation set
        mse_valRBF(i,j) = immse(y(val,[1]), p_y); %Validation error
        
        p_y = predict(Mdl, X(ts, :)); %Predict on test set
        mse_testRBF(i,j) = immse(y(ts,[1]), p_y); %Testing error
        R2RBF(i,j) = regress(y(ts,[1]), p_y); %R2 value
        
    end
end 

% %Plot Training Data: 
% figure (1); h1 = heatmap(mse_trainRBF, 'Colormap', summer);
% h1.XData = KernelScalevec; h1.YData = BoxConstraintvec;
% h1.XLabel = 'gamma'; h1.YLabel = 'C'; h1.Title = 'Training Data MSE analysis'; 
% 
% %Plot Validation Data: 
% figure (2); h2 = heatmap(mse_valRBF, 'Colormap', summer);
% h2.XData = KernelScalevec; h2.YData = BoxConstraintvec;
% h2.XLabel = 'gamma'; h2.YLabel = 'C'; h2.Title = 'Validation Data MSE analysis'; 
% 
% 
