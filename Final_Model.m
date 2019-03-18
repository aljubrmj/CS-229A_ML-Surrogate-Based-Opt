%% Final Model for ERE 291-Project
% Anthony Boukarim
% Mohammad (Jabs) Aljubran

% Clearing Workspace
clc; clear all; close all;

%% Solving using fixed parameters

% Definining the number of variables
nvars = 3; 

% Constraints for genetic algorithm
A = []; b = []; % No additional inequality constraints are considered
Aeq = []; beq = []; % No equality constraints are considered
lb = zeros(1,nvars); % Lower bounds on decision variables for ICV config.   
ub = 10*ones(1,nvars); % Upper bounds on decision variables for ICV config.
nonlcon = []; % No non-linear constraints are considered

% Indicating the required output from the genetic algorithm
options = optimoptions('ga','PlotFcn',@gaplotbestf);


% Forcing the decision variables to be integer
IntCon = [1,2,3];

% Solving the model and storing the optimal solution
[x,fval,exitflag] = ga(@ObjFun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,IntCon,options)
%axis([0 80 -9.14e7 -9e7]) %display only the relevant part of the graph

display(['The optimal profit made after 2000 days is = ', num2str(-fval),' USD'])
display(['The optimal ICV configuration is: ', num2str(x)])

% Comments:
% Please refer to the objective function, statistical model, and sensitvity
% analysis in ObjFun.m, ANNFun.m, and SensFun.m respectively.