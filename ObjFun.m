function [Profit] = ObjFun(X)
% The function takes as input the ICV configuration and computes the
% objective function value - which is here profit

Time = 2000; % Time over which the optimization is considered [days]
Prod = ANNFun([Time; X'*1e-4]); % Output values from ANN
Oil_Prod = Prod(1); % Cumulative oil production for ICV setting [stb]
Water_Prod = Prod(2); % Cumulative water production for ICV setting [stb]
Oil_Price = 50; % Oil marginal price after subtracting OPEX (without water)
Water_Cost = 20; % Average wastewater management and OPEX costs

% Objective function is: Profit = Revenue(from oil) - Cost(from water)
% The minus sign converts the problem from a maximization to a minimization
Profit= -(Oil_Price*Oil_Prod - Water_Cost*Water_Prod); 
end

