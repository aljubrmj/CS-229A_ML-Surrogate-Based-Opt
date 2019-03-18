function [trainvalindex] = Partition(i, ICVrecord)

%Generate trainingvalidation partition using lhs
[sample,~]=lhs(i,[0 0 0],[10  10 10]);

%Switch elements to ICV constriction area sizes
function mapout = my_changem(mapout, newcode, oldcode)
   assert(numel(newcode) == numel(oldcode), 'newcode and oldecode must have the same number of elements');
   [toreplace, bywhat] = ismember(mapout, oldcode);
   mapout(toreplace) = newcode(bywhat(toreplace));
end

sample = my_changem(sample, 0:0.0001:0.001, 0:10);

%Find equivalent indices from the sumulation dataset
[~,trainvalindex]= ismember(sample,ICVrecord,'rows');
%Sort values from lowest to highest
trainvalindex = unique(trainvalindex);
trainvalindex = trainvalindex(trainvalindex ~=0);
testindex = setdiff([1:1330], trainvalindex);

%figure
% scatter3(X_scaled(:,1),X_scaled(:,2), X_scaled(:,3),'*')
% title('Random Variables')
% xlabel('X1')
% ylabel('X2')
% grid on
end 