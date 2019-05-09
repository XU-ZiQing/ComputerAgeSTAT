function [estimateclasstotal,model]=Adaboost2(mode,datafeatures,dataclass_or_model,itt)

switch(mode)
    case 'train'
        
        % Train the adaboost model 
        % Set the data class 
        dataclass=dataclass_or_model(:);
        model=struct;  

        % Weight of training samples, first every sample is even important (same weight)
        D=ones(length(dataclass),1)/length(dataclass);
        % This variable will contain the results of the single weak classifiers weight by their alpha
        estimateclasssum=zeros(size(dataclass));

        % Do all model training itterations
        for t=1:itt
            
            % Find the best treshold to separate the data in two classes
            [direc,dimens,thres,err]= WeightedThresholdClassifier(datafeatures,dataclass,D);

            % Weak classifier influence on total result is based on the current classification error
            alpha=1/2 * log((1-err)/max(err,eps));

            % Store the model parameters
            model(t).alpha = alpha;
            model(t).dimension=dimens;
            model(t).threshold=thres;
            model(t).direction=direc;
            % We update D so that wrongly classified samples will have more weight
            estimateclass = (datafeatures(:,dimens)<=datafeatures(thres,dimens))-...
                (datafeatures(:,dimens)>datafeatures(thres,dimens));
            if direc=='Large'   
                estimateclass = -estimateclass;                        
            end            
            D = D.* exp(model(t).alpha.*(dataclass~=estimateclass));
            D = D./sum(D);

            % Calculate the current error of the cascade of weak classifiers
            estimateclasssum=estimateclasssum +estimateclass*model(t).alpha;
            estimateclasstotal=sign(estimateclasssum);
            model(t).error=sum(estimateclasstotal~=dataclass)/length(dataclass);

            if(model(t).error==0)
                break; 
            end
        end

    case 'apply' 

        % Apply Model on the test data
        model=dataclass_or_model;    

        % Add all results of the single weak classifiers weighted by their alpha 
        estimateclasssum=zeros(size(datafeatures,1),1);

        for t=1:length(model)
            estimateclasssum=estimateclasssum+model(t).alpha*ApplyClassTreshold(model(t), datafeatures);
        end

        % If the total sum of all weak classifiers is less than zero it is probablly class -1 otherwise class 1;
        estimateclasstotal=sign(estimateclasssum);
end

end

function [direc,dimens,thres,err] = WeightedThresholdClassifier(datafeatures,dataclass,dataweight)

featureNumber = size(datafeatures,2);
errorSmall = zeros(featureNumber,2);
errorLarge = zeros(featureNumber,2);

for j = 1:featureNumber
    % choose the filter to be: feature<=threshold =>1, otherwise =>-1
    estimateClassSmall = (datafeatures(:,j)<=(datafeatures(:,j)'))-...
        (datafeatures(:,j)>(datafeatures(:,j)'));
    estimateTestSmall = (dataclass~=estimateClassSmall);
    estimateErrorSmall = (dataweight')*estimateTestSmall;
    estimateErrorLarge = 1 - estimateErrorSmall;
    [errorSmall(j,1), errorSmall(j,2)] = min(estimateErrorSmall);
    [errorLarge(j,1), errorLarge(j,2)] = min(estimateErrorLarge);
end

[errSmall, featureSmall] = min(errorSmall(:,1));
[errLarge, featureLarge] = min(errorLarge(:,1));

if errSmall<=errLarge
    direc = 'Small';
    dimens = featureSmall;
    thres = errorSmall(dimens,2);    
    err = errorSmall(dimens,1);
else
    direc = 'Large';
    dimens = featureLarge;
    thres = errorLarge(dimens,2);
    err = errorLarge(dimens,1);
end

end

function y = ApplyClassTreshold(h, x)

if(h.direction == 'Small')
    y =  double(x(:,h.dimension) <= h.threshold);
else
    y =  double(x(:,h.dimension) > h.threshold);
end

y(y==0) = -1;

end
