% We want to evaluate the price of a Bermudan put option under Heston model
clear 
warning off

% set the random seed
rng(1);

% number of simulated paths for both training set and testing set
n = 1e5;

% number of monitoring points. 
N = 11;

% We divide each time interval (with length 1/(NN-1)) into 5 sub-intervals, to reduce discrete error in the simmulation
multi = 5;
NN = 1 + multi*(N-1);

% degree of basis functions, k1 for asset price and k2 for its volatility
k1 = 2;
k2 = 2;
k = k1+k2+1;

K = 10; % strike price
F0 = 9; % spot price
T = 0.25; % time to maturity
r = 0.1; % riskfree interest rate
kappa = 5; % reverting rate of the volatility process
sigma = 0.9; % volatility of volatility
theta = 0.16; % long-tern mean of volatility
V0 = 0.0625; % initial volatility
rho = 0.1; % correlation of the two Brownian motions
q = 0; % dividend yield
mu = r-q; % drift rate

deltaT = T/(N-1); % length of each time interval
deltaTT = deltaT/multi; % length of each sub-interval
disc = exp(-r*deltaT); % discount factor

tic

FF = zeros(n,NN); 
FF(:,1) = ones(n,1)*F0; % asset price matrix
VV1 = zeros(n,N);
VV1(:,1) = ones(n,1)*V0; % volatility matrix

% simualte the training set using Milstein discretization. 
% To make sure the volatility is non-negative, we use the full-truncation scheme.
for j = 2 : NN
    W1 = randn(n,1) * sqrt(deltaTT); 
    W2 = randn(n,1) * sqrt(deltaTT); 
    FF(:,j) = FF(:,j-1).*exp((mu-max(VV1(:,j-1),0)/2)*deltaTT+sqrt(max(VV1(:,j-1),0)).* ...
        (rho*W1+sqrt(1-rho^2)*W2)+0.25*(rho^2*W1.^2+(1-rho^2)*W2.^2-deltaTT));
    VV1(:,j) = VV1(:,j-1) + kappa*(theta-max(VV1(:,j-1),0))*deltaTT + ...
        sqrt(max(VV1(:,j-1),0))*sigma.*W1 + 0.25*sigma^2*(W1.^2-deltaTT);    
    j;
end

% I can not recall why I first use FF than replace it F.
F = FF(:,1:multi:NN);
V1 = VV1(:,1:multi:NN);

clear FF VV1

% do the same thing for the tesing set
FF2 = zeros(n,NN);
FF2(:,1) = ones(n,1)*F0;
VV2 = zeros(n,N);
VV2(:,1) = ones(n,1)*V0;

for j = 2 : NN    
    W1 = randn(n,1) * sqrt(deltaTT); 
    W2 = randn(n,1) * sqrt(deltaTT); 
    FF2(:,j) = FF2(:,j-1).*exp((mu-max(VV2(:,j-1),0)/2)*deltaTT+sqrt(max(VV2(:,j-1),0)).* ...
        (rho*W1+sqrt(1-rho^2)*W2)+0.25*(rho^2*W1.^2+(1-rho^2)*W2.^2-deltaTT));
    VV2(:,j) = VV2(:,j-1) + kappa*(theta-max(VV2(:,j-1),0))*deltaTT + ...
        sqrt(max(VV2(:,j-1),0))*sigma.*W1 + 0.25*sigma^2*(W1.^2-deltaTT);
    
    j ;   
end

F2 = FF2(:,1:multi:NN);
V2 = VV2(:,1:multi:NN);
clear FF2 VV2

toc

% calculate the terminal payoff and its mean, i.e., vanilla option price
Payoff =max(0,-F(:,N)+K);
C = mean(exp(-r*T)*Payoff); 

Payoff2 =max(0,-F2(:,N)+K);
C2 = mean(exp(-r*T)*Payoff2);

%  estimated Bermudan option price
s1 = zeros(20,1); 
s2 = zeros(20,1);

%  We try our algorithm in 20 different iterations, to see the performance Vs. iterations
for k = 1:20
P = zeros(n,N);
P(:,N) = Payoff; 
Pp = P;
Pp_1 = P; %  payoff matrix for the training set

P2 = zeros(n,N);
P2(:,N) = Payoff2; 
Pp2 = P2;
Pp2_1 = P2; % payoff matrix for the testing set

p = zeros(N-1,k+2+1);

tic
for nn = N-1:-1:2 %the nn-th time interval
    y = max(0,-F(:,nn)+K); % payoff of put option, n*1 vector
    
    inTheMoney = find(y>0); % find the in-the-money paths
    yex = y(inTheMoney); % the payoff vector of in-the-money paths
    X = F(inTheMoney,nn); % the asset price vector of in-the-money paths
    Yp = Pp(inTheMoney,nn+1:N)*(disc.^(1:N-nn))'; % continuation value vector of in-the-money paths

    % basis functions for polynomials
    VX = V1(inTheMoney,nn); % the volatility vector of in-the-money paths
    A = [X.^(0:k1) VX.^(1:k2) X.^(1:2).*VX X.*VX.^2]; % basis function matrix

    % Now let us use boosting to classify whether the holder should
    % exercise the option
    % 1 stands for early exercise and -1 means not.
    ex = (yex > Yp)-(yex <= Yp);
    
    % train the boosting algorithm
    [exhat, Adabst] = adaboost('train',A,ex,k);    
    
    % stopping rule
    iitmp = find(exhat==1); % the index of index of predicted in-the-money paths
    indp = inTheMoney(iitmp);  % the index of predicted in-the-money paths
    Pp(indp,:) = 0;
    Pp(indp,nn) = yex(iitmp); % overwrite the payoff matrix
   
    y2 = max(0,-F2(:,nn)+K); %p ayoff of put option, n*1 vector
    
    % do the same for the testing set
    itm2 = find(y2>0);
    yex2 = y2(itm2);
    X2 = F2(itm2,nn);
    Yp2 = Pp2(itm2,nn+1:N)*(disc.^(1:N-nn))';    
    
    VX2 = V2(itm2,nn);
    A2 = [X2.^(0:k1) VX2.^(1:k2) X2.^(1:2).*VX2 X2.*VX2.^2];

    ex2 = (yex2 > Yp2)-(yex2 <= Yp2);
    
    % apply the model we've trained to the testing set
    exhat2 = adaboost('apply',A2,Adabst); 

    % stopping rule
    iitmp2 = find(exhat2==1);
    indp2 = itm2(iitmp2);
    Pp2(indp2,:) = 0;
    Pp2(indp2,nn) = yex2(iitmp2);
   
end
toc

% From now on, 1 stands for training set and 2 stands for testing set

% the option price vector computed from the payoff matrix
price1 = Pp*disc.^(0:N-1)';
price2 = Pp2*disc.^(0:N-1)';

% Bermudan option price. A small difference between the values from training set and testing set demonstrates no over-fitting
Burmudan1 = mean(price1);
s1(k) = Burmudan1;
Burmudan2 = mean(price2);
s2(k) = Burmudan2;

% standard deviation
std1 = std(price1)/sqrt(n);
std2 = std(price2)/sqrt(n);

% the final result
price = mean([price1; price2]);
stde = std([price1; price2])/sqrt(2*n);

end

% three benchmarks

% the option price via logistic regression
logitPrice = HestonLogit();

% the option price from least square Monte Carlo
LSMCPrice = HestonLSMC();

% vanilla option price
vanillaPrice = HestonVanilla();

% plot the figure
plot(1:20,s1,'b--');hold on
plot(1:20,s2,'g--');hold on
plot(1:20,(s1+s2)/2,'m--');
plot(1:20,LSMCPrice*ones(1,20),'k-*');
plot(1:20,logitPrice*ones(1,20),'r-d');
plot(1:20,vanillaPrice*ones(1,20),'y-^');
legend('Training price','Testing price','Boosting price','LSMC price','Logit price','vanilla price');
xlabel('Number of boosting iterations');
ylabel('Option price');

% end
