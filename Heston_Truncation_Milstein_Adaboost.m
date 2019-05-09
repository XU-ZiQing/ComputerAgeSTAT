% Here we try to use the Milstein discretization instead of the Euler's.
clear 
warning off

rng(1);
n = 1e5;
N = 11;
multi = 5;
NN = 1 + multi*(N-1);

k1 = 2;
k2 = 2;
k = k1+k2+1;

K = 10;
F0 = 9;
T = 0.25;
r = 0.1;
kappa = 5;
sigma = 0.9;
theta = 0.16;
V0 = 0.0625;
rho = 0.1;
q = 0;
mu = r-q;

deltaT = T/(N-1);
deltaTT = deltaT/multi;
disc = exp(-r*deltaT);

%Euler Discretization invovles a discrete error, which can be reduced by
%increasing the time stamps when generating the price matrix
tic

FF = zeros(n,NN);
FF(:,1) = ones(n,1)*F0;
VV1 = zeros(n,N);
VV1(:,1) = ones(n,1)*V0;

for j = 2 : NN
    W1 = randn(n,1) * sqrt(deltaTT); 
    W2 = randn(n,1) * sqrt(deltaTT); 
    FF(:,j) = FF(:,j-1).*exp((mu-max(VV1(:,j-1),0)/2)*deltaTT+sqrt(max(VV1(:,j-1),0)).* ...
        (rho*W1+sqrt(1-rho^2)*W2)+0.25*(rho^2*W1.^2+(1-rho^2)*W2.^2-deltaTT));
    VV1(:,j) = VV1(:,j-1) + kappa*(theta-max(VV1(:,j-1),0))*deltaTT + ...
        sqrt(max(VV1(:,j-1),0))*sigma.*W1 + 0.25*sigma^2*(W1.^2-deltaTT);    
    j;
end

F = FF(:,1:multi:NN);
V1 = VV1(:,1:multi:NN);
clear FF VV1

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
%We are supposed to apply some special treatment to the variance process
%when it hits 0. Here we use the full-truncation scheme.

Payoff =max(0,-F(:,N)+K);
C = mean(exp(-r*T)*Payoff); % European put option price calculated under Heston Model

Payoff2 =max(0,-F2(:,N)+K);
C2 = mean(exp(-r*T)*Payoff2);


s1 = zeros(20,1); % high order TERM
s2 = zeros(20,1);


for k = 1:20
P = zeros(n,N);
P(:,N) = Payoff; % Terminal payoff without early excercise
Pp = P;
Pp_1 = P;

P2 = zeros(n,N);
P2(:,N) = Payoff2; 
Pp2 = P2;
Pp2_1 = P2;

p = zeros(N-1,k+2+1);

tic
for nn = N-1:-1:2 %the nn-th time interval
    y = max(0,-F(:,nn)+K); %payoff of put option, n*1 vector
    
    inTheMoney = find(y>0);
    yex = y(inTheMoney);
    X = F(inTheMoney,nn);
    Yp = Pp(inTheMoney,nn+1:N)*(disc.^(1:N-nn))';

    % basis functions for polynomials
    VX = V1(inTheMoney,nn);
    A = [X.^(0:k1) VX.^(1:k2) X.^(1:2).*VX X.*VX.^2];

    % Now let us use boosting to classify whether the holder should
    % exercise the option
    % 1 stands for early exercise and 0 means not.
    ex = (yex > Yp)-(yex <= Yp);
    
%    [exhat, Adabst] = adaboost('train',A,ex,k);
     adabst = fitensemble(A,ex,'AdaBoostM1',5*k,'Tree');
     exhat = predict(adabst,A);
    
    
    % stopping rule, using tree
    iitmp = find(exhat==1);
    indp = inTheMoney(iitmp);
    Pp(indp,:) = 0;
    Pp(indp,nn) = yex(iitmp);    
    
    
    sum(ex==exhat)/length(ex)
    sum(exhat)/length(exhat);

    
    y2 = max(0,-F2(:,nn)+K); %payoff of put option, n*1 vector
    
    itm2 = find(y2>0);
    yex2 = y2(itm2);
    X2 = F2(itm2,nn);
    Yp2 = Pp2(itm2,nn+1:N)*(disc.^(1:N-nn))';
    
    
    VX2 = V2(itm2,nn);
    A2 = [X2.^(0:k1) VX2.^(1:k2) X2.^(1:2).*VX2 X2.*VX2.^2];

    ex2 = (yex2 > Yp2)-(yex2 <= Yp2);

%    exhat2 = adaboost('apply',A2,Adabst); 
     exhat2 = predict(adabst,A2);

    %stopping rule
    iitmp2 = find(exhat2==1);
    indp2 = itm2(iitmp2);
    Pp2(indp2,:) = 0;
    Pp2(indp2,nn) = yex2(iitmp2);
   
   
    sum(ex2==exhat2)/length(ex2)


    nn
   
end
toc


price1 = Pp*disc.^(0:N-1)';
price2 = Pp2*disc.^(0:N-1)';


Burmudan1 = mean(price1);
s1(k) = Burmudan1;
Burmudan2 = mean(price2);
s2(k) = Burmudan2;

std1 = std(price1)/sqrt(n-1);
std2 = std(price2)/sqrt(n-1);

price = mean([price1; price2]);
stde = std([price1; price2])/sqrt(2*n);

end

logitPrice = HestonLogit();
LSMCPrice = HestonLSMC();
vanillaPrice = HestonVanilla();

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
