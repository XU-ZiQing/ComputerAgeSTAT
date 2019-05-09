function price = HestonVanilla()

rng(1);
n = 1e4;
N = 11;
multi = 5;
NN = 1 + multi*(N-1);

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
end

F = FF(:,1:multi:NN);
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
    
end

F2 = FF2(:,1:multi:NN);
clear FF2 VV2

toc

Payoff =max(0,-F(:,N)+K);
C = mean(exp(-r*T)*Payoff); % European put option price calculated under Heston Model

Payoff2 =max(0,-F2(:,N)+K);
C2 = mean(exp(-r*T)*Payoff2);

price = mean([C C2]);

end