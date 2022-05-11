%% Simulation of the Feller Square-Root Process
%  The formula for the FSRP is
%  dX(t) = alpha*(mu - X)*dt + sigma*sqrt(X)*dW(t)
% As with the OUP/Vasicek Model, the FSRP goes by another name in Finance
% and that's the Cox-Ingersoll-Ross Process (CIRP). Again they applied the
% process to model interest rates.

clear all
close all

% Parameters
npaths = 20000 ; % Number of simulations 
T = 1 ; % Our time horizon
nsteps = 200 ; % The number of timesteps
dt = T/nsteps ; % The size of the timesteps
t = 0:dt:T ; % The discretization of the grid
alpha = 5 ;
mu = 0.07 ;
sigma = 0.265 ;
X0 = 0.03 ; % Our initial value

% We introduce a variable for monitoring purposes. If our feller ratio,
% defined below, is > 1, then X will never reach 0. 
feller_ratio = (2*alpha*mu)/(sigma^2) 

%% 1A Monte Carlo Simulation - Paths x Timesteps

% Paths as ROWS!
% Timesteps as COLUMNS!
%         t0   t1    t2   ...
% path 1: (0, 0.1, 0.4, ...)
% path 2: (0, -0.3, 0.1, ...)

% Since we now have our variable X with the equation (as part of the dt &
% dW terms) we can no longer simply compute an entre matrix for dX. We must
% use an iterative approach.

% Set up an [npaths,nsteps] matrix, with the first column all equal to X0
% (i.e. all paths start at X0) and the rest zeros (these will be filled out
% later).

X = [ X0*ones([npaths,1]) zeros([npaths,nsteps]) ] ;

% Define an [npaths,nsteps] matrix of normally distributed random numbers
N = randn([npaths,nsteps]) ;

% ----------------------------------------------
% 1. Euler-Maruyama Method
% ----------------------------------------------

for i = [1:nsteps]
    X(:,i+1) = X(:,i) + alpha*(mu-X(:,i))*dt + sigma*sqrt(X(:,i)*dt).*N(:,i) ;
    % To avoid any issues with our paths venturing below zero and giving us
    % complex numbers, we include this extra condition. 
    X(:,i+1) = max( X(:,i+1) , zeros([npaths,1]) );
end

% ----------------------------------------------
% 2. Euler-Maruyama Method with Analytic Moments
% ----------------------------------------------
% To use the analytic moments method we need analytic expressions for the
% expectation E(X) and varaince Var(X)
% For the OUP we have these expressions (see Ballotta & Fusai p.94)

% E(X) = X0*exp(-alpha*t) + mu*( 1-exp(-alpha*t) )
% Var(X) = X0*(sigma^2/alpha)*(exp(-1*alpha*t) - exp(-2*alpha*t) ) + ...
% ... mu*(sigma^2/2*alpha)*(1-exp(-1*alpha*t))^2

% We then ignore the form of our model and compute:
% dX = E(X) + sqrt(Var(X))*randn()
% Substituting our dt for t, and X0 with the X from the previous timestep

% Since Var(X) is long and cumbersome, we will break it up into two pieces
% This will allow us to write the Var(X) = aX+b

a = (sigma^2/alpha)*(exp(-1*alpha*dt) - exp(-2*alpha*dt) ) ;
b = mu*(sigma^2/(2*alpha))*(1-exp(-1*alpha*dt))^2 ;

for i = [1:nsteps]
    X(:,i+1) = X(:,i)*exp(-alpha*dt) + mu*(1-exp(-alpha*dt)) + sqrt(a*X(:,i)+b).*N(:,i) ;
    X(:,i+1) = max( X(:,i+1) , zeros([npaths,1]) ) ;
end

% ----------------------------------------------
% 3. Exact Method
% ----------------------------------------------
% It turns out we can actually compute the simulation exactly by using the
% non-central Chi-sq. distribution. We need to calculate some further
% parameters first though.
% Recall our N.C. Chi-sq. Distribution needs:
% d : degrees of freedom
% lambda : non-centrality parameter (which will change for each loop) see
% Ballotta & Fusai p.111
% We also need k (a multiplying factor)
% 
% d = 4*alpha*mu/sigma^2 ; 
% k = sigma^2*(1-exp(-alpha*dt))/(4*alpha) ;
% 
% for i = [1:nsteps]
%     lambda = 4*alpha*X(:,i)/(sigma^2*(exp(alpha*dt)-1)) ;
%     X(:,i+1) = ncx2rnd(d,lambda,npaths,1)*k ;
% end

%% 2A Expected, mean and sample paths

close all
figure(1)
EX = X0*exp(-alpha*t) + mu*( 1-exp(-alpha*t) ) ; 
plot(t,EX,'r',t,mean(X,1),':k',t,mu*ones(size(t)),'k--',t,X(1:1000:end,:))
legend('Expected path','Mean path','\mu')
xlabel('t')
ylabel('X')
% We compute the sqrt(VarX) as t -> inf
sdevinfty = sigma*sqrt(mu/(2*alpha)) ;
ylim([-0.02 mu+4*sdevinfty])
title('Paths of a Feller square-root process dX = \alpha(\mu-X)dt + \sigmaX^{1/2}dW')

%% 3A Variance = Mean Square Displacement

% Theoretical Var(X)
VARX = X0*(sigma^2/alpha)*(exp(-alpha*t) - exp(-2*alpha*t)) ...
    + mu*(sigma^2/(2*alpha))*(1 - exp(-1*alpha*t)).^2 ;

% Asymptote for Var(X) as t -> 0+
% We compute varzero, by taking the analytic expression for Var(X),
% differentiating and setting t=0 (i.e. we are finding the gradient at t=0)
varzero = X0*sigma^2*t ;

% Asymptote for Var(X) as t -> inf
varinf = mu*(sigma^2/(2*alpha))*ones(size(t)) ; 


figure(2)
plot(t,VARX,'r',t,varzero,'g',t,varinf,'b',t,var(X,'',1),'m',t,mean((X-EX).^2,1),'c--') ;
legend('Theory','X_0\sigma^2t','\mu\sigma^2/(2\alpha)','Sampled 1','Sampled 2','Location','SouthEast')
xlabel('t')
ylabel('Var(X) = E((X-E(X))^2)')
ylim([0 0.0008])
title('Feller Square-Root process: variance')

%% 4A Autocovariance

C = zeros([npaths,2*nsteps+1]) ;
for i = 1:npaths 
    C(i,:) = xcorr(X(i,:)-EX,'unbiased') ;
end
C = mean(C,1) ;

% It can be shown that as we take t -> inf, the covariance simply becomes a
% function of Tau, the lag. 
% Theoretical value of C_X(t,s) with t<s as we take t -> inf
COVX = exp(-alpha*t)*(sigma^2*mu)/(2*alpha) ;

figure(3)
plot(t,COVX,'r',t,C(nsteps+1:end),'b',t,varinf,'g.',t,mean(var(X,0,1))*ones(size(t)),'c.')
xlabel('\tau')
ylabel('C(\tau)')
legend('Theory','Sampled','Var for infinite t','Average sampled Var','Location','East')
title('Ornstein-Uhlenbeck process: autocovariance')

%% 5A Autocorrelation

% Taking COVX/Var(X) in the limit t -> inf, it can be shown that the
% autocorrelation becomes
% Corr(s,t) = exp(-1*alpha*tau)     with t < s

% Theoretical autocorrelation
CORRX = exp(-1*alpha*t) ;

figure(4)
plot(t,CORRX,'r',t,C(nsteps+1:end)/C(nsteps+1),'b')
xlabel('\tau')
ylabel('c(\tau)')
legend('Theory','Sampled')
title('Feller Square-Root process: autocorrelation')

