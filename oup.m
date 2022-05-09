%% Simulation of the Ornstein-Uhlenbeck Process
%  The formula for OUP is
%  dX(t) = alpha*(mu - X)*dt + sigma*dW(t)
% This could also be described as the Vasicek model, however, it is worth noting 
% they are the same thing. Vasicek just applied OUP to finance,
% specifically interest rates.

clear all
close all

% Parameters
npaths = 20000 ; % Number of paths to be simulated
T = 1 ; % Time horizon 
nsteps = 200 ; % Number of steps in [0,T]
dt = T/nsteps ; % Time grid
t = 0:dt:T ; % Discretization of our time grid
alpha = 5 ; % Speed at which it is mean reverting
mu = 0.07 ; % Long run mean
sigma = 0.07 ; % Vol/diffusion term
X0 = 0.03 ; % Initial value (e.g. current interest rate)

%% 1A Monte Carlo Simulation - Paths x Timesteps

% Paths as ROWS!
% Timesteps as COLUMNS!
%         t0   t1    t2   ...
% path 1: (0, 0.1, 0.4, ...)
% path 2: (0, -0.3, 0.1, ...)

% Since we now have our variable X with the equation (as part of the dt
% term) we can no longer simply compute an entre matrix for dX. We must use
% an iterative approach.

% Set up an [npaths,nsteps] matrix, with the first column all equal to X0
% (i.e. all paths start at X0) and the rest zeros (these will be filled out
% later).

X = [ X0*ones([npaths,1]) zeros([npaths,nsteps]) ] ;

% Define an [npaths,nsteps] matrix of normally distributed random numbers
N = randn([npaths,nsteps]) ;

% ----------------------------------------------
% 1. Euler-Maruyama Method
% ----------------------------------------------

% for i = [1:nsteps]
%     X(:,i+1) = X(:,i) + alpha*(mu-X(:,i))*dt + sigma*sqrt(dt)*N(:,i) ;
% end

% ----------------------------------------------
% 2. Euler-Maruyama Method with Analytic Moments
% ----------------------------------------------
% To use the analytic moments method we need analytic expressions for the
% expectation E(X) and varaince Var(X)
% For the OUP we have these expressions (see Ballotta & Fusai p.94)

% E(X) = X0*exp(-alpha*t) + mu*( 1-exp(-alpha*t) )
% Var(X) = (sigma^2/2*alpha) * ( 1-exp(-2*alpha*t) )

% We then ignore the form of our model and compute:
% dX = E(X) + sqrt(Var(X))*randn()
% Substituting our dt for t, and X0 with the X from the previous timestep

for i = [1:nsteps]
    X(:,i+1) = X(:,i)*exp(-1*alpha*dt) + mu*(1-exp(-1*alpha*dt)) + ...
                sigma*sqrt((1-exp(-2*alpha*dt))/(2*alpha))*N(:,i) ;
end

%% 1B Monte Carlo Simulation - Timesteps x Paths

% Timesteps as ROWS!
% Paths as COLUMNS!
%   path1    path2
% t0 ( 0  ,  0  ... )
% t1 (0.1 , -0.3 .. )
% t2 (0.4 , 0.1 ... )

X = [ X0*ones([1,npaths]) ; zeros([nsteps,npaths]) ] ;

% Define an [nsteps,npaths] matrix of normally distributed random numbers
N = randn([nsteps,npaths]) ;

% ----------------------------------------------
% 1. Euler-Maruyama Method
% ----------------------------------------------

% for i = [1:nsteps]
%     X(i+1,:) = X(i,:) + alpha*(mu-X(i,:))*dt + sigma*sqrt(dt)*N(i,:) ;
% end

% ----------------------------------------------
% 2. Euler-Maruyama Method with Analytic Moments
% ----------------------------------------------

% E(X) = X0*exp(-alpha*t) + mu*( 1-exp(-alpha*t) )
% Var(X) = (sigma^2/2*alpha) * ( 1-exp(-2*alpha*t) )

for i = [1:nsteps]
    X(i+1,:) = X(i,:)*exp(-1*alpha*dt) + mu*(1-exp(-1*alpha*dt)) + ...
                sigma*sqrt((1-exp(-2*alpha*dt))/(2*alpha))*N(i,:) ;
end

%% 2A Expected, mean and sample paths and long term average

close all
figure(1)
EX = X0*exp(-alpha*t) + mu*( 1-exp(-alpha*t) ) ;
plot(t,EX,'r.',t,mean(X,1),'k.',t,mu*ones(size(t)),'k--',t,X(1:1000:end,:))
legend('Expected path','Mean path','Long-term average')
xlabel('t')
ylabel('X')
% We also compute the standard deviation (sqrt(VarX)) as t -> inf
sdevinfty = sigma/sqrt(2*alpha);
ylim([mu-4*sdevinfty,mu+4*sdevinfty])
title('Ornstein-Uhlenbeck process dX = \alpha(\mu-X)dt + \sigmadW')

%% 2B Expected, mean and sample paths and long term average

close all
figure(2)
EX = X0*exp(-alpha*t) + mu*( 1-exp(-alpha*t) ) ;
plot(t,EX,'r.',t,mean(X,2),'k.',t,mu*ones(size(t)),'k--',t,X(:,1:1000:end))
legend('Expected path','Mean path','Long-term average')
xlabel('t')
ylabel('X')
% We also compute the standard deviation (sqrt(VarX)) as t -> inf
sdevinfty = sigma/sqrt(2*alpha);
ylim([mu-4*sdevinfty,mu+4*sdevinfty])
title('Ornstein-Uhlenbeck process dX = \alpha(\mu-X)dt + \sigmadW')

%% 3A Variance = Mean Square Deviation

figure(3)
VARX = sigma^2/(2*alpha)*(1-exp(-2*alpha*t)) ;
plot(t,VARX,'r',t,sigma^2*t,'g',t,sigma^2/(2*alpha)*ones(size(t)),'b',t,var(X,'',1),'m', ...
    t,mean((X-EX).^2,1),'c--')
legend('Theory','\sigma^2t','\sigma^2/(2\alpha)','Sampled 1','Sampled 2','Location','SouthEast')
xlabel('t')
ylabel('Var(X) = E((X-E(X))^2)')
ylim([0 0.0006])
title('Ornstein-Uhlenbeck process: variance')

%% 3B Variance = Mean Square Deviation

figure(4)
VARX = sigma^2/(2*alpha)*(1-exp(-2*alpha*t)) ;
plot(t,VARX,'r',t,sigma^2*t,'g',t,sigma^2/(2*alpha)*ones(size(t)),'b',t,var(X,'',2),'m', ...
    t,mean((X-EX').^2,2),'c--')
% Note the transpose of EX in the above line to account for dimensions
legend('Theory','\sigma^2t','\sigma^2/(2\alpha)','Sampled 1','Sampled 2','Location','SouthEast')
xlabel('t')
ylabel('Var(X) = E((X-E(X))^2)')
ylim([0 0.0006])
title('Ornstein-Uhlenbeck process: variance')

%% 4A Mean Absolute Deviation

figure(5)
plot(t,sigma*sqrt((1-exp(-2*alpha*t))/(pi*alpha)),'r',t,sigma*sqrt(2*t/pi),'g', ...
    t,sigma/sqrt(pi*alpha)*ones(size(t)),'b',t,mean(abs(X-EX),1),'m')
legend('Theory','\sigma(2t/\pi)^{1/2}','Long-term average','Sampled','Location','SouthEast')
xlabel('t')
ylabel('E(|X-E(X)|) = (2Var(X)/pi)^{1/2}')
ylim([0 0.02])
title('Ornstein-Uhlenbeck process: mean absolute deviation')

%% 4B Mean Absolute Deviation

figure(6)
plot(t,sigma*sqrt((1-exp(-2*alpha*t))/(pi*alpha)),'r',t,sigma*sqrt(2*t/pi),'g', ...
    t,sigma/sqrt(pi*alpha)*ones(size(t)),'b',t,mean(abs(X-EX'),2),'m')
% Note the transpose of EX in the above line to account for dimensions
legend('Theory','\sigma(2t/\pi)^{1/2}','Long-term average','Sampled','Location','SouthEast')
xlabel('t')
ylabel('E(|X-E(X)|) = (2Var(X)/pi)^{1/2}')
ylim([0 0.02])
title('Ornstein-Uhlenbeck process: mean absolute deviation')

%% 5A Autocovariance

C = zeros([npaths,2*nsteps+1]) ;
for i = [1:npaths] 
    C(i,:) = xcorr(X(i,:)-EX,'unbiased') ;
end
C = mean(C,1) ;

figure(7)
plot(t,sigma^2/(2*alpha)*exp(-alpha*t),'r',t,C(nsteps+1:end),'b',t,sigma^2/(2*alpha)*ones(size(t)),'g.',t,mean(var(X,0,1))*ones(size(t)),'c.')
xlabel('\tau')
ylabel('C(\tau)')
legend('Theory','Sampled','Var for infinite t','Average sampled Var','Location','East')
title('Ornstein-Uhlenbeck process: autocovariance')

%% 5B Autocovariance

C = zeros([2*nsteps+1,npaths]) ;
for i = [1:npaths]
    C(:,i) = xcorr(X(:,i)-EX','unbiased') ;
% Note the transpose of EX in the above line to account for dimensions    
end
C = mean(C,2);

figure(8)
plot(t,sigma^2/(2*alpha)*exp(-alpha*t),'r',t,C(nsteps+1:end),'b',t,sigma^2/(2*alpha)*ones(size(t)),'go',t,mean(var(X,0,2))*ones(size(t)),'co')
xlabel('\tau')
ylabel('C(\tau)')
legend('Theory','Sampled','Var for infinite t','Average sampled Var','Location','East')
title('Ornstein-Uhlenbeck process: autocovariance')

%% 6A Autocorrelation

% The autocorrelation is the Covariance/Variance. However, since our OUP is
% only quasi-stationary (i.e. it is only stationary in the limit t -> inf)
% we will compute the autocorrelation as we have done above, in the limit
% as t -> inf

% It can be shown that in the limit, the autocorrelation becomes
% Corr(t,s) = exp(-1*alpha*tau)     with t < s

% Theoretical autocorrelation
CORRX = exp(-1*alpha*t) ;

figure(9)
plot(t,CORRX,'r',t,C(nsteps+1:end)/C(nsteps+1),'b')
xlabel('\tau')
ylabel('c(\tau)')
legend('Theory','Sampled')
title('Ornstein-Uhlenbeck process: autocorrelation')

%% 6B Autocorrelation

% Theoretical autocorrelation
CORRX = exp(-1*alpha*t) ;

figure(10)
plot(t,CORRX,'r',t,C(nsteps+1:end)/C(nsteps+1),'b')
xlabel('\tau')
ylabel('c(\tau)')
legend('Theory','Sampled')
title('Ornstein-Uhlenbeck process: autocorrelation')