%% Geometric Brownian Motion simulation
%   The formula for GBM is
%   dS(t) = mu*S*dt + sigma*S*dW(t) 
% However, using the transform X = log(S/S0) we can show that GBM is in 
% fact ABM, with a = (mu - 0.5*simga^2) multiplying dt, and sigma
% multiplying dW. 
% That is: dX = (mu - 0.5*sigma^2)*dt + sigma*dW
% We will use this result to compute our ABM (as before) and then transform
% back at the end using S = S0*exp(X)

clear all
close all

% --------------------------------------------------
% Simulate our ABM
% --------------------------------------------------

% Parameters
npaths = 20000; % Number of paths to be simulated
T = 1 ; % Time horizon
nsteps = 200 ; % Number of steps to over in [0,T]
dt = T/nsteps ; % Size of the timesteps
t = 0:dt:T ; % Define our time grid
mu = 0.2 ; % Mean/drift for our ABM
sigma = 0.4 ; % Vol/diffusion for our ABM
S0 = 1 ; % Our initial stock price

%% 1A Monte Carlo Simulation - Paths x Timesteps

% Paths as ROWS!
% Timesteps as COLUMNS!
%         t0   t1    t2   ...
% path 1: (0, 0.1, 0.4, ...)
% path 2: (0, -0.3, 0.1, ...)

% Create an [npaths,nsteps] matrix to simulate the value at each time step
% along each path
dX = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*randn([npaths,nsteps]) ;

% Now we need to cumulatively sum the values over the time steps to get
% each path
X = [zeros([npaths,1]) cumsum(dX,2)] ;
% Note the 2 in cumsum to show we are adding each column to the prev. one

% Now we must transform back to what we want
S = S0*exp(X) ;

%% 1B Monte Carlo Simulation - Timesteps x Paths

% Timesteps as ROWS!
% Paths as COLUMNS!
%   path1    path2
% t0 ( 0  ,  0  ... )
% t1 (0.1 , -0.3 .. )
% t2 (0.4 , 0.1 ... )

% Create an [nsteps,npaths] matrix to simulate the value at each time step
% along each path
dX = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*randn([nsteps,npaths]) ;

% Now we need to cumulateively sum the values over the time steps to get
% each path
X = [zeros([1,npaths]) ; cumsum(dX,1)] ;
% Note the 1 in cumsum to show we are adding each row to the prev. one
% AND we have a ';' to account for the rows going downwards

% Now we must transform back to what we want
S = S0*exp(X) ;

%% 2A Expected, mean and sample paths - Paths x Timesteps

close all
figure(1)
ES = S0*exp(mu*t) ; % The expected path, i.e. with no randomness dW
plot(t,ES,'r.',t,mean(S,1),'k.',t,S(1:1000:end,:))
% Note the 1 in the mean expression to indicate we are taking an average
% of all paths (rows) at each timestep (columns). This will give us one
% path with an average value at each timestep, i.e. a [1,nsteps] vector.
legend('Expected path','Mean path')
xlabel('t')
ylabel('S')
ylim([0,2.5])
title('Geometric Brownian motion dS = \muSdt + \sigmaSdW')

%% 2B Expected, mean and sample paths - Timesteps x Paths

close all
figure(2)
ES = S0*exp(mu*t) ; % The expected path, i.e. with no randomness dW
plot(t,ES,'r.',t,mean(S,2)','k.',t,S(:,1:1000:end))
% Note the 2 in the mean expression to indicate we are taking an average
% of all paths (columns) at each timestep (rows). This will give us one
% path with an average value at each timestep, i.e. a [nsteps,1] vector. 
legend('Expected path','Mean path')
xlabel('t')
ylabel('S')
ylim([0,2.5])
title('Geometric Brownian motion dS = \muSdt + \sigmaSdW')

%% 3A Variance

% Theoretical value of 2nd moment
%ES2 = (S0^2)*exp(2*t*(mu-0.5*sigma^2) + 2*t*sigma^2) ; 
ES2 = (S0^2)*exp(2*t*mu + t*sigma^2) ; 

% Theoretical value of Var(S)
VARS = ES2 - ES.^2 ;

figure(3)
plot(t,VARS,'r',t,var(S,'',1),'m',t,mean((S-ES).^2,1),'c--') 
legend('Theory','Sampled 1','Sampled 2','Location','SouthEast')
xlabel('t')
ylabel('Var(X) = E((X-E(X))^2)')
%ylim([0 0.0006])
title('Geometric Brownian Motion: variance')




