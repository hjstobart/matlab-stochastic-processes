%% Simulation of the Merton Jump Diffusion Process
% We now introduce stochastic processes with jumps, this means at a random
% point our process can jump up. This is generally considered a better way
% of modelling prices since empiracally they do tend to jump.

%  Our formula for the MJD is given in terms of X(t) rather than dX(t)
%  X(t) = (mu_S - 0.5*sigma_S^2)*t + sigma*W(t) + sum_{i=1}^{N(t)} Z_i

% Note the above is our ABM for X(t), where X(t) is log(S/S0) i.e. the log
% of the stock price.

% The extra sum term in our solution is the random jump part.
% N(t) is a Poisson process, with arrival rate lamdba
% Z_i is our sequence of i.i.d random variables
% In the MJD process the random variables we will be using as our Z_i are
% normal distributions
% Unfortunately, since we are now dealing with two different distributions
% we need to distinguish between the parameters associated with each of
% them.

% We shall define:
% mu_S : the mean/drift of our traditional ABM (for simplicity muS)
% sigma_S : the vol/diffusion of our traditional ABM (again sigmaS)
% ... AND ...
% lambda : the rate of arrival for our Poisson Process
% mu_J : the mean/drift of our i.i.d Gaussian random variables (muJ)
% sigma_J : the vol/diffusion of our i.i.d Gaussians (sigmaJ)

clear all
close all

% Parameters
npaths = 20000 ; % Number of paths to be simulated
T = 1 ; % Time horzion
nsteps = 200 ; % Number of timesteps
dt = T/nsteps ; % Size of timesteps
t = 0:dt:T ; % Discretization of our grid
muS = 0.2 ; % Drift for ABM
sigmaS = 0.3 ; % Diffusion for ABM
lambda = 0.5 ; % Rate of arrival for Poisson Process
muJ = -0.1 ;  % Drift for Jumps
sigmaJ = 0.15 ; % Diffusion for Jumps
S0 = 1 ; % Initial stock price

%% Monte Carlo Simulation - npaths x nsteps

% We calculate our traditional ABM of the form of the equation
% Algorithm Step 1 - Ballotta & Fusai p.178
dW = (muS - 0.5*sigmaS^2)*dt + sigmaS*sqrt(dt)*randn([npaths,nsteps]) ;

% Recall a Poisson Distribution ~Poi(lambda) can be interpreted by thinking
% of lambda as the expected number of events occuring. For instance,
% arrivals at a hospital in a certain hour can be modelled as a Poi(3)
% meaning we expect 3 people to arrive in any given hour. But of course it
% could be 1 (unlikley), 2 (more likely), right the way up to 10 and beyond
% (v. unlikely). They are all discrete though. So in our situation here,
% with lambda = 0.5, we are saying that we expect to jump about half the
% time, which means our values will be 0 (we don't jump) or 1 (we do jump)
% or potentially 2 on rare occasions (a v. big jump)

% We now need to compute an [npaths,nsteps] matrix of the jump points. That
% is the frequency of the jumps.
% Algorithm Step 2 - Ballotta & Fusai p.178
dN = poissrnd(lambda*dt,[npaths,nsteps]) ;

% Now we need to compute the size of the jumps.
% Algorithm Step 3 - Ballotta & Fusai p.178
dJ = muJ*dN + sigmaJ*sqrt(dN).*randn([npaths,nsteps]) ; 
% Here we are using the 'scale and shift' of our standard normal ~N(0,1) to
% get the scaled normal ~N(muS,sigmaS^2) which determines our jump sizes.

% Adding the two components together gives us the complete value at each 
% timestep for the MJD process
% Algorithm Step 4 - Ballotta & Fusai p.178
dX = dW + dJ ;

% Our final step is to cumulatively sum the columns to produce paths
X = [ zeros([npaths,1]) , cumsum(dX,2)] ;

% Note this computes the paths of the log prices since we have used ABM
% To transform back to stock prices we require one final step
%S = S0*exp(X) ;

%% Expected, mean and sample paths

close all
figure(1)
EX = (muS + lambda*muJ)*t ;
plot(t,EX,'r',t,mean(X,1),'k')%,t,X(1:1000:end,:)) ;
legend('Expected path','Mean path')
xlabel('t')
ylabel('X')
ylim([-1,1.2]);
title('Paths of a Merton jump-diffusion process X = \mut + \sigmaW(t) + \Sigma_{i=1}^{N(t)} Z_i')

%% Variance  = Mean Square Displacement

% Theoretical value for Var(X)
VARX = t*(sigmaS^2 + lambda*(muJ^2 +sigmaJ^2)) ;

figure(2)
plot(t,VARX,'r',t,var(X,'',1),'m',t,mean((X-EX).^2,1),'c--')
legend('Theory','Sampled 1','Sampled 2','Location','SouthEast')
xlabel('t')
ylabel('Var(X) = E((X-E(X))^2)')
ylim([0 0.12])
title('Merton Jump Diffusion process: variance')

%% Probability Density Function at different time

% Parameters for x-axis
dx = 0.02 ;
x = -1:dx:1 ;
xx = x + dx/2 ; % Shift required for bar chart

h1 = histogram(X(:,40),'BinEdges',x,'Normalization','pdf') ;
H1 = [h1.Values,0] ;

h2 = histogram(X(:,100),'BinEdges',x,'Normalization','pdf') ;
H2 = [h2.Values,0] ;

h3 = histogram(X(:,end),'BinEdges',x,'Normalization','pdf') ;
H3 = [h3.Values,0] ;

figure(3)

subplot(3,1,1)
bar(xx,H1)
ylabel('f_X(x,0.2)')
xlim([-1,1])
ylim([0,3])
title('Probability density function of a Merton jump-diffusion process at different times')

subplot(3,1,2)
bar(xx,H2)
xlim([-1,1])
ylim([0,3])
ylabel('f_X(x,0.5)')

subplot(3,1,3)
bar(xx,H3)
xlim([-1,1])
ylim([0,3])
xlabel('x')
ylabel('f_X(x,1)')

