%% Simulation of the Kou Jump Diffusion Process
% Another of our jump diffusion processes is the KJD. This follows the same
% appraoch as the MJD process but uses a different random variable as the
% i.i.d components of the jumps.
% Where the MJD used Gaussians, we will now use the Bilateral Exponential
% distribution. This is a minor modification of the Laplace (or double
% exponential) Distribution as it is no longer symmetric down the y-axis.
% That is we have different exponential distributions for x>0 and for x<0,
% reflecting the fact that prices tend to be asymmetric.

% We follow the same approach as for the MJD, and display the KJD in its
% X(t) form:
%  X(t) = (mu - 0.5*sigma^2)*t + sigma*W(t) + sum_{i=1}^{N(t)} Z_i

% Note the above is our ABM for X(t), where X(t) is log(S/S0) i.e. the log
% of the stock price.

% Let us again define our parameters:
% mu : the mean/drift of our traditional ABM
% sigma : the vol/diffusion of our traditional ABM
% ... AND ...
% lambda : the rate of arrival for our Poisson Process
% eta1 : the upward jump parameter of Bilat. Exp. random variables
% This means the upward jumps have mean 1/eta1
% eta2 : the downward jump parameter of our i.i.d Bilat. Exp.
% This means the downward jumps have mean 1/eta2
% p : the probability of a jump for our i.i.d Bilat. Exp.

clear all
close all

% Parameters
npaths = 20000 ; % Number of paths to be simulated
T = 1 ; % Time horzion
nsteps = 200 ; % Number of timesteps
dt = T/nsteps ; % Size of timesteps
t = 0:dt:T ; % Discretization of our grid
mu = 0.2 ; % Drift for ABM
sigma = 0.3 ; % Diffusion for ABM
lambda = 0.5 ; % Rate of arrival for Poisson Process
eta1 = 6 ;  % Parameter for upward jumps
eta2 = 8 ; % Parameter for downward jumps
p = 0.4 ; % Probability of an upward jump 
S0 = 1 ; % Initial stock price

%% Generating the Bilateral Exponential Random Deviates

% Generate a [npaths,nsteps] matrix of standard uniform random devaites 
U = rand([npaths,nsteps]) ;

% Convert those values in Bilateral Exponential (BE) random deviates
BE = -1/eta1*log((1-U)/p).*(U>=1-p) + 1/eta2*log(U/(1-p)).*(U<1-p) ; 

%% Monte Carlo Simulation - npaths x nsteps

% We calculate our traditional ABM of the form of the equation
dW = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*randn([npaths,nsteps]) ;

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
dN = poissrnd(lambda*dt,[npaths,nsteps]) ;

% Now we need to compute the size of the jumps.
% This is simply computing the size of the jumps (given by matrix BE) and
% when they occur (given by matrix dN)
% Its output will be a matrix that has components 0 (no jump) or some 
% value (the size of the jump)
dJ = dN .* BE ;

% Adding the two components together gives us the complete value at each 
% timestep for the KJD process
dX = dW + dJ ;

% Our final step is to cumulatively sum the columns to produce paths
X = [ zeros([npaths,1]) , cumsum(dX,2)] ;

% Note this computes the paths of the log prices since we have used ABM
% To transform back to stock prices we require one final step
%S = S0*exp(X) ;

%% Expected, mean and sample paths

close all
figure(1)
EX = (mu + lambda * (p/eta1 - (1-p)/eta2) ) * t ;
plot(t,EX,'r',t,mean(X,1),'k',t,X(1:1000:end,:)) ;
legend('Expected path','Mean path')
xlabel('t')
ylabel('X')
ylim([-1,1.2]);
title('Paths of a Kou jump-diffusion process X = \mut + \sigmaW(t) + \Sigma_{i=1}^{N(t)} Z_i')

%% Variance = Mean Square Displacement

% Theoretical value for Var(X)
VARX = t*( sigma^2 + 2*lambda*( p/(eta1^2) + (1-p)/(eta2^2) ) ) ;
figure(2)
plot(t,VARX,'r',t,var(X,'',1),'m',t,mean((X-EX).^2,1),'c--')
legend('Theory','Sampled 1','Sampled 2','Location','SouthEast')
xlabel('t')
ylabel('Var(X) = E((X-E(X))^2)')
ylim([0 0.3])
title('Kou Jump Diffusion process: variance')

%% Probability Density Function at different times

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
title('Probability density function of a Kou jump-diffusion process at different times')

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



