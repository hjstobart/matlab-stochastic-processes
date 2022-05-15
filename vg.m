%% Simulation of the Variance Gamma Process

% The VG Process is again different to the SDEs we've seen before. Instead
% of traditional SDEs (such as GBM or OUP) and their extension to include 
% random jumps (as seen in MJD or KJD) we now consider time changed
% Brownian Motion. That is we take our traditional ABM but instead of it
% evolving over normal ("calendar") time, we have it evolve over random
% time using a so called "random clock".

%  The formula for our new stochastic process is
%  X(t) = theta*G(t) + sigma*W(G(t))

% For our purposes we will define G(t) to be a Gamma Process with the
% following parameters:
% alpha = lambda = 1/k, where k >0 
% Where we have chosen the above such that:
% E(X) = t : this means our random clock evolves in line with calendar time
% on average
% Var(X) = kt 

% We will now work with the VG in its differential form
%  dX(t) = theta*dG(t) + sigma*dW(G(t))

clear all
close all

% Parameters
npaths = 20000 ; % Number of paths to be simulated
T = 1 ; % Time horizon
nsteps = 200 ; % Number of timesteps
dt = T/nsteps ; % Size of the timesteps
t = 0:dt:T ; % Discretization of our time grid
theta = 0.2 ; % Drift term for our time-changed process
sigma = 0.3 ; % Vol/diffusion term for our time-changed process
kappa = 0.05 ; % Parameter for the Gamma Process = 1/lambda = 1/rate

%% Monte Carlo Simualtion - npaths x nsteps

% First we must compute a [npaths,nsteps] matrix containing the Gamma
% increments of the Gamma random clock.
% Algorithm Step 1 - Ballotta & Fusai p.189
dG = gamrnd(dt/kappa,kappa,[npaths,nsteps]) ;

% Now we compute our traditional ABM but under the Gamma random clock
dX = theta*dG + sigma*sqrt(dG).*randn([npaths,nsteps]) ;

% Now we cumulatively sum the increments
X = [zeros([npaths,1]) , cumsum(dX,2)] ;

%% Expected, mean and sample paths

close all

figure(1)
EX = theta*t ;
plot(t,EX,'r',t,mean(X,1),'k',t,X(1:1000:end,:)) 
legend('Expected path','Mean path')
xlabel('t')
ylabel('X')
ylim([-0.8,1.2])
title('Paths of a variance Gamma process dX(t) = \thetadG(t) + \sigmadW(G(t))')

%% Probability Density Function at different times - Histograms

% Parameters for x-axis
dx = 0.02 ;
x = -0.8:dx:1.2 ;
xx = x + dx/2 ; % Shift required for bar chart

figure(2)
subplot(3,1,1)
h1 = histogram(X(:,40),'BinEdges',x,'Normalization','pdf') ;
H1 = [h1.Values,0] ;
ylabel('f_X(x,0.2)')
xlim([-1,1])
ylim([0,4])
title('Probability density function of a Variance Gamma process at different times')

subplot(3,1,2)
h2 = histogram(X(:,100),'BinEdges',x,'Normalization','pdf') ;
H2 = [h2.Values,0] ;
xlim([-1,1])
ylim([0,4])
ylabel('f_X(x,0.5)')

subplot(3,1,3)
h3 = histogram(X(:,end),'BinEdges',x,'Normalization','pdf') ;
H3 = [h3.Values,0] ;
xlim([-1,1])
ylim([0,4])
xlabel('x')
ylabel('f_X(x,1)')

%% Probability Density Function at different times - Bar Charts
figure(3)
subplot(3,1,1)
bar(xx,H1)
ylabel('f_X(x,0.2)')
xlim([-1,1])
ylim([0,4])
title('Probability density function of a Variance Gamma process at different times')

subplot(3,1,2)
bar(xx,H2)
xlim([-1,1])
ylim([0,4])
ylabel('f_X(x,0.5)')

subplot(3,1,3)
bar(xx,H3)
xlim([-1,1])
ylim([0,4])
xlabel('x')
ylabel('f_X(x,1)')



