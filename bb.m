%% Simulation of a Brownian Bridge
%  The formula for our Brownian Bridge is
%  dX(t) = (b-X)/(T-t) *dt + sigma*dW(t)
% The BB presents a different type of SDE than what we have previously seen
% as it requires both a known start and end point. It then proceeds to
% simulate the different paths between those two points but its start and
% end will always be the same regardless of the path taken.
% This applies in Fixed Income, e.g. Bond pricing

clear all
close all

% Parameters
npaths = 20000 ; % Number of paths to be simulated
T = 1 ; % Time Horizon
nsteps = 200 ; % Number of timesteps
dt = T/nsteps ; % Size of our timesteps
t = 0:dt:T ; % Discretization of our grid
a = 0.8 ; % Our starting point
b = 1 ; % Our ending point
sigma = 0.3 ; % Our vol/diffusion term

%% Monte Carlo Simulation - npaths x nsteps - Traditional for loop

% We need to initialise our matrix such that the start and end points are a
% and b respectively. Note we use (nsteps-1) zeros in between.
X = [a*ones([npaths,1]) , zeros([npaths,nsteps-1]) , b*ones([npaths,1])] ;

% We will use Euler-Maruyama
 for i = [1:nsteps-1]
     X(:,i+1) = X(:,i) + (b - X(:,i))/(nsteps-i+1) + sigma*sqrt(dt)*randn([npaths,1]) ;
 end

%% Monte Carlo Simulation - npaths x nsteps - BB via ABM

% Due to some nice properties, we can simulate the BB via a driftless ABM
% i.e. A purely random ABM with the formula
% dX(t) = sigma*dW(t)

% We will use dW rather than dX to save X for our BB
% Simulate our ABM
%dW = sigma*sqrt(dt)*randn([npaths,nsteps]) ;

% Now we cumulatively sum the columns, remembering to include a at the
% beginning since we ultimately want a BB
%W = cumsum( [a*ones([npaths,1]) , dW], 2);

% Our BB formula is then given by X(t) = W(t) + (b-W(T))/T*t
%X = W + repmat( b-W(:,end),1,nsteps+1)/T .* repmat(t,npaths,1);
% In the above:
% repmat( b-W(:,end),1,nsteps+1) means take b subtract the last column of W
% for all rows and repeat it 1 times down (i.e. keep the same number of
% paths) and nsteps+1 times across (i.e. the number of timesteps) to form a
% [npaths,nsteps+1] matrix. 
% AND
% repmat(t,npaths,1) means create a [1,t] vector with the time t
% as each column and do that npaths many rows to create a [npaths,nsteps+1]
% matrix

%% Expected, mean and sample paths

close all
figure(1)
% The expected path below comes from Ballotta & Fusai p.135, where they
% have defined the E(X) on an interval [s,T], which is more general. In our
% case we have defined our interval to be [0,T], hence s=0 drops out and we
% are left with the formula below.
EX = a + t*(b-a)/T ; 
plot(t,EX,'r',t,mean(X,1),'k',t,X(1:1000:end,:))
legend('Expected path','Mean path')
xlabel('t')
ylabel('X')
% The below are some additonal conditions that scale the picture according
% to some conditions - more of a nice to have. 
sdevmax = sigma*sqrt(T)/2;
ylim([(a+b)/2-4*sdevmax (a+b)/2+4*sdevmax])
title('Brownian bridge dX = ((b-X)/(T-t))dt + \sigmadW')

%% Variance = Mean Square Deviation

figure(2)
VARX = (sigma^2)*(t/T).*(T-t) ;
% Note that in Ballotta & Fusai they have no sigma term in the SDE for the
% BB (i.e. sigma=1). However, when they quote the variance then there is a
% sigma^2 missing from their figure (since sigma^2 = 1^2 = 1). But we have
% taken sigma to be a different value and therefore must multiply our
% theoretical variance formula by sigma^2.
plot(t,VARX,'r',t,var(X,'',1),'m',t,mean((X-EX).^2,1),'c--')
legend('Theory','Sampled 1','Sampled 2','Location','SouthEast')
xlabel('t')
ylabel('Var(X) = E((X-E(X))^2)')
%ylim([0 0.0006])
title('Brownian Bridge Process: variance')


%% Autocovariance

% Is this possible?? 
% Does it have to be a meshgrid instead?

C = zeros([npaths,2*nsteps+1]) ;
for i = [1:npaths]
    C(i,:) = xcorr(X(i,:)-EX,'unbiased') ;
end
C = mean(C,1) ;

% The theoretical value for the autocovariance is (s<t) on [0,T]
%  c_X(s,t) = min(s,t) - (s*t)/T
% Because we want the autocovariance across the entire range [0,T], we can
% set s = t (our timesteps) and t = T (our time horizon). This reduce our
% equation to

figure(3)
plot(t,C(nsteps+1:end),'r')






