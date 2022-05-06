%% Arithmetic Brownian Motion Simulation
%   The formula for ABM is
%   dX(t) = mu*dt + sigma*dW(t) 

clear all
close all

% Parameters
npaths = 20000; % Number of paths to be simulated
T = 1 ; % Time horizon
nsteps = 200 ; % Number of steps to over in [0,T]
dt = T/nsteps ; % Size of the timesteps
t = 0:dt:T ; % Define our time grid
mu = 0.12 ; % Mean/drift for our ABM
sigma = 0.4 ; % Vol/diffusion for our ABM


%% 1A Monte Carlo Simulation - Paths x Timesteps

% Paths as ROWS!
% Timesteps as COLUMNS!
%         t0   t1    t2   ...
% path 1: (0, 0.1, 0.4, ...)
% path 2: (0, -0.3, 0.1, ...)

% Create an [npaths,nsteps] matrix to simulate the value at each time step
% along each path
dX = mu*dt + sigma*sqrt(dt)*randn([npaths,nsteps]) ;

% Now we need to cumulatively sum the values over the time steps to get
% each path
X = [zeros([npaths,1]) cumsum(dX,2)] ;
% Note the 2 in cumsum to show we are adding each column to the prev. one

%% 1B Monte Carlo Simulation - Timesteps x Paths

% Timesteps as ROWS!
% Paths as COLUMNS!
%   path1    path2
% t0 ( 0  ,  0  ... )
% t1 (0.1 , -0.3 .. )
% t2 (0.4 , 0.1 ... )

% Create an [nsteps,npaths] matrix to simulate the value at each time step
% along each path
dX = mu*dt + sigma*sqrt(dt)*randn([nsteps,npaths]) ;

% Now we need to cumulateively sum the values over the time steps to get
% each path
X = [zeros([1,npaths]) ; cumsum(dX,1)] ;
% Note the 1 in cumsum to show we are adding each row to the prev. one
% AND we have a ';' to account for the rows going downwards

%% 2A Expected, mean and sample paths - Paths x Timesteps

close all
figure(1)
EX = mu*t ; % The expected path, i.e. with no randomness dW
plot(t,EX,'r.',t,mean(X,1),'k.',t,X(1:1000:end,:))
% Note the 1 in the mean expression to indicate we are taking an average
% of all paths (rows) at each timestep (columns). This will give us one
% path with an average value at each timestep, i.e. a [1,nsteps] vector.
legend('Expected path','Mean path')
xlabel('t')
ylabel('X')
ylim([-1,1]);
title('Arithmetic Brownian motion dX(t) = \mudt + \sigmadW(t)')

%% 2B Expected, mean and sample paths - Timesteps x Paths

close all
figure(2)
EX = mu*t ; % The expected path, i.e. with no randomness dW
plot(t,EX,'r.', t,mean(X,2),'k.',t,X(:,1:1000:end))
% Note the 2 in the mean expression to indicate we are taking an average
% of all paths (columns) at each timestep (rows). This will give us one
% path with an average value at each timestep, i.e. a [nsteps,1] vector. 
legend('Expected path','Mean path')
xlabel('t')
ylabel('X')
ylim([-1,1]);
title('Arithmetic Brownian motion dX(t) = \mudt + \sigmadW(t)')

%% 3A Variance = Mean Square Deviation = Mean Square Displacement of Random Part
% From formula for ABM we know the random part: sigma*dW(t)
% So the square of this is: sigma^2*dt (since dW^2 = dt)

close all
figure(3)
plot(t,sigma^2*t,'r',t,var(X,'',1),'k')
legend('Theory: \sigma^2t = 2Dt','Sampled','Location','NorthWest')
% Here our 2Dt refers to the Fokker-Planck equation. D in that equation
% multiples the diffusion part and is set to 0.5*sigma^2 hence, subbing D
% into the above equation yields sigma^2*t giving equality.
xlabel('t')
ylabel('Var(X) = E((X-E(X))^2)')
title('Arithmetic Brownian motion: MSD')

%% 3B Variance = Mean Square Deviation = Mean Square Displacement of Random Part

close all
figure(4)
plot(t,sigma^2*t,'r',t,var(X,'',2),'k')
legend('Theory: \sigma^2t = 2Dt','Sampled','Location','NorthWest')
xlabel('t')
ylabel('Var(X) = E((X-E(X))^2)')
title('Arithmetic Brownian motion: MSD')

%% 4A Mean Absolute Deviation
% This is given by E(|X - EX|)
% Apparently if you compute this for ABM you reach a theoretical value of
% sigma*sqrt(2t/pi). Which is equivalent to sqrt(2*VarX / pi)
% Unfortunately I cannot get there, so we will have to take his word

close all
figure(5)
plot(t,sigma*sqrt(2*t/pi),t,mean(abs(X-EX),1))
legend('Theory: \sigma(2t/\pi)^{1/2}','Sampled','Location','NorthWest')
xlabel('t')
ylabel('E(|X-E(X)|) = (2Var(X)/pi)^{1/2}')
ylim([0 0.02])
title('Arithmetic Brownian Motion: mean absolute deviation')

%% 4B Mean Absolute Deviation
% This is given by E(|X - EX|)

close all
figure(6)
plot(t,sigma*sqrt(2*t/pi),t,mean(abs(X-EX'),2))
% Note that we need to transpose our EX vector. Alternatively we could do
% that further up where we have defined EX, but we've done it here. 
legend('Theory: \sigma(2t/\pi)^{1/2}','Sampled','Location','NorthWest')
xlabel('t')
ylabel('E(|X-E(X)|) = (2Var(X)/pi)^{1/2}')
ylim([0 0.02])
title('Arithmetic Brownian Motion: mean absolute deviation')

%% 5A Probability Distribution at different times
% Here we are plotting histograms at different times, to show how the
% probability distribution evolves (by considering the paths)
% Note the difference between the column number (20/80/end) and the ylabel,
% this is because we discretized our grid into 200 steps so at timestep 80
% we are 40% (0.4) of the way through to T=1

figure(7)

subplot(3,1,1)
histogram(X(:,20),-1:0.02:1,'normalization','pdf');
ylabel('f_X(x,0.1)')
xlim([-1,1])
ylim([0,3.5])
title('Arithmetic Brownian motion: PDF at different times')

subplot(3,1,2)
histogram(X(:,80),-1:0.02:1,'normalization','pdf');
xlim([-1,1])
ylim([0,3.5])
ylabel('f_X(x,0.4)')

subplot(3,1,3)
histogram(X(:,end),-1:0.02:1,'normalization','pdf');
xlim([-1,1])
ylim([0,3.5])
xlabel('x')
ylabel('f_X(x,1)')

%% 5B Probability Distribution at different times

figure(8)

subplot(3,1,1)
histogram(X(20,:),-1:0.02:1,'normalization','pdf');
ylabel('f_X(x,0.1)')
xlim([-1,1])
ylim([0,3.5])
title('Arithmetic Brownian motion: PDF at different times')

subplot(3,1,2)
histogram(X(80,:),-1:0.02:1,'normalization','pdf');
xlim([-1,1])
ylim([0,3.5])
ylabel('f_X(x,0.4)')

subplot(3,1,3)
histogram(X(end,:),-1:0.02:1,'normalization','pdf');
xlim([-1,1])
ylim([0,3.5])
xlabel('x')
ylabel('f_X(x,1)')




