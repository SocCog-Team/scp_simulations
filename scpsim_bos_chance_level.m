% function scpsim_bos_chance_level
% BoS game simulations

% matlabpool(4); % use only once to open the parallel pool

n_trials = 10000;

P_own_A = 0.75;
P_own_B = 0.75;

% 1 - own
% 0 - other

% payoff matrix
PA = [2 4; ...
      3 1]; % for A, own choice is 1, other - 2
PB = [2 3; ...
      4 1]; % for B, own choice is 2, other - 1

CA = rand(1,n_trials) < P_own_A;
CB = rand(1,n_trials) < P_own_B;

RA = sum(PA(1,1)*(CA & CB) + PA(2,2)*(~CA & ~CB) + PA(1,2)*(CA & ~CB) + PA(2,1)*(CB & ~CA))/n_trials
RB = sum(PB(1,1)*(CA & CB) + PB(2,2)*(~CA & ~CB) + PB(1,2)*(CA & ~CB) + PB(2,1)*(CB & ~CA))/n_trials

JR = (RA+RB)/ 2
