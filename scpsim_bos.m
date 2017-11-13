function scpsim_bos
% BoS game simulations

% matlabpool(4); % use only once to open the parallel pool

n_trials = 10000;

P_own_A = 0.8;
P_own_B = 0.8;

% 1 - own
% 0 - other

% payoff matrix
PA = [4 2; 1 3]; % for A, own choice is 1, other - 2
PB = [3 2; 1 4]; % for B, own choice is 2, other - 1

n_exp = 50;
P_own_A = 0:1/(n_exp-1):1;
P_own_B = 0:1/(n_exp-1):1;

TOPLOT = 0;
mean_joint_reward	= zeros(n_exp,n_exp);
R_A			= zeros(n_exp,n_exp);
R_B			= zeros(n_exp,n_exp);

for e1 = 1:n_exp,
	parfor e2 = 1:n_exp,
		[mean_joint_reward(e1,e2), mean_R_A(e1,e2), mean_R_B(e1,e2), O_A, O_B, R_A, R_B] = testsim_bos_one_exp(n_trials,P_own_A(e1),P_own_B(e2),PA,PB,TOPLOT);
	end
end

figure;
% plot(P_own_A,mean_joint_reward,'k.:');
% xlabel('prob_{own}');
% ylabel('mean joint reward');


pcolor(P_own_A,P_own_B,mean_joint_reward);
xlabel('P_{A own}');
ylabel('P_{B own}');
title('Mean joint reward')
axis equal
axis square

figure;
pcolor(P_own_A,P_own_B,mean_R_A-mean_R_B);
xlabel('P_{A own}');
ylabel('P_{B own}');
title('mean reward A - mean reward B')
axis equal
axis square

function [mean_joint_reward, mean_R_A, mean_R_B, O_A, O_B, R_A, R_B] = testsim_bos_one_exp(n_trials,P_own_A,P_own_B,PA,PB,TOPLOT)
% run one experiment/session of BoS


for t = 1:n_trials,
	O_A(t) = 2 - binornd(1,P_own_A); 
	O_B(t) = 1 + binornd(1,P_own_B); 
	
	R_A(t) = PA(O_A(t),O_B(t));
	R_B(t) = PB(O_A(t),O_B(t));
end
mean_joint_reward = mean((R_A + R_B)/2);
mean_R_A = mean(R_A);
mean_R_B = mean(R_B);

if TOPLOT,
	figure('Position',[100 100 1400 800]);
	subplot(2,1,1);
	
	plot(R_A,'r:'); hold on
	plot(R_B,'b:');
	plot((R_A + R_B)/2,'m:');
	
	line([1 n_trials],[mean_joint_reward mean_joint_reward],'Color',[0.5 0 0.5],'LineWidth',2);
	set(gca,'Ylim',[0.9 4.1]);
end




