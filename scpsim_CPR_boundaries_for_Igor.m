% addpath /Users/fschneider/Documents/GitHub/CPR/Matlab/PSY_Setup/random_walk
% addpath /Users/fschneider/Documents/MATLAB/CircStat2012a/

close all
clear all

nRep = 10000;
for iTrial = 1:nRep
    % Generate CPR stimulus (1 min == 7200 samples)
    STIM                    = CPR_create_random_walk_v3();
    rdp_dir                 = STIM.RDP_direction_deg;
    
    % Random response
    js_dir                  = rand(size(rdp_dir)).*360;
    
    % Computer response (shifted RDP)
    AGNT.dir_sigma       	= 0;                                        % Direction sigma [deg]
    AGNT.str_sigma       	= 0;                                        % Eccentricity sigma [%]
    AGNT.lag              	= 50;                                       % Delay to reference point [samples]
    AGNT.win              	= 50;                                       % Smoothing window size [samples]
    AGNT.smooth_kernel     	= 'gaussian';                               % Smoothing kernel [samples]
    AGNT                   	= CPR_create_agent_random_walk(STIM,AGNT);
    
    % Random response: Calculate circular distance to RDP direction
    js_ang_dev_random      	= rad2deg(circ_dist(deg2rad(js_dir),deg2rad(rdp_dir))); 
    js_acc_random          	= abs(1 - abs(js_ang_dev_random / 180));    % normalised accuracy
       
    % Computer player: Calculate circular distance to RDP direction
    idx                     = 200:7200; % to remove transient from random starting point of computer player     
    js_ang_dev_agnt        	= rad2deg(circ_dist(deg2rad(AGNT.dir_smooth(idx)),deg2rad(rdp_dir(idx)))); 
    js_acc_agnt            	= abs(1 - abs(js_ang_dev_agnt / 180)); 
    
    % Average of trial
    js_ang_dev_mean_random(iTrial) 	= mean(abs(js_ang_dev_random));
    js_acc_mean_random(iTrial)     	= mean(js_acc_random);
    js_ang_dev_mean_agnt(iTrial)    = mean(abs(js_ang_dev_agnt));
    js_acc_mean_agnt(iTrial)    	= mean(js_acc_agnt);
end

%% PLOT

nBin = 20;
figure
subplot(2,2,1)
histogram(js_ang_dev_mean_random,nBin)
title('Ang. error: Random')
xlabel('Avg. angular distance [deg]')
ylabel('Trials [#]')

subplot(2,2,3)
histogram(js_ang_dev_mean_agnt,nBin)
title('Ang. error: AGNT')
xlabel('Avg. angular distance [deg]')
ylabel('Trials [#]')

subplot(2,2,2)
histogram(js_acc_mean_random,nBin)
title('Accuracy: Random')
xlabel('Avg. accuracy [norm]')
ylabel('Trials [#]')

subplot(2,2,4)
histogram(js_acc_mean_agnt,nBin)
title('Accuracy: AGNT')
xlabel('Avg. accuracy [norm]')
ylabel('Trials [#]')