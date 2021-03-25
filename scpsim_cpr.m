function scpsim_cpr

%{
% CPR dyadic game simulations
SA = 1;
SB = 1;

CA = 0.7;
CB = 0.4;

R = 1;

RA = SA*max([CA CB])*R*(CA/(CA+CB))
RB = SB*max([CA CB])*R*(CB/(CA+CB))
%}

%% Reward matrix: Solo task

% Calculate initial reward matrix
acc              	= 0:.001:1;
conf                = 0:.001:1;
rew                 = acc' * conf; % IK: acc' .* conf does not work for me in MATLAB 2014


% Visualise matrix
f                   = figure;
h                   = surf(acc,conf,rew);
h.LineStyle         = 'none';
ax                  = gca;
ax.XLabel.String    = 'Accuracy';
ax.YLabel.String    = 'Confidence';
ax.ZLabel.String    = '% Reward';
ax.FontSize         = 16;
colormap(jet(256))

% Determine arc width for each confidence level
for j = 1:length(conf)
    arc(j)          = 180 - (180 * conf(j));
end

% Cap arc width at target diameter (currently 2dva == 12.7587deg at chosen position)
idx                 = arc < 12.76;
arc(idx)            = 12.76;

% For each confidence level, calculate minimum accuracy required to hit
% the target at given arc width -> normalised values
hit_width_acc       = 1 - ((arc/2) / 180);
hit_width_acc(idx)  = 1 - (12.76/2)/180; % arc width fixed to target diameter

% Remove position from reward matrix that cannot possibly yield reward due
% to arc width modulation
for iAcc = 1:length(acc)
    indx            = conf < hit_width_acc(iAcc);
    rew(iAcc,indx)  = 0;
end

% Plot actual reward matrix
f                   = figure;
ax                  = gca;
im                  = imagesc(acc,conf,rew);
ax.XLabel.String    = 'Accuracy';
ax.YLabel.String    = 'Confidence';
ax.FontSize         = 16;
ax.XLim             = [hit_width_acc(1) 1];
cb                  = colorbar;
cb.Label.String     = '% Reward';
cmap                = [jet(256)];
set(gca,'Ydir','normal');
colormap(cmap)

% Plot average reward distribution for hits at different confidence/accuracy levels
for k = 1:length(conf)
    rew_conf(k)     = nanmean(randsample(rew(conf==conf(k),:),100000,'true'));
    rew_acc(k)      = nanmean(randsample(rew(:,acc==acc(k)),100000,'true'));
end

figure
plot(conf,rew_conf)
xlabel('Confidence');
ylabel('Mean reward');

figure
plot(conf,rew_acc)
xlabel('Accuracy');
ylabel('Mean reward');


% Initial version
%{ 
% reward solo task
acc = 0:.01:1;
conf = 0:.01:1;
% rew = acc' .* conf; % I showed this before

 
for a = 1:length(acc)
        for c = 1:length(conf)
            % rew(c,a) = mean([acc(a) conf(c)]);
            rew(c,a) = acc(a)*conf(c); % note: conf. is the first index (row)
            if acc(a) < conf(c),
                   rew(c,a) = 0;
            end
        end
end

 
surf(acc,conf,rew)
xlabel('Accuracy');
ylabel('Confidence');
zlabel('% Reward');

% random accuracy, at different confidence levels
for k = 1:length(conf),
    rew_conf(k) = mean(randsample(rew(find(conf==conf(k)),:),100000,'true'));
end
figure
plot(conf,rew_conf)
xlabel('Confidence');
ylabel('Mean reward');
%}



