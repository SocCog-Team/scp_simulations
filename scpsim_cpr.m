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




