% scpsim_cpr_solo_dyadic_predictions

% Define the vector P1
Ps = -0.1:0.01:0.1;

% Generate all possible combinations using ndgrid
[Ps1, Ps2] = ndgrid(Ps, Ps);

% Reshape the matrices into two columns
Ps1 = Ps1(:);
Ps2 = Ps2(:);

N = numel(Ps1);

% add jitter
Ps1 = Ps1 + 0.01*(rand(N,1)-0.5);
Ps2 = Ps2 + 0.01*(rand(N,1)-0.5);



% AUC is proportional to difference in solo

random_offset = (rand(N,1) - 0.5)/1.*abs(Ps1-Ps2);

AUC1 = 0.5 - (Ps1 - Ps2) + random_offset;
AUC2 = 0.5 - (Ps2 - Ps1) + random_offset;

figure
plot(Ps1 - Ps2, AUC1, 'ko','MarkerFaceColor',[0 0 0]); hold on
plot(Ps1 - Ps2, AUC2, 'ko','MarkerFaceColor',[1 1 1]); hold on
line([Ps1 - Ps2, Ps1 - Ps2]',[AUC1, AUC2]');








