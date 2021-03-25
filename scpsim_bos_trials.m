n_trials = 1000;

DA = randsample([1:0.5:3],n_trials,true);
DB = randsample([1:0.5:3],n_trials,true);


D1 = min(DA,DB);
D2 = abs(DA-DB);

subplot(4,1,1)
hist(D1)

subplot(4,1,2)
hist(D2)

% from perspective of agent A
subplot(4,1,3)
D1_A = DA;
hist(D1_A)

subplot(4,1,4)
D2_A = DA-DB;
hist(D2_A)


figure
plot(D1,D2,'o');