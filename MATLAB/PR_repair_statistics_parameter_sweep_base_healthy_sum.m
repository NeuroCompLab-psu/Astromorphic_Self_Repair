totaltime = 400;
dt = 0.001;
totalStep = 0:dt:totaltime;

middle = round(totaltime/dt * 1/2);
seqend = round(totaltime/dt) + 1;

healthy_stable_start = round(totaltime/dt * 7/16);
healthy_stable_end = middle;
longterm_repair_start = round(totaltime/dt * 3/4);
longterm_repair_end = seqend;

num_round = 400;
synapse_per_neuron = 80;
raisingTimeMeasureWindowSize = 500;

healthy_sum_PR_before_fault = zeros(1, num_round);
all_sum_PR_before_fault = zeros(1, num_round);
PR_raising_ratio = zeros(1, num_round);
raising_time_constant = zeros(1, num_round);
reach_1_indicator = zeros(1, num_round);

parfor i = 1:num_round
    fprintf("Round: %d\n", i)

    faulty_percentage = rand()*0.6 + 0.3;
    num_healthy_synapse = synapse_per_neuron - round(synapse_per_neuron * faulty_percentage);
    
    ave_PR = rand() * 0.4 + 0.2;
    
    PR_seq = PR_simulation_stuck_to_zero(totaltime,dt,synapse_per_neuron,faulty_percentage,ave_PR);
    all_PR_before_fault = mean(PR_seq(healthy_stable_start:healthy_stable_end,:),1);
    
    healthy_synapse_PR = PR_seq(:,(end-num_healthy_synapse+1):end);
    healthy_stable_ave_PR = mean(healthy_synapse_PR(healthy_stable_start:healthy_stable_end,:),1);
    
    longterm_repair_ave_PR = mean(healthy_synapse_PR(longterm_repair_start:longterm_repair_end,:),1);
    
    reach_1_indicator(i) = any(reshape(healthy_synapse_PR(longterm_repair_start:longterm_repair_end,:),1,[]) >= 1.0);
    healthy_sum_PR_before_fault(i) = sum(healthy_stable_ave_PR);
    all_sum_PR_before_fault(i) = sum(all_PR_before_fault);
    
    PR_raising_ratio(i) = mean(longterm_repair_ave_PR./healthy_stable_ave_PR);
    raising_thres_632 = 0.632 * longterm_repair_ave_PR + (1 - 0.632) * healthy_stable_ave_PR;
    
    move_mean_all_channel = movmean(healthy_synapse_PR, raisingTimeMeasureWindowSize, 1);
    raising_hit_thres_idx = zeros(1,num_healthy_synapse);
    for syn = 1:num_healthy_synapse
        raising_hit_thres_idx(syn) = find(move_mean_all_channel((middle + 1):end, syn) > raising_thres_632(syn), 1);
    end
    raising_time_constant(i) = (mean(raising_hit_thres_idx) - 1) * dt;
end

%% q
f1=figure;
f1.Position = [100 100 800 400];
h = axes;
set(h,'position',[.1 .1 .85 .9]);
set(gcf,'color','w');
hold on
x = 0.02:0.01:1;
y = 1.03./(x+0.04);
plot(x,y,'r--','linewidth',1)
plot(healthy_sum_PR_before_fault'./all_sum_PR_before_fault', PR_raising_ratio', 'k.','MarkerSize',10)
axis([0,1,0,10])

ax = gca;
ax.LineWidth = 2;
daspect([1 22 1])

legend('$q=\frac{1.03}{z+0.04}$','Simulation results','Interpreter','latex')
xlabel("$z$",'Interpreter','latex')
ylabel("$q$",'Interpreter','latex')
set(gca,'fontsize',24, 'fontweight','normal','fontname','times')

%% tau
f2=figure;
f2.Position = [100 100 800 400];
h = axes;
set(h,'position',[.1 .1 .85 .9]);
set(gcf,'color','w');
plot(healthy_sum_PR_before_fault./all_sum_PR_before_fault, raising_time_constant,'k.','MarkerSize',10)
axis([0,1,0,4.5])
box off

ax = gca;
ax.LineWidth = 2;
daspect([1 9.5 1])
xlabel("$z$",'Interpreter','latex')
ylabel("$\tau$ (s)",'Interpreter','latex')
set(gca,'fontsize',24, 'fontweight','normal','fontname','times')





