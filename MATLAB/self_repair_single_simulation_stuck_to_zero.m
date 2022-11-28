% A single run of astrocyte repair simulation
% The PR of a portion of the synapses drop to zero in the middle of the entire
% simulation

clear all 
close all
clc

%Comments
%Each synapse is getting same spikes
%1 neuron, 10 synapse

del_t = 0.001;                   %1 ms
totalTime = 400;                 %400 s
totalStep = 0:del_t:totalTime;   %a vector with each time step

middle = round(totalTime/del_t * 1/2);
seqend = round(totalTime/del_t) + 1;

healthy_stable_start = round(totalTime/del_t * 7/16);
healthy_stable_end = middle;
longterm_repair_start = round(totalTime/del_t * 3/4);
longterm_repair_end = seqend;


%%%constants
%astrocyte
IP3star = 0.16;         %Baseline IP3, uM
r_IP3 = 0.5;            %Production IP3, uM/s
r_Glu = 10;             %Production Glu, uM/s
r_AG = 0.8;             %Production AG, uM/s
tau_IP3 = 7;            %Degradation time const, s
tau_Ca = 1;             %Decay rate Ca, s      NO WHERE USED
tau_AG = 10;            %Decay rate AG, s, default 10 ----------
tau_Glu = 0.1;          %Decay rate Glu, s
tau_eSP = 40;           %Decay rate eSP, s
a2 = 0.2;               %Inactivation binding rate, uM/s
r_C = 6;                %Maximum rate of CICR, /s
r_L = 0.11;             %Ca leakage rate, /s
c0 = 2;                 %Total free Ca Conc, uM
c1 = 0.185;             %Ratio of ER vol to cyst vol
v_ER = 0.8;             %Max rate of SERCA uptake, uM/s
k_ER = 0.1;             %SERCA pump Act. const, uM
d1 = 0.13;              %IP3 dissociation const, uM
d2 = 1.049;             %Ca inact dissociation const, uM
d3 = 0.9434;            %IP3 dissociation const, uM
d5 = 0.08234;           %Ca2+ act dissociation const,uM
m_eSP = 55e3;           %eSP weighting factor
Ca_threshold = 0.3;     %Glu release threshold,uM
K_AG = -1000;           %scaling factor
%neuron and synapse
vth = 9;                %firing threshold voltage, mV
Rm = 1.2;               %Membrane Resistance, G Ohm (e9)
tau_mem = 0.06;         %Membrane Time Const, s
I_inj = 6650;           %Injected Current, pA (e-12)
t_refrac = 0.002;       %Refractory period, s

%%%initialization
Ca2 = zeros(size(totalStep,2), 1);
Ca2(1) = 0.071006;
h = zeros(size(totalStep,2),1);
h(1) = 0.7791;
IP3 = zeros(size(totalStep,2),1);
IP3(1) = 0.16;
m_inf = zeros(size(totalStep,2),1);
n_inf = zeros(size(totalStep,2),1);
m_inf(1) = IP3(1)/(IP3(1)+d1);
n_inf(1) = Ca2(1)/(Ca2(1)+d5);
J_chan = zeros(size(totalStep,2),1);
J_chan(1) = r_C*((m_inf(1)*n_inf(1)*h(1))^3)*(c0 - (1+c1)*Ca2(1));
J_pump = zeros(size(totalStep,2),1);
J_pump(1) = v_ER*(((Ca2(1))^2)/(k_ER^2 + (Ca2(1))^2)); 
J_leak = zeros(size(totalStep,2),1);
J_leak(1) = r_L*(c0-(1+c1)*Ca2(1));

AG_N1 = zeros(size(totalStep,2),1);
AG_N2 = zeros(size(totalStep,2),1);

Glu = zeros(size(totalStep,2),1);
eSP = zeros(size(totalStep,2),1);
Q2 = zeros(size(totalStep,2),1);
Q2(1) = d2*((IP3(1)+d1) / (IP3(1)+d3));
tau_h = zeros(size(totalStep,2),1);
tau_h(1) = 1/(a2*(Q2(1)+Ca2(1)));
h_inf = zeros(size(totalStep,2),1);
h_inf(1) = Q2(1)/(Q2(1)+Ca2(1));

synapse_per_neuron = 10;
num_faulty_synapse = round(synapse_per_neuron * 0.7);
average_PR = 0.3;
PR_N1 = zeros(size(totalStep,2), synapse_per_neuron);
PR_N1(1,:) = average_PR*ones(1);
PR_N2 = zeros(size(totalStep,2), synapse_per_neuron);

while true
    PR_N2(1,:) = (randn(1,synapse_per_neuron) - 0.5) * average_PR * 0.3 + average_PR;
    PR_N2(1,:) = PR_N2(1,:) + average_PR - mean(PR_N2(1,:));
    validPR = isempty(find((PR_N2(1,:) > 1),1)) && isempty(find((PR_N2(1,:) < 0),1));
    if validPR
        break
    end
end


Isyn_N1 = zeros(size(totalStep,2),synapse_per_neuron);
Isyn_N2 = zeros(size(totalStep,2),synapse_per_neuron);
DSE_N1 = zeros(size(totalStep,2),1);
DSE_N1(1) = AG_N1(1)*K_AG;
DSE_N2 = zeros(size(totalStep,2),1);
DSE_N2(1) = AG_N2(1)*K_AG;

v_N1 = zeros(size(totalStep,2),1);
v_N2 = zeros(size(totalStep,2),1);

t_sp_N1 = -1;
t_sp_N2 = -1;
t_Ca = -1;
Ca2_reached_threshold = 0;

%%%Generate spikes for the synapses of N1 & N2
spikesPerS = 20;      % 10 spikes per second, on average
input_dist_coef = 0.8;
spikes_N1 = zeros(size(totalStep,2),synapse_per_neuron);
spikes_N2 = zeros(size(totalStep,2),synapse_per_neuron);
post_spikes_N1 = zeros(size(totalStep,2),1);
post_spikes_N2 = zeros(size(totalStep,2),1);
for j = 1:synapse_per_neuron
    current_threshold = sharpRaising01(rand(), input_dist_coef, 50);
    vt = rand(size(totalStep));
    spikes_N1(:,j) = (current_threshold * spikesPerS * del_t) > vt;
    vt = rand(size(totalStep));
    spikes_N2(:,j) = (current_threshold * spikesPerS * del_t) > vt;
end

%%
for t = 1:size(totalStep,2)-1
    %Synapse in N1 & N2
    for j = 1:synapse_per_neuron
        if(spikes_N1(t,j))
            Isyn_N1(t,j) = I_inj*(PR_N1(t,j) > rand);  
        end
        if(spikes_N2(t,j))
            Isyn_N2(t,j) = I_inj*(PR_N2(t,j) > rand); 
        end
    end
    
    %Neuron N1 & N2
    if t == t_sp_N1 || t == t_sp_N1+1
        v_N1(t+1) = 0;
    else
        v_N1(t+1) = v_N1(t)+( -v_N1(t)+Rm*sum(Isyn_N1(t,:)) ) * (del_t/tau_mem);
    end
    if t == t_sp_N2
        v_N2(t+1) = 0;
    else
        v_N2(t+1) = v_N2(t)+( -v_N2(t)+Rm*sum(Isyn_N2(t,:)) ) * (del_t/tau_mem);
    end
    %Neuron thresholds
    if v_N1(t+1) >= vth
        t_sp_N1 = t+1;
        post_spikes_N1(t+1) = 1;
    end
    if v_N2(t+1) >= vth
        t_sp_N2 = t+1;
        post_spikes_N2(t+1) = 1;
    end
    
    %AG of N1 & N2
    if(t == t_sp_N1)
        AG_N1(t+1) = AG_N1(t)+((-AG_N1(t)/tau_AG)+r_AG)*del_t;
    else
        AG_N1(t+1) = AG_N1(t)+(-AG_N1(t)/tau_AG)*del_t;
    end
    if(t == t_sp_N2)
        AG_N2(t+1) = AG_N2(t)+((-AG_N2(t)/tau_AG)+r_AG)*del_t;
    else
        AG_N2(t+1) = AG_N2(t)+(-AG_N2(t)/tau_AG)*del_t;
    end
    
    %DSE of N1 & N2
    DSE_N1(t+1) = AG_N1(t+1)*K_AG;
    DSE_N2(t+1) = AG_N2(t+1)*K_AG;
    
    %-----INSIDE ASTROCYTE-----%
    %IP3
    IP3(t+1) = IP3(t) + (((IP3star-IP3(t))/tau_IP3)+r_IP3*AG_N1(t)+r_IP3*AG_N2(t))*del_t;
    
    Q2(t+1) = d2*((IP3(t+1)+d1) / (IP3(t+1)+d3));
    m_inf(t+1) = IP3(t+1)/(IP3(t+1)+d1);
    
    %Ca2
    Ca2(t+1) = Ca2(t) + (J_chan(t)+J_leak(t)-J_pump(t))*del_t;
    if Ca2_reached_threshold == 0
        if Ca2(t+1) >= Ca_threshold
            t_Ca = t;
            Ca2_reached_threshold = 1;
        end
    end

    h(t+1) = h(t) + ( (h_inf(t)-h(t)) / tau_h(t) )*del_t;
    
    h_inf(t+1) = Q2(t+1)/(Q2(t+1)+Ca2(t+1));
    tau_h(t+1) = 1/(a2*(Q2(t+1)+Ca2(t+1)));
    n_inf(t+1) = Ca2(t+1)/(Ca2(t+1)+d5);
    
    J_chan(t+1) = r_C*((m_inf(t+1)*n_inf(t+1)*h(t+1))^3)*(c0 - (1+c1)*Ca2(t+1));
    J_leak(t+1) = r_L*(c0-(1+c1)*Ca2(t+1));
    J_pump(t+1) = v_ER*(((Ca2(t+1))^2)/(k_ER^2 + (Ca2(t+1))^2));
    
    %Glu
    if(Ca2_reached_threshold)
        if t > t_Ca+300
            Glu(t+1) = Glu(t) + ((-Glu(t)/tau_Glu)+r_Glu)*del_t;
            t_Ca = t;
        else
            Glu(t+1) = Glu(t) + (-Glu(t)/tau_Glu)*del_t;
        end
    end
    
    %eSP
    eSP(t+1)=eSP(t)+((-eSP(t)+m_eSP*Glu(t))/tau_eSP)*del_t;

    PR_N1(t+1,:) = (PR_N1(1,:) + PR_N1(1,:)*(DSE_N1(t+1)+eSP(t+1))/100);
    for j = 1:synapse_per_neuron
        if PR_N1(t+1,j) < 0 
            PR_N1(t+1,j) = 0;
        elseif PR_N1(t+1,j)>1
            PR_N1(t+1,j) = 1;
        end
    end
    
    if t <= (size(totalStep,2)-1)/2
        PR_N2(t+1,:) = (PR_N2(1,:) + PR_N2(1,:)*(DSE_N2(t+1)+eSP(t+1))/100);
        for j = 1:synapse_per_neuron
            if PR_N2(t+1,j) < 0 
                PR_N2(t+1,j) = 0;
            elseif PR_N2(t+1,j) > 1
                PR_N2(t+1,j) = 1;
            end
        end
    else
        PR_N2(t+1,1:num_faulty_synapse) =  0;
        PR_N2(t+1,(num_faulty_synapse+1):synapse_per_neuron) = (PR_N2(1,(num_faulty_synapse+1):synapse_per_neuron) + PR_N2(1,(num_faulty_synapse+1):synapse_per_neuron)*(DSE_N2(t+1)+eSP(t+1))/100);
        for j = 1:synapse_per_neuron
            if PR_N2(t+1,j) < 0 
                PR_N2(t+1,j) = 0;
            elseif PR_N2(t+1,j) > 1
                PR_N2(t+1,j) = 1;
            end
        end
    end
end


%% PR - time
f = figure;
f.Position = [100 100 800 400];
h = axes;
set(h,'position',[.1 .1 .85 .9]);
set(gcf,'color','w');

hold on
for i = 1:synapse_per_neuron
    if i <= num_faulty_synapse
        color = 'r';
    else
        color = 'k';
    end
    p = plot(totalStep,PR_N2(:,i), color,'LineWidth',1);
    if i == 1
        pFaulty = p;
    else
        pHealthy = p;
    end
end
prMax = max(PR_N2(:));
axis([0,400,0,prMax+0.03])
% grid on
% box on

ax = gca;
ax.LineWidth = 2;
daspect([400 prMax*2+0.22 1])

legend([pFaulty, pHealthy],'Faulty synapses','Healthy synapses','Location','southeast')
xlabel('time (s)')
ylabel('PR')

rectangle('Position',[healthy_stable_start*del_t, 0, (healthy_stable_end-healthy_stable_start)*del_t, 1], 'FaceColor',[0.2,0.2,0.2,0.1], 'EdgeColor', [0.2,0.2,0.2,0.1])
rectangle('Position',[longterm_repair_start*del_t, 0, (longterm_repair_end-longterm_repair_start)*del_t, 1], 'FaceColor',[0.2,0.2,0.2,0.1], 'EdgeColor', [0.2,0.2,0.2,0.1])
text(187.5-8, prMax+0.06, 'BF','FontSize',24,'fontname','times')
text(350-8, prMax+0.06, 'AS','FontSize',24,'fontname','times')

set(gca,'fontsize',24, 'fontweight','normal','fontname','times')

