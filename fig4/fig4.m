setenv('TZ', 'America/New_York');
fclose('all');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',50); %get(groot,'factory')
set(groot,'defaultAxesLineWidth',4);
set(groot,'defaultLineLineWidth',4);
set(groot,'defaultLineMarkerSize',50);
set(groot,'defaultErrorbarLineWidth',4);
set(groot,'defaultErrorbarMarkerSize',50);
set(groot,'defaultErrorbarCapSize',20);
set(groot,'defaultAxesView',[0,90]);
set(groot,'defaultAxesBox','on');
set(groot,'defaultTextFontSize',50);
set(groot,'defaultConstantlineLineWidth',4);
set(groot,'defaultConstantlineAlpha',1);
set(groot,'defaultAxesLabelFontSizeMultiplier',1);
set(groot,'defaultFigurePosition',[790 1 1267 1173]);
mainhere = string(datetime('now','Format','M-d-y@HH.mm.ss'))+"fig4";
mkdir(mainhere);

tau = 2.4188843265864e-17; %seconds. tau = hbar/Eh
me = 9.1093837139e-31; %kilograms
u = 1.66053906892e-27; %kilograms
amu = u/me; %1822.9 electron masses, au
EhK = 3.1577502480398e5; %Kelvin

ax = -2.982e-4;
qx = 0.219;
fRF = 2.5e6 * tau; %au
OmegaRF = 2*pi*fRF; %au
w = 1/2*OmegaRF*sqrt(ax+1/2*qx^2); %approximately qx*OmegaRF/2^(3/2);

%m6Li = 6.0151228874 * amu; C4Li = 82.0563;
%m23Na = 22.989767 * amu; C4Na = 81.3512;
%m39K = 38.9637064864 * amu; C4K = 146.4389;
%m87Rb = 86.9091805310 * amu; C4Rb = 319.091/2;

%main figure
figure; hold on;
xlabel("Collision energy, $E=\frac{1}{2}m_a v_0^2$ ($\mu$K)");
ylabel("Probability of complex formation");
xscale("log");

%Li
load("fig4Li.mat");
assert(matom == 6.0151228874 * amu);
semilogx(1/2*matom*vs.^2*EhK*1e6,probabilities);
W0 = 2*(matom/(mion+matom))^(5/3)*(mion^2*w^4*C4/qx^2)^(1/3);
W03D = 4*W0/(3*pi);
xline(W03D*EhK*1e6,"-.","Color",[0.00,0.45,0.74]);

%Na
load("fig4Na.mat");
assert(matom == 22.989767 * amu);
semilogx(1/2*matom*vs.^2*EhK*1e6,probabilities);
W0 = 2*(matom/(mion+matom))^(5/3)*(mion^2*w^4*C4/qx^2)^(1/3);
W03D = 4*W0/(3*pi);
xline(W03D*EhK*1e6,"-.","Color",[0.85,0.33,0.10]);

%K
load("fig4K.mat");
assert(matom == 38.9637064864 * amu);
semilogx(1/2*matom*vs.^2*EhK*1e6,probabilities);
W0 = 2*(matom/(mion+matom))^(5/3)*(mion^2*w^4*C4/qx^2)^(1/3);
W03D = 4*W0/(3*pi);
xline(W03D*EhK*1e6,"-.","Color",[0.93,0.69,0.13]);

%Rb
load("fig4Rb.mat");
assert(matom == 86.9091805310 * amu);
semilogx(1/2*matom*vs.^2*EhK*1e6,probabilities);
W0 = 2*(matom/(mion+matom))^(5/3)*(mion^2*w^4*C4/qx^2)^(1/3);
W03D = 4*W0/(3*pi);
xline(W03D*EhK*1e6,"-.","Color",[0.49,0.18,0.56]);

xticks([0.01,1,100]);
xline(3/2,"k-.");
xlim([1.15e-4,max(1/2*1/(1/matom+1/mion)*vs.^2*EhK*1e6)]);
ylim([0,max(probabilities)]);
legend("Yb$^+$Li","","Yb$^+$Na","","Yb$^+$K","","Yb$^+$Rb","","$T=1\mu$K","Position",[0.2 0.63 0.1 0.2]);
print(gcf,'-vector','-dsvg',mainhere+"/fig4.svg");
hold off;

%inset
set(groot,"defaultFigurePosition",[1430,558,627,648]);
figure; hold on;
xlabel("Collision energy ($\mu$K)");
ylabel("Probability of complex");
xscale("log");

%q = 0.021
load("fig4trapq0.025.mat");
assert(matom == 6.0151228874 * amu);
semilogx(1/2*1/(1/matom+1/mion)*vs.^2*EhK*1e6,probabilities,"Color",[0.47,0.67,0.19]);

%q = 0.042
load("fig4trapq0.050.mat");
assert(matom == 6.0151228874 * amu);
semilogx(1/2*1/(1/matom+1/mion)*vs.^2*EhK*1e6,probabilities,"Color",[0.30,0.75,0.93]);

%q = 0.113
load("fig4trapq0.100.mat");
assert(matom == 6.0151228874 * amu);
semilogx(1/2*1/(1/matom+1/mion)*vs.^2*EhK*1e6,probabilities,"Color",[0.64,0.08,0.18]);

%q = 0.219
load("fig4trapq0.219.mat");
assert(matom == 6.0151228874 * amu);
semilogx(1/2*1/(1/matom+1/mion)*vs.^2*EhK*1e6,probabilities,"Color",[1.00,0.41,0.16]);

xticks([0.01,1]);
xlim([3.5e-4,2.3]);
ylim([0,max(probabilities)]);
legend("$q\approx 0.025$","$q\approx 0.050$","$q\approx 0.100$","$q\approx 0.219$","location","southwest");
print(gcf,'-vector','-dsvg',mainhere+"/fig4inset.svg");
hold off;