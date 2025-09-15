clear all;
setenv('TZ', 'America/New_York');
fclose('all');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',50); %get(groot,'factory')
set(groot,'defaultAxesLineWidth',10);
set(groot,'defaultLineLineWidth',10);
set(groot,'defaultLineMarkerSize',100);
set(groot,'defaultErrorbarLineWidth',10);
set(groot,'defaultErrorbarMarkerSize',100);
set(groot,'defaultAxesView',[0,90]);
set(groot,'defaultAxesBox','on');
set(groot,'defaultTextFontSize',50);
set(groot,'defaultConstantlineLineWidth',10);
set(groot,'defaultFigurePosition',[790 1 1267 1173]);
main = string(datetime('now','Format','M-d-y@HH.mm.ss'))+"code1gpu";
mkdir(main);

a0 = 5.29177210544e-11; %meters
hbar = 1.054571817e-34; %Joules*seconds
Eh = 4.3597447222060e-18; %Joules
tau = 2.4188843265864e-17; %seconds. tau = hbar/Eh
vau = 2.18769126216e6; %meters per second
me = 9.1093837139e-31; %kilograms
u = 1.66053906892e-27; %kilograms
amu = u/me; %1822.9 electron masses, au
kB = 1.380649e-23; %Joules per Kelvin
EhK = 3.1577502480398e5; %Kelvin

tmax = 1000e-6 / tau; %seconds -> au.
b0 = 0; %au
r0 = 5000; %au
umin = 0; umax = 1/2; ntheta = 3163;
phimin = 0; phimax = pi/2; nphi = 3163;
us = linspace(umin,umax,ntheta);
thetas = acos(1-2*us);
phis = linspace(phimin,phimax,nphi);
T = 1e-6; %Kelvin
mion = 170.936323 * amu; %u -> au
matom = 86.9091805310 * amu; %u -> au
De = 1 / EhK; %Kelvin -> Hartree
C4 = 319.091/2; %au
C8 = C4^2/(4*De); %au
v0 = sqrt(3*kB*T/Eh/matom); %au
Re = (2*C8/C4)^(1/4); %au
rbounce = Re; %au
rtol = 1e-10;
atol = 1e-20;

%m6Li = 6.0151228874 * amu; C4Li = 82.0563;
%m23Na = 22.989767 * amu; C4Na = 81.3512;
%m39K = 38.9637064864 * amu; C4K = 146.4389;
%m87Rb = 86.9091805310 * amu; C4Rb = 319.091/2;

%Realistic potential has depth    De = 5102.25 K -> C8 = 104179 au
%Paper's potential has depth De = 30.082470876 K -> C8 = 17669658 au
%First suggested depth is               De = 1 K -> C8 = 531546970 au
%Second suggested depth is             De = 1 mK -> C8 = 531546970412 au
%Third suggested depth is             De = 1 muK -> C8 = 531546970412480 au

ax = -2.982e-4;
ay = ax;
az = -2*ax;
qx = 0.219;
qy = -qx;
qz = 0;
fRF = 2.5e6 * tau; %au
OmegaRF = 2*pi*fRF; %au

% Our trap has a q=0.2, ax=-2.982e-4, fRF=2.5 MHz, r0=6mm, z0=17.5mm, 
% no micromotion yet, C4 = 92.2 au and C4/C6=5e-19. 
% If you need something else, let me know.

fprintf("\n\nr0 = %.20e;\n",r0);
fprintf("v0 = %.20e;\n",v0);
fprintf("tfinal = %.20e;\n",tmax);
fprintf("rtol = %.20e;\n",rtol);
fprintf("atol = %.20e;\n",atol);
fprintf("C4 = %.20e;\n",C4);
fprintf("C8 = %.20e;\n",C8);
fprintf("mion = %.20e;\n",mion);
fprintf("matom = %.20e;\n",matom);
fprintf("ax = %.20e;\n",ax);
fprintf("az = %.20e;\n",az);
fprintf("qx = %.20e;\n",qx);
fprintf("OmegaRF = %.20e;\n",OmegaRF);
fprintf("rbounce = %.20e;\n\n",rbounce);

angle = zeros(ntheta,nphi);
bounces = zeros(ntheta,nphi);
lifetime = zeros(ntheta,nphi);
position = zeros(ntheta,nphi);
transfer = zeros(ntheta,nphi);
dist = zeros(ntheta,nphi);
nsteps = zeros(ntheta,nphi);
KE = zeros(ntheta,nphi);
rthetas = repmat(thetas',1,nphi);
rphis = repmat(phis,ntheta,1);

n = ntheta*nphi;
ng = gpuDeviceCount("available");
fprintf("There are %d GPUs available.\n",ng);
p = parpool(ng);
anglep = cell(ng,1);
bouncesp = cell(ng,1);
lifetimep = cell(ng,1);
positionp = cell(ng,1);
transferp = cell(ng,1);
distp = cell(ng,1);
nstepsp = cell(ng,1);
KEp = cell(ng,1);
rthetasp = cell(ng,1);
rphisp = cell(ng,1);
for i = 1:ng
    rthetasp{i} = rthetas(floor(n/ng*(i-1))+1 : floor(n/ng*i));
    rphisp{i} = rphis(floor(n/ng*(i-1))+1 : floor(n/ng*i));
end

tstart = tic;
parfor i = 1:ng
    [anglep{i}, bouncesp{i}, lifetimep{i}, positionp{i}, transferp{i}, distp{i}, nstepsp{i}, KEp{i}] = ...
    arrayfun(@g, gpuArray(rthetasp{i}), gpuArray(rphisp{i}));
end
tend = toc(tstart);
fprintf("It took %.3f seconds.\n",tend);

for i = 1:ng
    angle(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = gather(anglep{i});
    bounces(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = gather(bouncesp{i});
    lifetime(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = gather(lifetimep{i});
    position(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = gather(positionp{i});
    transfer(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = gather(transferp{i});
    dist(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = gather(distp{i});
    nsteps(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = gather(nstepsp{i});
    KE(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = gather(KEp{i});
end

clear anglep bouncesp lifetimep positionp transferp distp nstepsp KEp rthetasp rphisp
save(main+"/fig1data.mat");



fprintf("The probability of forming a complex is %.20f.\n",length(bounces(bounces>=2))/length(bounces(:)));
fprintf("The probability of leaving the complex is %.20f.\n",length(dist(dist>r0))/length(dist(:)));
[maxnsteps, imax] = max(nsteps(:));
fprintf("The most annoying trajectory was theta=%.20f, phi=%.20f. It took %d timesteps.\n",rthetas(imax),rphis(imax),maxnsteps);
fprintf("The average number of timesteps was %d.\n",round(mean(nsteps(:))));



color = jet(1024);

%angle
figure; hold on;
imagesc(us,phis,angle');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Scattering angle (radians)";
c.TickLabelInterpreter = "Latex";
saveas(gcf,main+'/angle.png');
hold off;
ma = max(angle(:));
mi = min(angle(:));
A = ones(ntheta,nphi,3);
if ma ~= mi
    for i = 1:ntheta
        for j = 1:nphi
            A(i,j,:) = color(round((angle(i,j)-mi)/(ma-mi)*1023)+1,:);
        end
    end
end
imwrite(rot90(A),main+"/zangleHD.png");
figure; hold on;
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
histogram(angle);
xlabel("Scattering angle (radians)");
ylabel("Histogram count");
saveas(gcf,main+'/yangleHist.png');
hold off;



%bounces
figure; hold on;
B = max(log10(bounces),-eps);
imagesc(us,phis,B');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Log10 of number of bounces";
c.TickLabelInterpreter = "Latex";
saveas(gcf,main+'/bounces.png');
hold off;
ma = max(B(:));
mi = min(B(:));
A = ones(ntheta,nphi,3);
if ma ~= mi
    for i = 1:ntheta
        for j = 1:nphi
            A(i,j,:) = color(round((B(i,j)-mi)/(ma-mi)*1023)+1,:);
        end
    end
end
imwrite(rot90(A),main+"/zbouncesHD.png");
figure; hold on;
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
histogram(B);
xlabel("Log10 of number of bounces");
ylabel("Histogram count");
saveas(gcf,main+'/ybouncesHist.png');
hold off;



%lifetime
figure; hold on;
imagesc(us,phis,lifetime');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Complex lifetime (a.u.)";
c.TickLabelInterpreter = "Latex";
saveas(gcf,main+'/lifetime.png');
hold off;
ma = max(lifetime(:));
mi = min(lifetime(:));
A = ones(ntheta,nphi,3);
if ma ~= mi
    for i = 1:ntheta
        for j = 1:nphi
            A(i,j,:) = color(round((lifetime(i,j)-mi)/(ma-mi)*1023)+1,:);
        end
    end
end
imwrite(rot90(A),main+"/zlifetimeHD.png");
figure; hold on;
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
histogram(lifetime);
xlabel("Complex lifetime (a.u.)");
ylabel("Histogram count");
saveas(gcf,main+'/ylifetimeHist.png');
hold off;



%position
figure; hold on;
imagesc(us,phis,position');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Ion distance from origin at moment of collision (a.u.)";
c.TickLabelInterpreter = "Latex";
saveas(gcf,main+'/position.png');
hold off;
ma = max(position(:));
mi = min(position(:));
A = ones(ntheta,nphi,3);
if ma ~= mi
    for i = 1:ntheta
        for j = 1:nphi
            A(i,j,:) = color(round((position(i,j)-mi)/(ma-mi)*1023)+1,:);
        end
    end
end
imwrite(rot90(A),main+"/zpositionHD.png");
figure; hold on;
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
histogram(position);
xlabel("Ion distance from origin at moment of collision (a.u.)");
ylabel("Histogram count");
saveas(gcf,main+'/ypositionHist.png');
hold off;



%transfer
figure; hold on;
B = transfer; %min(1e-3,transfer);
imagesc(us,phis,B');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Momentum transfer, $q=|\vec{p}_{atom,f}-\vec{p}_{atom,i}|$ (a.u.)";
c.TickLabelInterpreter = "Latex";
saveas(gcf,main+'/transfer.png');
hold off;
ma = max(B(:));
mi = min(B(:));
A = ones(ntheta,nphi,3);
if ma ~= mi
    for i = 1:ntheta
        for j = 1:nphi
            A(i,j,:) = color(round((B(i,j)-mi)/(ma-mi)*1023)+1,:);
        end
    end
end
imwrite(rot90(A),main+"/ztransferHD.png");
figure; hold on;
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
histogram(transfer);
xlabel(["Momentum transfer", "$q=|\vec{p}_{atom,f}-\vec{p}_{atom,i}|$ (a.u.)"]);
ylabel("Histogram count");
saveas(gcf,main+'/ytransferHist.png');
hold off;



%dist
figure; hold on;
imagesc(us,phis,dist');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Final atom-ion distance (a.u.)";
c.TickLabelInterpreter = "Latex";
saveas(gcf,main+'/dist.png');
hold off;
ma = max(dist(:));
mi = min(dist(:));
A = ones(ntheta,nphi,3);
if ma ~= mi
    for i = 1:ntheta
        for j = 1:nphi
            A(i,j,:) = color(round((dist(i,j)-mi)/(ma-mi)*1023)+1,:);
        end
    end
end
imwrite(rot90(A),main+"/zdistHD.png");
figure; hold on;
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
histogram(dist);
xlabel("Final atom-ion distance (a.u.)");
ylabel("Histogram count");
saveas(gcf,main+'/ydistHist.png');
hold off;



%nsteps
figure; hold on;
imagesc(us,phis,nsteps');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Number of timesteps";
c.TickLabelInterpreter = "Latex";
saveas(gcf,main+'/nsteps.png');
hold off;
ma = max(nsteps(:));
mi = min(nsteps(:));
A = ones(ntheta,nphi,3);
if ma ~= mi
    for i = 1:ntheta
        for j = 1:nphi
            A(i,j,:) = color(round((nsteps(i,j)-mi)/(ma-mi)*1023)+1,:);
        end
    end
end
imwrite(rot90(A),main+"/znstepsHD.png");
figure; hold on;
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
histogram(nsteps);
xlabel("Number of timesteps");
ylabel("Histogram count");
saveas(gcf,main+'/ynstepsHist.png');
hold off;



%KE
figure; hold on;
imagesc(us,phis,KE');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Final ion kinetic energy";
c.TickLabelInterpreter = "Latex";
saveas(gcf,main+'/KE.png');
hold off;
ma = max(KE(:));
mi = min(KE(:));
A = ones(ntheta,nphi,3);
if ma ~= mi
    for i = 1:ntheta
        for j = 1:nphi
            A(i,j,:) = color(round((KE(i,j)-mi)/(ma-mi)*1023)+1,:);
        end
    end
end
imwrite(rot90(A),main+"/zKEHD.png");
figure; hold on;
set(gcf,'Position', [790 1 1267 1173]); %get(0, 'Screensize')
histogram(KE);
xlabel("Final ion kinetic energy");
ylabel("Histogram count");
saveas(gcf,main+'/yKEHist.png');
hold off;



%wherever it says ode, compute the f(t,y) function
function [angle,bounces,lifetime,position,transfer,dist,nsteps,KE] = g(theta0,phi0)

r0 = 5.00000000000000000000e+03;
v0 = 7.74388437761750933038e-09;
tfinal = 4.13413733351701562500e+13;
rtol = 1.00000000000000003643e-10;
atol = 9.99999999999999945153e-21;
C4 = 1.59545500000000004093e+02;
C8 = 2.00949488627505397797e+09;
mion = 3.11597855084210925270e+05;
matom = 1.58425744542229716899e+05;
ax = -2.98199999999999984177e-04;
az = 5.96399999999999968353e-04;
qx = 2.19000000000000000222e-01;
OmegaRF = 3.79957461514366392655e-10;
rbounce = 7.08448174250339519631e+01;

    y07 = r0*sin(theta0)*cos(phi0);
    y08 = r0*sin(theta0)*sin(phi0);
    y09 = r0*cos(theta0);
    y010 = -v0*sin(theta0)*cos(phi0);
    y011 = -v0*sin(theta0)*sin(phi0);
    y012 = -v0*cos(theta0);
    bounces = 0;
    bouncing = false;
    tfirst = 0;
    tlast = 0;
    tmin = 0;
    rmin = r0;
    lifetime = 0;
    pmin = r0;
    position = 0;

    nsteps  = 0;
    f01 = 0; f02 = 0; f03 = 0; %ode start
    f07 = y010; f08 = y011; f09 = y012;
    r = sqrt(y07^2+y08^2+y09^2);
    dVdr = 4*C4/r^5 - 8*C8/r^9;
    f04 = -1/mion*dVdr*(-y07)/r;
    f05 = -1/mion*dVdr*(-y08)/r;
    f06 = -1/mion*dVdr*(-y09)/r;
    f010 = -1/matom*dVdr*(y07)/r;
    f011 = -1/matom*dVdr*(y08)/r;
    f012 = -1/matom*dVdr*(y09)/r; %odeend
    threshold = atol / rtol;
    hmax = min(tfinal,max(0.1*tfinal,16.0*eps*tfinal));
    t = 0;
    y1 = 0; y2 = 0; y3 = 0; y4 = 0; y5 = 0; y6 = 0;
    y7 = y07; y8 = y08; y9 = y09; y10 = y010; y11 = y011; y12 = y012;
    pow = 1/5;
    a2=1/5;
    a3=3/10;
    a4=4/5;
    a5=8/9;
    b11=1/5;
    b21=3/40;
    b31=44/45;
    b41=19372/6561;
    b51=9017/3168;
    b61=35/384;
    b22=9/40;
    b32=-56/15;
    b42=-25360/2187;
    b52=-355/33;
    b33=32/9;
    b43=64448/6561;
    b53=46732/5247;
    b63=500/1113;
    b44=-212/729;
    b54=49/176;
    b64=125/192;
    b55=-5103/18656;
    b65=-2187/6784;
    b66=11/84;
    e1=71/57600;
    e3=-71/16695;
    e4=71/1920;
    e5=-17253/339200;
    e6=22/525;
    e7=-1/40;
    hmin = 16*eps(t);
    absh = min(hmax, tfinal);
    maxabs = abs(f01 / threshold);
    maxabs = max(maxabs, abs(f02 / threshold));
    maxabs = max(maxabs, abs(f03 / threshold));
    maxabs = max(maxabs, abs(f04 / threshold));
    maxabs = max(maxabs, abs(f05 / threshold));
    maxabs = max(maxabs, abs(f06 / threshold));
    maxabs = max(maxabs, abs(f07 / max(abs(y07),threshold)));
    maxabs = max(maxabs, abs(f08 / max(abs(y08),threshold)));
    maxabs = max(maxabs, abs(f09 / max(abs(y09),threshold)));
    maxabs = max(maxabs, abs(f010 / max(abs(y010),threshold)));
    maxabs = max(maxabs, abs(f011 / max(abs(y011),threshold)));
    maxabs = max(maxabs, abs(f012 / max(abs(y012),threshold)));
    rh = maxabs / (0.8 * rtol^pow);
    if absh * rh > 1
        absh = 1 / rh;
    end
    absh = max(absh, hmin);
    f11 = f01; f12 = f02; f13 = f03; f14 = f04; f15 = f05; f16 = f06;
    f17 = f07; f18 = f08; f19 = f09; f110 = f010; f111 = f011; f112 = f012;
    done = false;
    err = eps; %uninit
    tnew = t; %uninit
    ynew1 = y1; ynew2 = y2; ynew3 = y3; ynew4 = y4; ynew5 = y5; ynew6 = y6; %uninit
    ynew7 = y7; ynew8 = y8; ynew9 = y9; ynew10 = y10; ynew11 = y11; ynew12 = y12; %uninit
    f71 = f11; f72 = f12; f73 = f13; f74 = f14; f75 = f15; f76 = f16; %uninit
    f77 = f17; f78 = f18; f79 = f19; f710 = f110; f711 = f111; f712 = f112; %uninit
    while ~done
        hmin = 16*eps(t);
        absh = min(hmax, max(hmin, absh));    % couldn't limit absh until new hmin
        h = absh;
        if 1.1*absh >= abs(tfinal - t)
            h = tfinal - t;
            absh = abs(h);
            done = true;
        end
        nofailed = true;                      % no failed attempts
        while true
            y21 = y1 + h * (b11*f11 );
            y22 = y2 + h * (b11*f12 );
            y23 = y3 + h * (b11*f13 );
            y24 = y4 + h * (b11*f14 );
            y25 = y5 + h * (b11*f15 );
            y26 = y6 + h * (b11*f16 );
            y27 = y7 + h * (b11*f17 );
            y28 = y8 + h * (b11*f18 );
            y29 = y9 + h * (b11*f19 );
            y210 = y10 + h * (b11*f110 );
            y211 = y11 + h * (b11*f111 );
            y212 = y12 + h * (b11*f112 );
            t2 = t + h * a2;
            f21 = y24; f22 = y25; f23 = y26; %ode start
            f27 = y210; f28 = y211; f29 = y212;
            r = sqrt((y27-y21)^2+(y28-y22)^2+(y29-y23)^2);
            dVdr = 4*C4/r^5 - 8*C8/r^9;
            f24 = -1/mion*dVdr*(y21-y27)/r;
            f25 = -1/mion*dVdr*(y22-y28)/r;
            f26 = -1/mion*dVdr*(y23-y29)/r;
            f210 = -1/matom*dVdr*(y27-y21)/r;
            f211 = -1/matom*dVdr*(y28-y22)/r;
            f212 = -1/matom*dVdr*(y29-y23)/r;
            f24 = f24 - (ax+2*qx*cos(OmegaRF*t2))*OmegaRF^2/4*y21;
            f25 = f25 - (ax+2*-qx*cos(OmegaRF*t2))*OmegaRF^2/4*y22;
            f26 = f26 - (az)*OmegaRF^2/4*y23; %odeend
            y21 = y1 + h * (b21*f11 + b22*f21 );
            y22 = y2 + h * (b21*f12 + b22*f22 );
            y23 = y3 + h * (b21*f13 + b22*f23 );
            y24 = y4 + h * (b21*f14 + b22*f24 );
            y25 = y5 + h * (b21*f15 + b22*f25 );
            y26 = y6 + h * (b21*f16 + b22*f26 );
            y27 = y7 + h * (b21*f17 + b22*f27 );
            y28 = y8 + h * (b21*f18 + b22*f28 );
            y29 = y9 + h * (b21*f19 + b22*f29 );
            y210 = y10 + h * (b21*f110 + b22*f210 );
            y211 = y11 + h * (b21*f111 + b22*f211 );
            y212 = y12 + h * (b21*f112 + b22*f212 );
            t2 = t + h * a3;
            f31 = y24; f32 = y25; f33 = y26; %ode start
            f37 = y210; f38 = y211; f39 = y212;
            r = sqrt((y27-y21)^2+(y28-y22)^2+(y29-y23)^2);
            dVdr = 4*C4/r^5 - 8*C8/r^9;
            f34 = -1/mion*dVdr*(y21-y27)/r;
            f35 = -1/mion*dVdr*(y22-y28)/r;
            f36 = -1/mion*dVdr*(y23-y29)/r;
            f310 = -1/matom*dVdr*(y27-y21)/r;
            f311 = -1/matom*dVdr*(y28-y22)/r;
            f312 = -1/matom*dVdr*(y29-y23)/r;
            f34 = f34 - (ax+2*qx*cos(OmegaRF*t2))*OmegaRF^2/4*y21;
            f35 = f35 - (ax+2*-qx*cos(OmegaRF*t2))*OmegaRF^2/4*y22;
            f36 = f36 - (az)*OmegaRF^2/4*y23; %odeend
            y21 = y1 + h * (b31*f11 + b32*f21 + b33*f31 );
            y22 = y2 + h * (b31*f12 + b32*f22 + b33*f32 );
            y23 = y3 + h * (b31*f13 + b32*f23 + b33*f33 );
            y24 = y4 + h * (b31*f14 + b32*f24 + b33*f34 );
            y25 = y5 + h * (b31*f15 + b32*f25 + b33*f35 );
            y26 = y6 + h * (b31*f16 + b32*f26 + b33*f36 );
            y27 = y7 + h * (b31*f17 + b32*f27 + b33*f37 );
            y28 = y8 + h * (b31*f18 + b32*f28 + b33*f38 );
            y29 = y9 + h * (b31*f19 + b32*f29 + b33*f39 );
            y210 = y10 + h * (b31*f110 + b32*f210 + b33*f310 );
            y211 = y11 + h * (b31*f111 + b32*f211 + b33*f311 );
            y212 = y12 + h * (b31*f112 + b32*f212 + b33*f312 );
            t2 = t + h * a4;
            f41 = y24; f42 = y25; f43 = y26; %ode start
            f47 = y210; f48 = y211; f49 = y212;
            r = sqrt((y27-y21)^2+(y28-y22)^2+(y29-y23)^2);
            dVdr = 4*C4/r^5 - 8*C8/r^9;
            f44 = -1/mion*dVdr*(y21-y27)/r;
            f45 = -1/mion*dVdr*(y22-y28)/r;
            f46 = -1/mion*dVdr*(y23-y29)/r;
            f410 = -1/matom*dVdr*(y27-y21)/r;
            f411 = -1/matom*dVdr*(y28-y22)/r;
            f412 = -1/matom*dVdr*(y29-y23)/r;
            f44 = f44 - (ax+2*qx*cos(OmegaRF*t2))*OmegaRF^2/4*y21;
            f45 = f45 - (ax+2*-qx*cos(OmegaRF*t2))*OmegaRF^2/4*y22;
            f46 = f46 - (az)*OmegaRF^2/4*y23; %odeend
            y21 = y1 + h * (b41*f11 + b42*f21 + b43*f31 + b44*f41 );
            y22 = y2 + h * (b41*f12 + b42*f22 + b43*f32 + b44*f42 );
            y23 = y3 + h * (b41*f13 + b42*f23 + b43*f33 + b44*f43 );
            y24 = y4 + h * (b41*f14 + b42*f24 + b43*f34 + b44*f44 );
            y25 = y5 + h * (b41*f15 + b42*f25 + b43*f35 + b44*f45 );
            y26 = y6 + h * (b41*f16 + b42*f26 + b43*f36 + b44*f46 );
            y27 = y7 + h * (b41*f17 + b42*f27 + b43*f37 + b44*f47 );
            y28 = y8 + h * (b41*f18 + b42*f28 + b43*f38 + b44*f48 );
            y29 = y9 + h * (b41*f19 + b42*f29 + b43*f39 + b44*f49 );
            y210 = y10 + h * (b41*f110 + b42*f210 + b43*f310 + b44*f410 );
            y211 = y11 + h * (b41*f111 + b42*f211 + b43*f311 + b44*f411 );
            y212 = y12 + h * (b41*f112 + b42*f212 + b43*f312 + b44*f412 );
            t2 = t + h * a5;
            f51 = y24; f52 = y25; f53 = y26; %ode start
            f57 = y210; f58 = y211; f59 = y212;
            r = sqrt((y27-y21)^2+(y28-y22)^2+(y29-y23)^2);
            dVdr = 4*C4/r^5 - 8*C8/r^9;
            f54 = -1/mion*dVdr*(y21-y27)/r;
            f55 = -1/mion*dVdr*(y22-y28)/r;
            f56 = -1/mion*dVdr*(y23-y29)/r;
            f510 = -1/matom*dVdr*(y27-y21)/r;
            f511 = -1/matom*dVdr*(y28-y22)/r;
            f512 = -1/matom*dVdr*(y29-y23)/r;
            f54 = f54 - (ax+2*qx*cos(OmegaRF*t2))*OmegaRF^2/4*y21;
            f55 = f55 - (ax+2*-qx*cos(OmegaRF*t2))*OmegaRF^2/4*y22;
            f56 = f56 - (az)*OmegaRF^2/4*y23; %odeend
            y21 = y1 + h * (b51*f11 + b52*f21 + b53*f31 + b54*f41 + b55*f51 );
            y22 = y2 + h * (b51*f12 + b52*f22 + b53*f32 + b54*f42 + b55*f52 );
            y23 = y3 + h * (b51*f13 + b52*f23 + b53*f33 + b54*f43 + b55*f53 );
            y24 = y4 + h * (b51*f14 + b52*f24 + b53*f34 + b54*f44 + b55*f54 );
            y25 = y5 + h * (b51*f15 + b52*f25 + b53*f35 + b54*f45 + b55*f55 );
            y26 = y6 + h * (b51*f16 + b52*f26 + b53*f36 + b54*f46 + b55*f56 );
            y27 = y7 + h * (b51*f17 + b52*f27 + b53*f37 + b54*f47 + b55*f57 );
            y28 = y8 + h * (b51*f18 + b52*f28 + b53*f38 + b54*f48 + b55*f58 );
            y29 = y9 + h * (b51*f19 + b52*f29 + b53*f39 + b54*f49 + b55*f59 );
            y210 = y10 + h * (b51*f110 + b52*f210 + b53*f310 + b54*f410 + b55*f510 );
            y211 = y11 + h * (b51*f111 + b52*f211 + b53*f311 + b54*f411 + b55*f511 );
            y212 = y12 + h * (b51*f112 + b52*f212 + b53*f312 + b54*f412 + b55*f512 );
            t2 = t + h;
            f61 = y24; f62 = y25; f63 = y26; %ode start
            f67 = y210; f68 = y211; f69 = y212;
            r = sqrt((y27-y21)^2+(y28-y22)^2+(y29-y23)^2);
            dVdr = 4*C4/r^5 - 8*C8/r^9;
            f64 = -1/mion*dVdr*(y21-y27)/r;
            f65 = -1/mion*dVdr*(y22-y28)/r;
            f66 = -1/mion*dVdr*(y23-y29)/r;
            f610 = -1/matom*dVdr*(y27-y21)/r;
            f611 = -1/matom*dVdr*(y28-y22)/r;
            f612 = -1/matom*dVdr*(y29-y23)/r;
            f64 = f64 - (ax+2*qx*cos(OmegaRF*t2))*OmegaRF^2/4*y21;
            f65 = f65 - (ax+2*-qx*cos(OmegaRF*t2))*OmegaRF^2/4*y22;
            f66 = f66 - (az)*OmegaRF^2/4*y23; %odeend
            tnew = t + h;
            if done
                tnew = tfinal;   % Hit end point exactly.
            end
            h = tnew - t;      % Purify h.
            ynew1 = y1 + h* ( b61*f11 + b63*f31 + b64*f41 + b65*f51 + b66*f61 );
            ynew2 = y2 + h* ( b61*f12 + b63*f32 + b64*f42 + b65*f52 + b66*f62 );
            ynew3 = y3 + h* ( b61*f13 + b63*f33 + b64*f43 + b65*f53 + b66*f63 );
            ynew4 = y4 + h* ( b61*f14 + b63*f34 + b64*f44 + b65*f54 + b66*f64 );
            ynew5 = y5 + h* ( b61*f15 + b63*f35 + b64*f45 + b65*f55 + b66*f65 );
            ynew6 = y6 + h* ( b61*f16 + b63*f36 + b64*f46 + b65*f56 + b66*f66 );
            ynew7 = y7 + h* ( b61*f17 + b63*f37 + b64*f47 + b65*f57 + b66*f67 );
            ynew8 = y8 + h* ( b61*f18 + b63*f38 + b64*f48 + b65*f58 + b66*f68 );
            ynew9 = y9 + h* ( b61*f19 + b63*f39 + b64*f49 + b65*f59 + b66*f69 );
            ynew10 = y10 + h* ( b61*f110 + b63*f310 + b64*f410 + b65*f510 + b66*f610 );
            ynew11 = y11 + h* ( b61*f111 + b63*f311 + b64*f411 + b65*f511 + b66*f611 );
            ynew12 = y12 + h* ( b61*f112 + b63*f312 + b64*f412 + b65*f512 + b66*f612 );
            f71 = ynew4; f72 = ynew5; f73 = ynew6; %ode start
            f77 = ynew10; f78 = ynew11; f79 = ynew12;
            r = sqrt((ynew7-ynew1)^2+(ynew8-ynew2)^2+(ynew9-ynew3)^2);
            dVdr = 4*C4/r^5 - 8*C8/r^9;
            f74 = -1/mion*dVdr*(ynew1-ynew7)/r;
            f75 = -1/mion*dVdr*(ynew2-ynew8)/r;
            f76 = -1/mion*dVdr*(ynew3-ynew9)/r;
            f710 = -1/matom*dVdr*(ynew7-ynew1)/r;
            f711 = -1/matom*dVdr*(ynew8-ynew2)/r;
            f712 = -1/matom*dVdr*(ynew9-ynew3)/r;
            f74 = f74 - (ax+2*qx*cos(OmegaRF*tnew))*OmegaRF^2/4*ynew1;
            f75 = f75 - (ax+2*-qx*cos(OmegaRF*tnew))*OmegaRF^2/4*ynew2;
            f76 = f76 - (az)*OmegaRF^2/4*ynew3; %odeend
            fE1 = f11*e1 + f31*e3 + f41*e4 + f51*e5 + f61*e6 + f71*e7;
            fE2 = f12*e1 + f32*e3 + f42*e4 + f52*e5 + f62*e6 + f72*e7;
            fE3 = f13*e1 + f33*e3 + f43*e4 + f53*e5 + f63*e6 + f73*e7;
            fE4 = f14*e1 + f34*e3 + f44*e4 + f54*e5 + f64*e6 + f74*e7;
            fE5 = f15*e1 + f35*e3 + f45*e4 + f55*e5 + f65*e6 + f75*e7;
            fE6 = f16*e1 + f36*e3 + f46*e4 + f56*e5 + f66*e6 + f76*e7;
            fE7 = f17*e1 + f37*e3 + f47*e4 + f57*e5 + f67*e6 + f77*e7;
            fE8 = f18*e1 + f38*e3 + f48*e4 + f58*e5 + f68*e6 + f78*e7;
            fE9 = f19*e1 + f39*e3 + f49*e4 + f59*e5 + f69*e6 + f79*e7;
            fE10 = f110*e1 + f310*e3 + f410*e4 + f510*e5 + f610*e6 + f710*e7;
            fE11 = f111*e1 + f311*e3 + f411*e4 + f511*e5 + f611*e6 + f711*e7;
            fE12 = f112*e1 + f312*e3 + f412*e4 + f512*e5 + f612*e6 + f712*e7;
            maxabs =             abs(fE1 / max(max(abs(y1),abs(ynew1)),threshold));
            maxabs = max(maxabs, abs(fE2 / max(max(abs(y2),abs(ynew2)),threshold)));
            maxabs = max(maxabs, abs(fE3 / max(max(abs(y3),abs(ynew3)),threshold)));
            maxabs = max(maxabs, abs(fE4 / max(max(abs(y4),abs(ynew4)),threshold)));
            maxabs = max(maxabs, abs(fE5 / max(max(abs(y5),abs(ynew5)),threshold)));
            maxabs = max(maxabs, abs(fE6 / max(max(abs(y6),abs(ynew6)),threshold)));
            maxabs = max(maxabs, abs(fE7 / max(max(abs(y7),abs(ynew7)),threshold)));
            maxabs = max(maxabs, abs(fE8 / max(max(abs(y8),abs(ynew8)),threshold)));
            maxabs = max(maxabs, abs(fE9 / max(max(abs(y9),abs(ynew9)),threshold)));
            maxabs = max(maxabs, abs(fE10 / max(max(abs(y10),abs(ynew10)),threshold)));
            maxabs = max(maxabs, abs(fE11 / max(max(abs(y11),abs(ynew11)),threshold)));
            maxabs = max(maxabs, abs(fE12 / max(max(abs(y12),abs(ynew12)),threshold)));
            err = absh * maxabs;
            if err > rtol                       % Failed step
                if nofailed
                    nofailed = false;
                    absh = max(hmin, absh * max(0.1, 0.8*(rtol/err)^pow));
                else
                    absh = max(hmin, 0.5 * absh);
                end
                h = absh;
                done = false;
            else                                % Successful step
                break;
            end
        end
        nsteps = nsteps + 1;
        if nsteps >= 1000000
            done = true;
        end
        if done
            break;
        end
        if nofailed
            temp = 1.25*(err/rtol)^pow;
            if temp > 0.2
                absh = absh / temp;
            else
                absh = 5.0*absh;
            end
        end
        t = tnew;
        y1 = ynew1; y2 = ynew2; y3 = ynew3; y4 = ynew4; y5 = ynew5; y6 = ynew6;
        y7 = ynew7; y8 = ynew8; y9 = ynew9; y10 = ynew10; y11 = ynew11; y12 = ynew12;
        f11 = f71; f12 = f72; f13 = f73; f14 = f74; f15 = f75; f16 = f76; % Already have f(tnew,ynew)
        f17 = f77; f18 = f78; f19 = f79; f110 = f710; f111 = f711; f112 = f712;
        if bouncing
            if r < rmin
                tmin = t;
                rmin = r;
                pmin = sqrt(y1^2+y2^2+y3^2);
            end
            if r >= rbounce
                bouncing = false;
                bounces = bounces + 1;
                if bounces == 1
                    tfirst = tmin;
                    position = pmin;
                else
                    tlast = tmin;
                end
                rmin = r0;
            end
        elseif ~bouncing 
            if r < rbounce
                bouncing = true;
            end
        end
    end % end of while loop
    adotb = y010*ynew10+y011*ynew11+y012*ynew12;
    norma = sqrt(y010^2+y011^2+y012^2);
    normb = sqrt(ynew10^2+ynew11^2+ynew12^2);
    angle = real(acos(complex(adotb/(norma*normb))));
    if bounces >= 2
        lifetime = tlast - tfirst;
    end
    transfer = matom * sqrt((y010-ynew10)^2+(y011-ynew11)^2+(y012-ynew12)^2);
    dist = r;
    KE = 1/2 * mion * (ynew4^2 + ynew5^2 + ynew6^2);
end
