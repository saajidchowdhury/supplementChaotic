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
main = string(datetime('now','Format','M-d-y@HH.mm.ss'))+"code1cpu";
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
umin = 0; umax = 1/2; ntheta = 10;
phimin = 0; phimax = pi/2; nphi = 10;
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
ng = 10;
p = gcp('nocreate');
if isempty(p)
    p = parpool(ng);
elseif p.NumWorkers ~= ng
    delete(p);
    p = parpool(ng);
end
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
    arrayfun(@(theta,phi)h(C4,C8,mion,matom,ax,ay,az,qx,qy,qz,OmegaRF,[0,tmax], ...
    [0,0,0,0,0,0,r0*sin(theta)*cos(phi),r0*sin(theta)*sin(phi),r0*cos(theta), ...
    -v0*sin(theta)*cos(phi),-v0*sin(theta)*sin(phi),-v0*cos(theta)],r0,rtol,atol), rthetasp{i}, rphisp{i});
end
tend = toc(tstart);
fprintf("It took %.3f seconds.\n",tend);

for i = 1:ng
    angle(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = (anglep{i});
    bounces(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = (bouncesp{i});
    lifetime(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = (lifetimep{i});
    position(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = (positionp{i});
    transfer(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = (transferp{i});
    dist(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = (distp{i});
    nsteps(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = (nstepsp{i});
    KE(floor(n/ng*(i-1))+1 : floor(n/ng*i)) = (KEp{i});
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

function [angle,bounces,lifetime,position,transfer,dist,nsteps,KE] = h(C4,C8,mion,matom,ax,ay,az,qx,qy,qz,OmegaRF,tspan,y0,r0,rtol,atol)
    options = odeset('Stats','off','RelTol',rtol,'AbsTol',atol,'OutputFcn',@halt);
    [tlist,y] = ode45(@(t,y)f(C4,C8,mion,matom,ax,ay,az,qx,qy,qz,OmegaRF,t,y), tspan, y0, options);
    rbounce = (2*C8/C4)^(1/4);
    bounces = 0;
    bouncing = false;
    tfirst = 0;
    tlast = 0;
    tmin = 0;
    rmin = r0;
    lifetime = 0;
    pmin = r0;
    position = 0;
    nsteps = length(tlist);
    r = r0;
    for i = 1:length(tlist)
        t = tlist(i);
        r = sqrt((y(i,1)-y(i,7))^2+(y(i,2)-y(i,8))^2+(y(i,3)-y(i,9))^2);
        if bouncing
            if r < rmin
                tmin = t;
                rmin = r;
                pmin = sqrt(y(i,1)^2+y(i,2)^2+y(i,3)^2);
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
    end
    adotb = y0(10)*y(end,10) + y0(11)*y(end,11) + y0(12)*y(end,12);
    norma = sqrt(y0(10)^2+y0(11)^2+y0(12)^2);
    normb = sqrt(y(end,10)^2+y(end,11)^2+y(end,12)^2);
    angle = real(acos(complex(adotb/(norma*normb))));
    if bounces >= 2
        lifetime = tlast-tfirst;
    end
    transfer = matom * sqrt((y0(10)-y(end,10))^2+(y0(11)-y(end,11))^2+(y0(12)-y(end,12))^2);
    dist = r;
    KE = 1/2 * mion * (y(end,4)^2+y(end,5)^2+y(end,6)^2);
end

function status = halt(t,~,flag)
    status = 0;
    persistent nstep
    switch flag
        case 'init'
            nstep = 0;
        case []
            nstep = nstep + length(t);
            if nstep >= 4000000
                status = 1;
            end
    end
end

function f = f(C4,C8,mion,matom,ax,ay,az,qx,qy,qz,OmegaRF,t,y)
    %y(1:3) = position of ion
    %y(4:6) = velocity of ion
    %y(7:9) = position of atom
    %y(10:12) = velocity of atom
    f = zeros(12,1);
    f(1:3) = y(4:6);
    f(7:9) = y(10:12);
    r = sqrt((y(1)-y(7))^2+(y(2)-y(8))^2+(y(3)-y(9))^2);
    dVdr = 4*C4/r^5 - 8*C8/r^9;
    f(4:6) = -1/mion*dVdr*(y(1:3)-y(7:9))/r;
    f(10:12) = -1/matom*dVdr*(y(7:9)-y(1:3))/r;
    f(4) = f(4) - (ax+2*qx*cos(OmegaRF*t))*OmegaRF^2/4*y(1);
    f(5) = f(5) - (ay+2*qy*cos(OmegaRF*t))*OmegaRF^2/4*y(2);
    f(6) = f(6) - (az+2*qz*cos(OmegaRF*t))*OmegaRF^2/4*y(3);
end