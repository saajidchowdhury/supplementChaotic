setenv('TZ', 'America/New_York');
fclose('all');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',80); %found using get(groot,'factory')
set(groot,'defaultAxesLineWidth',10);
set(groot,'defaultLineLineWidth',10);
set(groot,'defaultLineMarkerSize',100);
set(groot,'defaultErrorbarLineWidth',10);
set(groot,'defaultErrorbarMarkerSize',100);
set(groot,'defaultErrorbarCapSize',20);
set(groot,'defaultAxesView',[0,90]);
set(groot,'defaultAxesBox','on');
set(groot,'defaultTextFontSize',50);
set(groot,'defaultConstantlineLineWidth',10);
set(groot,'defaultConstantlineAlpha',1);
set(groot,'defaultAxesLabelFontSizeMultiplier',1);
set(groot,'defaultFigurePosition',[790 1 1267 1173]);
mainhere = string(datetime('now','Format','M-d-y@HH.mm.ss'))+"fig1";
mkdir(mainhere);

load("fig1data.mat");

color = jet(1024);

%angle
figure; hold on;
A = angle;
imagesc(us,phis,A');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Scattering angle (radians)";
c.TickLabelInterpreter = "Latex";
print(gcf,'-vector','-dsvg',mainhere+"/angle.svg");
hold off;

%bounces
figure; hold on;
B = log10(bounces);
imagesc(us,phis,B');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Log$_{10}$ (number of bounces)";
c.TickLabelInterpreter = "Latex";
print(gcf,'-vector','-dsvg',mainhere+"/bounces.svg");
hold off;

%momentum transfer
figure; hold on;
Q = min(transfer,15e-3);
imagesc(us,phis,Q');
axis([umin,umax,phimin,phimax]);
set(gca,'YDir','normal');
xlabel("$\theta$ (radians)");
ylabel("$\phi$ (radians)");
xticks(0.5*(1-cos([0 0.5 1 1.5])));
xticklabels({'0','0.5','1','1.5'});
colormap(jet);
c = colorbar;
c.Label.Interpreter = "Latex";
c.Label.String = "Momentum transfer, $q$ (a.u.)";
c.TickLabelInterpreter = "Latex";
print(gcf,'-vector','-dsvg',mainhere+"/transfer.svg");
hold off;

%sphere
figure; hold on;
thetas2 = thetas(1:2:end);
phis2 = phis(1:2:end);
angle2 = angle(1:2:end,1:2:end);
X = sin(thetas2)'*cos(phis2);
Y = sin(thetas2)'*sin(phis2);
Z = cos(thetas2)'*ones(size(phis2));
c = colormap(jet);
surf(X,Y,Z,angle2,"EdgeColor","none",'FaceAlpha',0.2);
surf(X,Y,-Z,angle2,"EdgeColor","none",'FaceAlpha',1);
surf(X,-Y,Z,angle2,"EdgeColor","none",'FaceAlpha',1);
surf(X,-Y,-Z,angle2,"EdgeColor","none",'FaceAlpha',1);
surf(-X,Y,Z,angle2,"EdgeColor","none",'FaceAlpha',1);
surf(-X,Y,-Z,angle2,"EdgeColor","none",'FaceAlpha',1);
surf(-X,-Y,Z,angle2,"EdgeColor","none",'FaceAlpha',1);
surf(-X,-Y,-Z,angle2,"EdgeColor","none",'FaceAlpha',1);
surf(X*0.99,Y*0.99,-Z*0.99,angle2,"EdgeColor","white",'FaceAlpha',1);
surf(X*0.99,-Y*0.99,Z*0.99,angle2,"EdgeColor","white",'FaceAlpha',1);
surf(X*0.99,-Y*0.99,-Z*0.99,angle2,"EdgeColor","white",'FaceAlpha',1);
surf(-X*0.99,Y*0.99,Z*0.99,angle2,"EdgeColor","white",'FaceAlpha',1);
surf(-X*0.99,Y*0.99,-Z*0.99,angle2,"EdgeColor","white",'FaceAlpha',1);
surf(-X*0.99,-Y*0.99,Z*0.99,angle2,"EdgeColor","white",'FaceAlpha',1);
surf(-X*0.99,-Y*0.99,-Z*0.99,angle2,"EdgeColor","white",'FaceAlpha',1);
plot3(sin(linspace(0,pi/2,100))*cos(0),sin(linspace(0,pi/2,100))*sin(0),cos(linspace(0,pi/2,100)),"Color","black","LineWidth",4);
plot3(sin(pi/2)*cos(linspace(0,pi/2,100)),sin(pi/2)*sin(linspace(0,pi/2,100)),cos(pi/2)*ones(1,100),"Color","black","LineWidth",4);
plot3(sin(linspace(0,pi/2,100))*cos(pi/2),sin(linspace(0,pi/2,100))*sin(pi/2),cos(linspace(0,pi/2,100)),"Color","black","LineWidth",4);
view(104.3706,11.641);
box off;
axis off;
saveas(gcf,mainhere+"/sphere.png"); %This causes the octant to appear dark.
% So, save it as png using the MATLAB gui while the figure is open.
hold off;