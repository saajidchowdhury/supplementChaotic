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
mainhere = string(datetime('now','Format','M-d-y@HH.mm.ss'))+"fig2revised";
mkdir(mainhere);

figure; hold on;

%Na
load("fig2Na.mat");
lifetimes = lifetime(dist>r0 & lifetime>0);
x = sort(lifetimes(:));
nt = length(lifetimes(:));
y = zeros(1,nt);
for i = 1:nt
    y(i) = (nt-i)/nt;
end
x = x(1:end-1);
y = y(1:end-1);
plot(x*tau*1e6,y,'.',"MarkerEdgeColor",[217,83,25]/255,"MarkerFaceColor",[217,83,25]/255);
fprintf("The mass of Na is %.2f.\n",matom/amu);
myfittype = fittype("a*x+b",...
    dependent="y",independent="x",...
    coefficients=["a" "b"]);
xlist = log(x*tau*1e6);
ylist = log(y');
f = fit(xlist(exp(xlist)>=22.0),ylist(exp(xlist)>=22.0),myfittype,'StartPoint',[-2,9]);
plot(exp(xlist),exp((f.a)*xlist+(f.b)),'black');
fprintf("The power law fit is y=ax^b where a=%.4f and b=%.4f.\n",f.a,f.b);
%a=-1.9178 b=4.2052

%K
load("fig2K.mat");
lifetimes = lifetime(dist>r0 & lifetime>0);
x = sort(lifetimes(:));
nt = length(lifetimes(:));
y = zeros(1,nt);
for i = 1:nt
    y(i) = (nt-i)/nt;
end
x = x(1:end-1);
y = y(1:end-1);
plot(x*tau*1e6,y,'.',"MarkerEdgeColor",[237,177,32]/255,"MarkerFaceColor",[237,177,32]/255);
fprintf("The mass of K is %.2f.\n",matom/amu);
myfittype = fittype("a*x+b",...
    dependent="y",independent="x",...
    coefficients=["a" "b"]);
xlist = log(x*tau*1e6);
ylist = log(y');
f = fit(xlist(exp(xlist)>=22.0),ylist(exp(xlist)>=22.0),myfittype,'StartPoint',[-2,9]);
plot(exp(xlist),exp((f.a)*xlist+(f.b)),'black');
fprintf("The power law fit is y=ax^b where a=%.4f and b=%.4f.\n",f.a,f.b);
%a=-1.7908 b=3.3054

%Rb
load("fig2Rb.mat");
lifetimes = lifetime(dist>r0 & lifetime>0);
x = sort(lifetimes(:));
nt = length(lifetimes(:));
y = zeros(1,nt);
for i = 1:nt
    y(i) = (nt-i)/nt;
end
x = x(1:end-1);
y = y(1:end-1);
plot(x*tau*1e6,y,'.',"MarkerEdgeColor",[126,47,142]/255,"MarkerFaceColor",[126,47,142]/255);
fprintf("The mass of Rb is %.2f.\n",matom/amu);
myfittype = fittype("a*x+b",...
    dependent="y",independent="x",...
    coefficients=["a" "b"]);
xlist = log(x*tau*1e6);
ylist = log(y');
f = fit(xlist(exp(xlist)>=11.0),ylist(exp(xlist)>=11.0),myfittype,'StartPoint',[-2,9]);
plot(exp(xlist),exp((f.a)*xlist+(f.b)),'black');
fprintf("The power law fit is y=ax^b where a=%.4f and b=%.4f.\n",f.a,f.b);
%a=-1.9441 b=3.2010

xscale("log");
yscale("log");
xlim([0.8,620]);
ylim([8e-5,1]);
xticks([1,10,100]);
legend("Yb$^+$Na","","Yb$^+$K","","Yb$^+$Rb","");
xlabel("Lifetime $t$ ($\mu$s)");
ylabel("Fraction of Complexes with Lifetime $\geq t$");
print(gcf,'-vector','-dsvg',mainhere+"/fig2.svg");
hold off;