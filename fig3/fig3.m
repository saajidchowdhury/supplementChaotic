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
main = string(datetime('now','Format','M-d-y@HH.mm.ss'))+"fig3";
mkdir(main);

dataLi = [0.12038129778504201572,0.12013181177519990928;
          0.02214927999397075381,0.02218586327906779329;
          0.00434511471708576378,0.00437030320846405317;
          0.00193691502352575100,0.00195620620938293313;];
dataNa = [0.13568630492727873360,0.13565142086580642133;
          0.03294114918893557842,0.03304840018595503470;
          0.00843734497707997189,0.00841735411090672667;
          0.00411102167419705958,0.00416689614515128055;];
dataK  = [0.19517882279586457051,0.19548278391602877391;
          0.05244693699448722130,0.05262765442469335975;
          0.01547972731259087721,0.01559287561513144661;
          0.00806811367886012959,0.00812728664273293551;]; %De=1K, De=100K
dataRb = [0.22478459591812499641,0.22508705772332621153;   %E=1uK
          0.06593947225512662713,0.06602873147259016862;   %E=10uK
          0.02358312487024678480,0.02357592815842441542;   %E=50uK
          0.01342656540226770320,0.01347274430312790129;]; %E=100uK

figure; hold on;

%Li
xval = 1;
h = bar3(flipud(dataLi),'grouped');
Xdat = get(h,'Xdata');
for i=1:length(Xdat)
    Xdat{i}=Xdat{i}+(xval-1)*ones(size(Xdat{i}));
    h(i).XData = Xdat{i};
end
h(1).FaceColor = [153,215,255]/255;
h(2).FaceColor = [0,114,189]/255;

%Na
xval = 2;
h = bar3(flipud(dataNa),'grouped');
Xdat = get(h,'Xdata');
for i=1:length(Xdat)
    Xdat{i}=Xdat{i}+(xval-1)*ones(size(Xdat{i}));
    h(i).XData = Xdat{i};
end
h(1).FaceColor = [244,188,164]/255;
h(2).FaceColor = [217,83,25]/255;

%K
xval = 3;
h = bar3(flipud(dataK),'grouped');
Xdat = get(h,'Xdata');
for i=1:length(Xdat)
    Xdat{i}=Xdat{i}+(xval-1)*ones(size(Xdat{i}));
    h(i).XData = Xdat{i};
end
h(1).FaceColor = [247,222,161]/255;
h(2).FaceColor = [237,177,32]/255;

%Rb
xval = 4;
h = bar3(flipud(dataRb),'grouped');
Xdat = get(h,'Xdata');
for i=1:length(Xdat)
    Xdat{i}=Xdat{i}+(xval-1)*ones(size(Xdat{i}));
    h(i).XData = Xdat{i};
end
h(1).FaceColor = [204,140,217]/255;
h(2).FaceColor = [126,47,142]/255;

view(3);
xlabel('Atom species');
ylabel('Bath temperature');
zlabel('Probability of complex formation');
xlim([0.5,4.5]);
ylim([0.5,4.5]);
xticks([1,2,3,4]);
xticklabels({'Li','Na','K','Rb'});
yticks([1,2,3,4]);
yticklabels(fliplr({'1$\mu$K','10$\mu$K','50$\mu$K','100$\mu$K'}));
print(gcf,'-vector','-dsvg',main+"/fig3.svg");
hold off;