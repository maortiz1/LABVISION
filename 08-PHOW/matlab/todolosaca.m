
dx=dir('imageNet');

allaca={};
alltime={};
allinf=struct()
k=1;
for i=1:1:length(dx)
  [~,~,ext]=fileparts(dx(i).name);
  if ~isempty(strfind(dx(i).name,'result')) && (isequal(ext,'.mat'))
   load(fullfile('imageNet',dx(i).name))
   allinf(k).aca=aca;  
   allinf(k).time=timeEx;
   allinf(k).numwords=conf.numWords;
   allinf(k).numspatialX=conf.numSpatialX;
   allinf(k).numspatialY=conf.numSpatialY;
   allinf(k).numTrain=conf.numTrain;
   allinf(k).numTest=conf.numTest;
   allinf(k).prefix=conf.prefix;
   allinf(k).C=conf.svm.C;
   allinf(k).numclasses=conf.numClasses;
   allinf(k).numWords=conf.numWords
allinf(k).confus=confus;
	k=k+1;
  end
end

