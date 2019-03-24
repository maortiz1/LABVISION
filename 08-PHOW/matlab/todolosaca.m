dx=dir('matlab/data');
allaca={};
for i=1:length(dx)
  if (contains(dx(i).name,'result'))
  load(fullfile('matlab','data',dx(i).name))
  allca{i}=aca;
  
  end



end