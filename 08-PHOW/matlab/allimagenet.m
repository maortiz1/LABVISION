numW=[200 300 400 500 600 700 1000 2000];

parfor i=1:1:length(numW)
  prefix=sprintf('numW_%d',numW(i));
  phow_imagenet(numW(i),prefix,10,200);
  
end

c=[0.01 0.001 0.1 10 100 1000];
parfor i=1:1:length(c)
  prefix=sprintf('c_%d',c(i));
  phow_imagenet(1000,prefix,c(i),200);  
end

nClass=[10 40 70 100 130  150 180 200 ];
parfor i=1:1:length(nClass)
  prefix=sprintf('nclass_%d',nClass(i));
  phow_imagenet(1000,prefix,10,nClass(i));  
end