c=[0.01;0.1;1;0.001;10;100;1000;0.00001]

for i=1:1:length(c)
  s=sprintf('img_c_%f',c(i));
  disp(s)
  phow_caltech(15,s,c(i))
end