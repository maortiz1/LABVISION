numS=[15 40 100 200 300];

for i=1:1:length(numS)
  ac= numS(i);
  s=sprintf('img%d',ac)
  disp(s)
  phow_caltech(ac,s,10,[4,2])
end