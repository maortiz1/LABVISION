numS=[[4 2];[4 8];[2 4];[9 4]]

for i=1:1:length(numS)
  ac= numS(i,1:2);
  s=sprintf('iprueb%d-%d',ac);
  disp(s)
  phow_caltech(15,s,10,ac)
end