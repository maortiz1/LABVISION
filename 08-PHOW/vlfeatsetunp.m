
if ~exist('vlfeat-0.9.21-bin.tar.gz','file')
  websave('vlfeat-0.9.21-bin.tar.gz','http://www.vlfeat.org/download/vlfeat-0.9.21-bin.tar.gz');
end

if ~exist('vlfeat-0.9.21','dir')
	untar('vlfeat-0.9.21-bin.tar.gz')
   disp('enter')
end
run('vlfeat-0.9.21/toolbox/vl_setup')


if ~exist('imageNet200.tar','file')
  websave('imageNet200.tar','http://bcv001.uniandes.edu.co/imageNet200.tar');
end
if ~exist('imageNet200','dir')
untar('imageNet200.tar')
end