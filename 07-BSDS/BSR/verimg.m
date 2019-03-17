load(fullfile('kmeans','101084.mat'));
img=imread(fullfile('BSDS500','data','images','test','101084.jpg'));
subplot(2,3,1)
imshow(img)
title('Original')

subplot(2,3,2)
imshow(segs{2},[])
title('K=3')

subplot(2,3,3)
imshow(segs{5},[])
title('K=7')


subplot(2,3,4)
imshow(segs{10},[])
title('K=11')

subplot(2,3,5)
imshow(segs{15},[])
title('K=16')


subplot(2,3,6)
imshow(segs{19},[])
title('K=19')

suptitle('KMeans LAB')
