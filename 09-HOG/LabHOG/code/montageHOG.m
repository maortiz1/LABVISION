
clc
clear all
train_path_pos='../data/caltech_faces/Caltech_CropFaces';
image_files = dir( fullfile( train_path_pos, '*.jpg') )

index=randi([1 length(image_files)],100,1);

image_files=image_files(index);

montagetemp={};
for i=1:length(image_files)
  montagetemp{i}=(fullfile(train_path_pos,image_files(i).name));

end

%montageTr=montage(montagetemp,'Size',[10,10]);

%imshow(montageTr)


train_path_pos='../data/test_scenes/test_jpg';
image_files = dir( fullfile( train_path_pos, '*.jpg') )

index=randi([1 length(image_files)],9,1);

image_files=image_files(index);

montagetemp={};

for i=1:length(image_files)
  montagetemp=imread(fullfile(train_path_pos,image_files(i).name));
  subplot(3,3,i)
  imshow(montagetemp)
  title(sprintf('Test: %d',index(i)))
end

