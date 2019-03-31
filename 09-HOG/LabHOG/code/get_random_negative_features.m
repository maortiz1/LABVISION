% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
samples_por_img=round(num_samples/num_images);

% placeholder to be deleted - THIS ONLY WORKS FOR THE INITIAL DEMO
s_sample=feature_params.template_size;%size of sample
dimensionality = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31; %dimensions
features_neg = zeros(samples_por_img*num_images,dimensionality);%

for i=1:1:num_images
  disp(fullfile(non_face_scn_path,image_files(i).name));
  img=imread(fullfile(non_face_scn_path,image_files(i).name));
  img=rgb2gray(img);
  [w,h]=size(img);
  if feature_params.template_size>=w ||feature_params.template_size>=h
    img=imresize(img,[feature_params.template_size*2,feature_params.template_size*2]);  
      [w,h]=size(img);
  end
  for j=1:1:samples_por_img
    x_beg = randi([1 w-s_sample]); %begginig of sample in x
    y_beg = randi([1 h-s_sample]); %ending of sample in y
    x_end = x_beg+s_sample;%end of sample in x
    y_end = y_beg+s_sample;% end of sample in y
    sample= img(x_beg:x_end,y_beg:y_end); 
    hog = vl_hog(single(sample),feature_params.hog_cell_size); 
    index=(i-1)*samples_por_img+j;
    features_neg(index,:)= reshape(hog, [1,dimensionality]);
  end
end

