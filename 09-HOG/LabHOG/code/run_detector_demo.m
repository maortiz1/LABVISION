% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params,confi)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpeg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);
cell_sz=feature_params.hog_cell_size;
wdw_sz = feature_params.template_size;
pxl = 6; % pixeles que voy a considerar en la ventana
%scales = [1.15,0.95,0.8,0.65,0.5,0.35,0.2,0.1,0.06];
scales = [1, 0.9, 0.8, 0.7,0.6,0.5,0.4,0.3,0.2,0.1];
%scales=[1,0.7,0.5,0.3,0.1,0.05];



parfor i = 1:length(test_scenes)
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img =imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255; % normalización de la imagen
    temp_bboxes=zeros(0,4);
    temp_confidences=zeros(0,1);
    temp_image_ids=cell(0,1);
    num_img=1;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
 for scale = scales
       img_temp=imresize(img,scale);
       %disp(size(img_temp))
       % posiciones que realmente me puedo mover sin salirme de la ventana
       dxtot=floor((size(img_temp,2)-wdw_sz)/pxl);
        dytot=floor((size(img_temp,1)-wdw_sz)/pxl);
        for dx=1:dxtot
            for dy=1:dytot
            %encuentro las posiciones de la ventana
                ypos1=int16((dy-1)*pxl+1);
                ypos2=int16(ypos1+wdw_sz-1);
                xpos1=int16((dx-1)*pxl+1);
                xpos2=int16(xpos1+wdw_sz-1);
                %window=histeq(img_temp(ypos1:ypos2,xpos1:xpos2,:));
                window=(img_temp(ypos1:ypos2,xpos1:xpos2,:));
   %for k=1:size(img_temp,1)-wdw_sz+1 
   % for j=1:size(img_temp,2)-wdw_sz+1    
               
               %window = img_temp(k:(k+wdw_sz-1),j:(j+wdw_sz-1));
                hog_feat=vl_hog(im2single(window),cell_sz);
                hog_feat=hog_feat(:)';
                confidence=hog_feat*w+b;
                % variar umbral de conficence
                %fprintf('confidence: %.4f \n', confidence)
                if confidence>=confi
                    box=int32([xpos1,ypos1,xpos2,ypos2]/scale);
                   % box=int32([j,k,(j+wdw_sz-1),(k+wdw_sz-1)]/scale);
                    temp_bboxes=[temp_bboxes;box;];
                    temp_confidences=[temp_confidences; confidence;];
                    temp_image_ids{num_img,1}=test_scenes(i).name;
                    num_img=num_img+1;
                end
            end
        end
    end
 %supresión de no máximos para saber cuales son las verdaderas
     [max] = non_max_supr_bbox(temp_bboxes, temp_confidences, size(img));

    temp_confidences = temp_confidences(max,:);
    temp_bboxes      = temp_bboxes(max,:);
    temp_image_ids   = temp_image_ids(max,:);
 
    bboxes      = [bboxes;      temp_bboxes];
    confidences = [confidences; temp_confidences];
    image_ids   = [image_ids;   temp_image_ids];
end




