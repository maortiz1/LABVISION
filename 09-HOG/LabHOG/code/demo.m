zipsave='../data/extra_test_scenes/demo.zip';

if ~exist(zipsave)

f_name=websave(zipsave,'https://www.dropbox.com/s/eclnk3qgg5tm760/test_HOG.zip?dl=1');
end
if ~exist('../data/extra_test_scenes/test_HOG','dir')
f_names=unzip('../data/extra_test_scenes/demo.zip','../data/extra_test_scenes/');

end


if ~exist('modelTrained.mat')

run('main.m')

end

label_path = fullfile('../data/','test_scenes/ground_truth_bboxes.txt'); 
load('modelTrained.mat')
test_scn_path='../data/extra_test_scenes/test_HOG';
ext='*.jpeg';
[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params,0.25,ext);
visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_scn_path,ext);
test_scn_path='../data/test_scenes/test_jpg';
load('modelTrained.mat')
%[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params,confi,ext);

randTestImages=randi([1 117],1,10);
visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path,false,randTestImages)
    
    

