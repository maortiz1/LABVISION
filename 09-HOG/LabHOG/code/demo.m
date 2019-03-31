zipsave='../data/extra_test_scenes/demo.zip';

if ~exist(zipsave)

f_name=websave(zipsave,'https://www.dropbox.com/s/eclnk3qgg5tm760/test_HOG.zip?dl=1');
end
if ~exist('../data/extra_test_scenes/test_HOG','dir')
f_names=unzip('../data/extra_test_scenes/demo.zip','../data/extra_test_scenes/');

end


if ~exist('modelTrained.mat')

run('main.m')

else
  
  load('modelTrained.mat')
  test_scn_path='../data/extra_test_scenes/test_HOG';
  ext='*.jpeg';
  [bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params,confi,ext);
  visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_scn_path)
end