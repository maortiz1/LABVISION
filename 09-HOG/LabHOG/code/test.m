
if ~exist('modelTrained.mat')

run('main.m')

end

load('modelTrained.mat')
test_scn_path='../data/test_scenes/test_jpg';
%[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params,confi,ext);

randTestImages=randi([1 117],1,10);
visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)
    
    

