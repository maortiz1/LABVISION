
function [bboxes, confidences, image_ids]  = detectorWALDO(dirtest,w,b,feature_params)

allImgs=dir(dirtest)
bboxes=[];
confidences=[];
image_ids=[];
for j=1:length(allImgs)
  dirpath=fullfile(dirtest,allImgs(j).name)
  fprintf('Detecting faces in %s\n', allImgs(j).name)
  [temp_bboxes, temp_confidences, temp_image_ids]  = run_detector(dirpath,w,b,feature_params);
  bboxes      = [bboxes;      temp_bboxes];
  confidences = [confidences; temp_confidences];
  image_ids   = [image_ids;   temp_image_ids];



end