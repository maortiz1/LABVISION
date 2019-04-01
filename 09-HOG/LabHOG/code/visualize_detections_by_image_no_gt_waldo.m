
function visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_scn_path,ext)



test_files = dir(fullfile(test_scn_path, ext));
num_test_images = length(confidences);

for i=1:num_test_images
   cur_test_image = imread( fullfile( test_scn_path, image_ids(i)));
      
   cur_detections = strcmp(test_files(i).name, image_ids);
   cur_bboxes = bboxes(i,:);
   cur_confidences = confidences(i);
   
   figure(15)
   imshow(cur_test_image);
   hold on;
   
   num_detections = sum(cur_detections);
   
   for j = 1:num_detections
       bb = cur_bboxes(j,:);
       plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'g:','linewidth',2);
   end
 
   hold off;
   axis image;
   axis off;
   title(sprintf('image: "%s" green=detection', test_files(i).name),'interpreter','none');
    
   set(15, 'Color', [.988, .988, .988])
   pause(0.1) %let's ui rendering catch up
   detection_image = frame2im(getframe(15));
   % getframe() is unreliable. Depending on the rendering settings, it will
   % grab foreground windows instead of the figure in question. It could also
   % return a partial image.
   imwrite(detection_image, sprintf('visualizations/detections_%s.png', test_files(i).name))
   
   fprintf('press any key to continue with next image\n');
   pause;
end



