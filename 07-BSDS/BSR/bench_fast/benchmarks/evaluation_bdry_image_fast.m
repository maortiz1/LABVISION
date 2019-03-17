function [thresh,cntR,sumR,cntP,sumP] = evaluation_bdry_image_curves(inFile,gtFile, prFile, nthresh, radius, thinpb)
% [thresh,cntR,sumR,cntP,sumP] = boundaryPR_image(inFile,gtFile, prFile, nthresh, maxDist, thinpb)
%
% Calculate precision/recall curve.
%
% INPUT
%	inFile  : Can be one of the following:
%             - a soft or hard boundary map in image format.
%             - a collection of segmentations in a cell 'segs' stored in a mat file
%             - an ultrametric contour map in 'doubleSize' format, 'ucm2'
%               stored in a mat file with values in [0 1].
%
%	gtFile	: File containing a cell of ground truth boundaries
%   prFile  : Temporary output for this image.
%	nthresh	: Number of points in PR curve.
%   radius : in pixels for computing Precision / Recall.
%   thinpb  : option to apply morphological thinning on segmentation
%             boundaries.
%
% OUTPUT
%	thresh		Vector of threshold values.
%	cntR,sumR	Ratio gives recall.
%	cntP,sumP	Ratio gives precision.
%
if nargin<6, thinpb = 1; end
if nargin<5, radius = 3; end
if nargin<4, nthresh = 99; end

[~,~,e]=fileparts(inFile);
if strcmp(e,'.mat'),
    load(inFile);
end

if exist('ucm2', 'var'),
    pb = double(ucm2(3:2:end, 3:2:end));
    clear ucm2;
elseif ~exist('segs', 'var')
    pb = double(imread(inFile))/255;
end


load(gtFile);
if isempty(groundTruth),
    error(' bad gtFile !');
end
% ojo: curves
human = zeros(size(groundTruth{1}.Boundaries));
for i = 1:numel(groundTruth),
    human = human + groundTruth{i}.Boundaries;
end


if ~exist('segs', 'var')
    thresh = linspace(1/(nthresh+1),1-1/(nthresh+1),nthresh)';
else
    if nthresh ~= numel(segs)
        warning('Setting nthresh to number of segmentations');
        nthresh = numel(segs);
    end
    thresh = 1:nthresh; thresh=thresh';
end

% zero all counts
cntR = zeros(size(thresh));
sumR = zeros(size(thresh));
cntP = zeros(size(thresh));
sumP = zeros(size(thresh));

for t = 2:nthresh,
    
    if ~exist('segs', 'var')
        bmap = (pb>=thresh(t));
    else
        bmap = logical(seg2bdry(segs{t},'imageSize'));
    end
    
    if t==2,
        bmap_old = bmap;
        same_bmp = false;
    else
        if isequal(bmap, bmap_old),
            same_bmp = true;
        else
            same_bmp = false;
            bmap_old = bmap;
        end
    end
    
    if ~same_bmp
        % thin the thresholded pb to make sure boundaries are standard thickness
        if thinpb,
            bmap = double(bwmorph(bmap, 'thin', inf));    % OJO
        end
               
        % curves: compare with all GT
        [match1, match2] = correspondCurves(bmap,human,radius);
        % calcular recall
        cntR(t) = sum(match2(:));
        sumR(t) = sum(human(:));
        
        % calcular precision
        cntP(t) =  sum(match1(:));
        sumP(t) =  sum(bmap(:));
    else
        cntR(t) = cntR(t-1);
        sumR(t) = sumR(t-1);
        
        % calcular precision
        cntP(t) =  cntP(t-1);
        sumP(t) =  sumP(t-1);
    end
    
end

% output
fid = fopen(prFile,'w');
if fid==-1,
    error('Could not open file %s for writing.', prFile);
end
fprintf(fid,'%10g %10g %10g %10g %10g\n',[thresh cntR sumR cntP sumP]');
fclose(fid);

%%
function [match1, match2] = correspondCurves(bmap1, bmap2, radius)


str = strel(fspecial('disk',radius));

% binarios
BW1 = logical(bmap1);
BW2 = (bmap2 >0);

% dilatar humano y pb para compararlos
% version continua : con Fast Marching
BW1d  = imdilate(BW1, str);
BW2d  = imdilate(BW2, str);

match1 = double( BW1 & BW2d );
% ojo : ya no es binario
match2 = bmap2.*( BW1d & BW2 );