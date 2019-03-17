function plot_eval(evalDir1,evalDir2,evalDir3,col)
% evalDir1 = pablo
% evalDir2 = Gmm us
% evalDir3 = kmeans us
% plot evaluation results.
% Pablo Arbelaez <arbelaez@eecs.berkeley.edu>

if nargin<2, col = 'r'; end

fwrite(2,sprintf('\n%s\n',evalDir1));

if exist(fullfile(evalDir1,'eval_bdry_thr.txt'),'file') && exist(fullfile(evalDir2,'eval_bdry_thr.txt'),'file') && exist(fullfile(evalDir3,'eval_bdry_thr.txt'),'file')
    open('isoF.fig');
    hold on
    prvals1 = dlmread(fullfile(evalDir1,'eval_bdry_thr.txt')); % thresh,r,p,f
    f=find(prvals1(:,2)>=0.01);
    prvals1 = prvals1(f,:);

    prvals2 = dlmread(fullfile(evalDir2,'eval_bdry_thr.txt')); % thresh,r,p,f
    f=find(prvals2(:,2)>=0.01);
    prvals2 = prvals2(f,:);

    prvals3 = dlmread(fullfile(evalDir3,'eval_bdry_thr.txt')); % thresh,r,p,f
    f=find(prvals3(:,2)>=0.01);
    prvals3 = prvals3(f,:);

    evalRes = dlmread(fullfile(evalDir1,'eval_bdry_thr.txt'));
    evalRes2 = dlmread(fullfile(evalDir2,'eval_bdry_thr.txt'));
    evalRes3 = dlmread(fullfile(evalDir3,'eval_bdry_thr.txt'));

    if size(prvals1,1)>1,

        plot(prvals1(1:end,2), prvals1(1:end,3),'r','LineWidth',3,'DisplayName','UCM2');

        plot(prvals2(1:end,2), prvals2(1:end,3),'b','LineWidth',3,'DisplayName','GMM-Lab');

        plot(prvals3(1:end,2), prvals3(1:end,3),'k','LineWidth',3,'DisplayName','Kmeans-HSV');

    else
        plot(evalRes(2),evalRes(3),'o','MarkerFaceColor',col,'MarkerEdgeColor',col,'MarkerSize',8);
    end
    hold off
%legend
    fprintf('Boundary UCM\n');
    fprintf('ODS: F( %1.2f, %1.2f ) = %1.2f   [th = %1.2f]\n',evalRes(2:4),evalRes(1));
    fprintf('OIS: F( %1.2f, %1.2f ) = %1.2f\n',evalRes(5:7));
    fprintf('Area_PR = %1.2f\n\n',evalRes(8));

    fprintf('Boundary GMM\n');
    fprintf('ODS: F( %1.2f, %1.2f ) = %1.2f   [th = %1.2f]\n',evalRes2(2:4),evalRes2(1));
    fprintf('OIS: F( %1.2f, %1.2f ) = %1.2f\n',evalRes2(5:7));
    fprintf('Area_PR = %1.2f\n\n',evalRes2(8));

    fprintf('Boundary Kmeans \n');
    fprintf('ODS: F( %1.2f, %1.2f ) = %1.2f   [th = %1.2f]\n',evalRes3(2:4),evalRes3(1));
    fprintf('OIS: F( %1.2f, %1.2f ) = %1.2f\n',evalRes3(5:7));
    fprintf('Area_PR = %1.2f\n\n',evalRes3(8));
end

if exist(fullfile(evalDir1,'eval_cover.txt'),'file') && exist(fullfile(evalDir2,'eval_cover.txt'),'file') && exist(fullfile(evalDir3,'eval_cover.txt'),'file')
    
    evalRes = dlmread(fullfile(evalDir1,'eval_cover.txt'));
    fprintf('Region UCM2 \n');
    fprintf('GT covering: ODS = %1.2f [th = %1.2f]. OIS = %1.2f. Best = %1.2f\n',evalRes(2),evalRes(1),evalRes(3:4));
    evalRes = dlmread(fullfile(evalDir1,'eval_RI_VOI.txt'));
    fprintf('Rand Index: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes(2),evalRes(1),evalRes(3));
    fprintf('Var. Info.: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes(5),evalRes(4),evalRes(6));

    evalRes1 = dlmread(fullfile(evalDir2,'eval_cover.txt'));
    fprintf('Region GMM \n');
    fprintf('GT covering: ODS = %1.2f [th = %1.2f]. OIS = %1.2f. Best = %1.2f\n',evalRes1(2),evalRes1(1),evalRes1(3:4));
    evalRes1 = dlmread(fullfile(evalDir2,'eval_RI_VOI.txt'));
    fprintf('Rand Index: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes1(2),evalRes1(1),evalRes1(3));
    fprintf('Var. Info.: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes1(5),evalRes1(4),evalRes1(6));


    evalRes2 = dlmread(fullfile(evalDir3,'eval_cover.txt'));
    fprintf('Region KMEANS \n');
    fprintf('GT covering: ODS = %1.2f [th = %1.2f]. OIS = %1.2f. Best = %1.2f\n',evalRes2(2),evalRes2(1),evalRes2(3:4));
    evalRes2 = dlmread(fullfile(evalDir3,'eval_RI_VOI.txt'));
    fprintf('Rand Index: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes2(2),evalRes2(1),evalRes2(3));
    fprintf('Var. Info.: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes2(5),evalRes2(4),evalRes2(6));


end