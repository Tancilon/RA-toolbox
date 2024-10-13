clear

%label
sim_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_labeled\test\cuhk03labeled_6workers.mat';
query_label_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_labeled\label&cam\bdb-cuhk03labeled-query_id-.mat';
gallery_label_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_labeled\label&cam\bdb-cuhk03labeled-gallery_idtest-.mat';
cam_gallery_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_labeled\label&cam\bdb-cuhk03labeled-gallery_camidstest-.mat';
cam_query_path = 'D:\RA_ReID\ReID_Dataset\CUHK03_labeled\label&cam\bdb-cuhk03labeled-query_camids-.mat';
datasetname = 'label';



sim = importdata(sim_path);
% ranker * query * gallery
% fprintf('market1501-eu-test:bdb\n');
fprintf('Running %s\n', datasetname);
% config; 

query_label0 = importdata(query_label_path);
% query_label = query_label0(1685:3368);
query_label = query_label0;
query_label = query_label';
gallery_label = importdata(gallery_label_path);
cam_gallery = importdata(cam_gallery_path);
cam_gallery = cam_gallery';
cam_query0 = importdata(cam_query_path);
% cam_query = cam_query0(1685:3368);
cam_query = cam_query0;
cam_query = cam_query';


iteration = 1; % Number of interaction rounds
K = 5; % Interaction per round

error_rate = 0.02; % Interaction error rate
fprintf('K = %d\n',K);

rankernum = size(sim,1);
querynum = size(sim,2);
gallerynum = size(sim,3);
[~,ranklist] = sort(-sim,3);
[~,rank] = sort(ranklist,3);

averageRank = sum(sim, 1);
averageRank = reshape(averageRank,querynum,gallerynum);
[~,pseudoRanklist] = sort(-averageRank,2);
[~,pseudoRank] = sort(pseudoRanklist,2);

feedtrue_G = zeros(querynum,gallerynum);
feeded_G = zeros(querynum,gallerynum);

weight = ones(querynum,rankernum);

%get origin rank
origin_sim = zeros(querynum,gallerynum);
for i=1:querynum
    for j = 1:rankernum
        origin_sim(i,:) = origin_sim(i,:) + reshape(sim(j,i,:) * weight(i,j),1,gallerynum);
    end
end
[~,origin_ranklist] = sort(-origin_sim,2);
[~,origin_rank] = sort(origin_ranklist,2);
total_ranklist = origin_ranklist;

%%% evaluation
result_1 = [];
[CMC_result, map_result, ~, ~] = evaluation(origin_rank', gallery_label, query_label, cam_gallery, cam_query);
auc_result = 0.5*(2*sum(CMC_result) - CMC_result(1) - CMC_result(end))/(length(CMC_result)-1);
result_1 = [CMC_result([1,5,10,20]).*100,auc_result, map_result];
fprintf('original r1:%.2f%% mAP:%.2f%%\n',100*CMC_result(1),100*map_result);


for i = 1:iteration
    
    new_weight = zeros(querynum,rankernum);
    tic
    for q = 1:querynum
        Qlabel = query_label(q);
        sed = 0;
        now_num = 1;
        while sed<K
            if feeded_G(q,total_ranklist(q,now_num)) == 0
                sed = sed +1;
                RT(sed) = total_ranklist(q,now_num);
                feeded_G(q,total_ranklist(q,now_num)) = 1;
            end
            now_num = now_num + 1;
        end
        RT_label = gallery_label(RT);
        scored_G = find(RT_label == Qlabel);
        for j = 1:K
            if ismember(j,scored_G)
                if rand(1) > error_rate
                    feedtrue_G(q,RT(j)) = 10;
                else
                    feedtrue_G(q,RT(j)) = -10;
                end
            else
                if rand(1) > error_rate
                    feedtrue_G(q,RT(j)) = -10;
                else
                    feedtrue_G(q,RT(j)) = 10;
                end
            end
        end
        scored_G = find(feedtrue_G(q,:)==10);
        %tic
        for j = 1:rankernum
            ranker_RT = ranklist(j,q,:);
            A = [];
            for k = 1:size(scored_G,2)
                x = find(ranker_RT==scored_G(k));
                score = ceil(x/K);
                new_weight(q,j) = new_weight(q,j) + 1/score;
            end
        end
        total_weight = max(new_weight(q,:));
        new_weight(q,:) = new_weight(q,:) ./ total_weight;
        %toc
    end
    
    
    weight = weight .* 0.1 + new_weight .* 0.9;
    for j = 1:querynum
        weight(j,:) = weight(j,:) ./ max(weight(j,:));
    end
    new_sim = zeros(querynum,gallerynum);
    for j=1:querynum
        for k = 1:rankernum
            new_sim(j,:) = new_sim(j,:) + reshape(sim(k,j,:) * weight(j,k),1,gallerynum);
        end
    end
    new_sim = new_sim + feedtrue_G;
    toc
    [~,total_ranklist] = sort(-new_sim,2);
    [~,total_rank] = sort(total_ranklist,2);
    

    %%% evaluation
    [CMC_result, map_result, ~, ~] = evaluation(total_rank', gallery_label, query_label, cam_gallery, cam_query);
    auc_result = 0.5*(2*sum(CMC_result) - CMC_result(1) - CMC_result(end))/(length(CMC_result)-1);
    result_1 = [result_1 ;CMC_result([1,5,10,20]).*100,auc_result, map_result];
    fprintf('iteration:%d r1:%.2f%% mAP:%.2f%%\n',i,100*CMC_result(1),100*map_result);
end

save('Rank_based_duke_result.mat','total_rank');
