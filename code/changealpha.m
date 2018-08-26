function imdb=changealpha(imdb,negval,posval)
     tmpnegtrain=load('negtrain.mat');
     tmppostrain=load('postrain.mat');
     negtrain=tmpnegtrain.negtrain;
     postrain=tmppostrain.postrain;
     negtrainlist = sort(negtrain);
     postrainlist = sort(postrain);
     negvallist = sort(negval);
     posvallist = sort(posval);
     total= [negtrainlist, negvallist];
     list = sort(total);
     imdb.images.alpha(negtrainlist)=0;
     imdb.images.alpha(postrainlist)=1;
     imdb.images.alpha(negvallist)=0;
     imdb.images.alpha(posvallist)=1;
%      newimdb.meta=imdb.meta;
%      newimdb.images.data=imdb.images.data(:,:,:,list);
%      newimdb.images.labels=imdb.images.labels(:,list);
%      newimdb.images.set=imdb.images.set(list);
%      newimdb.images.alpha=imdb.images.alpha(list);
%      [~,newimdb.images.order]=sort(list);
%      save('mat/cub200/falseimdb.mat','-struct','newimdb');
%      clear newimdb;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      rightlist=setdiff(imdb.images.order, list);
%      rightimdb.meta=imdb.meta;
%      rightimdb.images.data=imdb.images.data(:,:,:,rightlist);
%      rightimdb.images.labels=imdb.images.labels(:,rightlist);
%      rightimdb.images.set=imdb.images.set(rightlist);
%      rightimdb.images.alpha=1;
%      [~,rightimdb.images.order]=sort(rightlist);
%      save('mat/cub200/rightimdb.mat','-struct','rightimdb');
%      clear rightimdb;    
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

