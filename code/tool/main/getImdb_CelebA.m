function imdb = getImdb_CelebA(Name_batch,theConf,meta,IsTrain)
  reduce=10;
  fileID=fopen([theConf.data.catedir,'/Eval/list_eval_partition.txt'],'r');
  idsetPair=textscan(fileID,'%s %d');
  filename=fopen([theConf.data.catedir,'/Anno/list_attr_celeba.txt'],'r');
  idclassPair= textscan(filename,'%s%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d','headerlines',2);
  num=numel(idsetPair{1,1});
  image_size=meta.normalization.imageSize(1:2);
  data=zeros(image_size(1),image_size(2),3,ceil(num/reduce),'single');
  j = 1;
  for i=1:reduce:num
    fprintf('reading image(%d/%d)\n',i,num);
    tar=j;
    IsFlip=false;
    [I_patch,~]=getI_CelebA(i,theConf,image_size,IsFlip);
    imdb.images.data(:,:,:,tar)=I_patch;
    for k=1:40
    imdb.images.labels(k,j) = double(idclassPair{1,k+1}(j,1));
    end;
    imdb.images.set(j) = idsetPair{1,2}(i)+1;
    imdb.images.alpha(i)=1;
    imdb.images.order(j)=j;
    j = j+1;
  end
clear data
dataMean=mean(imdb.images.data(:,:,:,imdb.images.set==1),4);
imdb.images.data=bsxfun(@minus,imdb.images.data,dataMean);
% if(IsTrain)
%     list=1:ceil(num/2);
%     imdb.images.data=imdb.images.data(:,:,:,list);
%     imdb.images.labels=imdb.images.labels(:,list);
%     imdb.images.set=imdb.images.set(:,list);
%     imdb.images.alpha(list) = 1;
%     [~,imdb.images.order]=sort(list);
% end
imdb.meta.classes={Name_batch};
imdb.meta.sets={'train','val','test'} ;
imdb.meta.dataMean=dataMean;
end



