function imdb = getImdb_CelebA_plus(Name_batch,theConf,meta)
  fileID=fopen([theConf.data.catedir,'/Eval/list_eval_partition.txt'],'r');
  idsetPair=textscan(fileID,'%s %d');
  filename=fopen([theConf.data.catedir,'/Anno/list_attr_celeba.txt'],'r');
  idclassPair= textscan(filename,'%s%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d','headerlines',2);
  idclassPair = cat(2, idclassPair{2:end})';
  filebbox = fopen([theConf.data.catedir,'/Anno/list_bbox_celeba.txt'],'r');
  idBboxPair = textscan(filebbox,'%s%d%d%d%d','headerlines',2);
  idBboxPair = cat(2, idBboxPair{2:end});
  idBboxPair(:,3) = idBboxPair(:,3) + idBboxPair(:,1); idBboxPair(:,4) = idBboxPair(:,4) + idBboxPair(:,2);
  num=numel(idsetPair{1,1});
  image_size=meta.normalization.imageSize(1:2);
  dataMean = zeros(image_size(1),image_size(2),3);
  numtrain = 0;
  for i=1:num
    fprintf('reading image(%d/%d)\n',i,num);
    IsFlip=false;
    bbox = idBboxPair(i,:);
    [I_dir,I_patch,~]=getI_CelebA_plus(i,theConf,image_size,bbox,IsFlip);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    imdb.images.dir{i}=I_dir;
    imdb.images.labels(:,i) = double(idclassPair(:,i));
    imdb.images.set(i) = idsetPair{1,2}(i)+1;
    if idsetPair{1,2}(i) == 0
        dataMean = dataMean + I_patch;
        numtrain = numtrain + 1;
    end
    imdb.images.order(i)=i;
  end

dataMean = dataMean ./ numtrain;
%imdb.images.data=bsxfun(@minus,imdb.images.data,dataMean); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imdb.meta.classes={Name_batch};
imdb.meta.sets={'train','val','test'} ;
imdb.meta.dataMean=dataMean;
end



