function imdb=getImdb_cub200(Name_batch,theConf,meta,IsTrain)
trainRate = 0.9;
[objset,~]=readAnnotation(Name_batch,theConf);
num=length(objset);
image_size=meta.normalization.imageSize(1:2);
data=zeros(image_size(1),image_size(2),3,num,'single');
Name_batch = GetNameBatch(theConf.data.imgdir);
labelNum = numel(Name_batch);
labels=ones(labelNum,num)*-1;
for i=1:num
    fprintf('reading image(%d/%d)\n',i,num);
    tar=i;
    IsFlip=false;
    [I_patch,~]=getI(objset(i),theConf,image_size,IsFlip);
    data(:,:,:,tar)=I_patch;
    for l = 1:labelNum
        if strcmp(Name_batch{1,l},objset(i).name)
            labels(l,i) = 1;
            break;
        end
    end
end
fprintf('preparing imdb...\n');
imdb.images.data=zeros(image_size(1),image_size(2),3,num,'single');
list=1:(num);
imdb.images.data(:,:,:,list)=data;
clear data
imdb.images.labels=labels;
list_train=round(linspace(1,num,round(num*trainRate)));
imdb.images.set=zeros(1,num,'uint8');
imdb.images.set(1:end)=2;
imdb.images.set(list_train)=1;
dataMean=mean(imdb.images.data(:,:,:,imdb.images.set==1),4);
imdb.images.data=bsxfun(@minus,imdb.images.data,dataMean);
if(IsTrain)
    list=1:num;
    imdb.images.data=imdb.images.data(:,:,:,list);
    imdb.images.labels=imdb.images.labels(:,list);
    imdb.images.set=imdb.images.set(:,list);
    imdb.images.alpha(list) = 1;
    [~,imdb.images.order]=sort(list);
end
imdb.meta.classes={Name_batch};
imdb.meta.sets={'train','val','test'} ;
imdb.meta.dataMean=dataMean;
end

function Name_batch = GetNameBatch(direction)
files = dir(direction);
size0 = size(files);
length = size0(1);
names = files(3:length);
for i = 1:size(names)
    Name_batch{1,i} = names(i).name;
end
end
