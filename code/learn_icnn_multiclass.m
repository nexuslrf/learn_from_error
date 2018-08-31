function [net,info]=learn_icnn_multiclass(model,Name_batch,lossType,dropoutRate)
load(['./mat/',Name_batch{1},'/conf.mat'],'conf');
labelNum=numel(Name_batch);
str=[];
if(labelNum>10)
    for i=1:numel(Name_batch)
        str=sprintf('%s_%s',str,Name_batch{i}(end-1:end));
    end
else
    for i=1:numel(Name_batch)
        str=sprintf('%s_%s',str,Name_batch{i}(1:end));
    end
end
if(strcmp(Name_batch{1},'cub200'))
    str=sprintf('%s',Name_batch{1}(1:end));
end
if(strcmp(Name_batch{1},'CelebA'))
    str=sprintf('%s',Name_batch{1}(1:end));
end
conf.data.Name_batch=str;
opts.dataDir=conf.data.imgdir;
opts.expDir=fullfile(conf.output.dir,str);
opts.imdbPath=fullfile(opts.expDir,'imdb.mat');
opts.whitenData=true;
opts.contrastNormalization=true;
opts.networkType='simplenn';
try
    gpuID=4;
    for i=numel(gpuID)
        gpuDevice(gpuID(i));
    end
    opts.train=struct('gpus',gpuID);
catch
    error('Errors here: GPU invalid.\n')
end


%% Prepare model
if(strcmp(Name_batch{1},'cub200'))
    labelNum = numel(GetNameBatch(conf.data.imgdir));
end
if(strcmp(Name_batch{1},'CelebA'))
    labelNum = 40;
end
net=network_init(labelNum,model,dropoutRate,'networkType',opts.networkType);
net.layers{end}.type=lossType;
if(strcmp(lossType,'ourloss_softmaxlog'))
    if(labelNum>10)
        net.meta.trainOpts.learningRate=net.meta.trainOpts.learningRate./50; %10 %%%%%%%%%%%%%%%%%%%
        if(strcmp(model,'vgg-m'))
            net.meta.trainOpts.learningRate=net.meta.trainOpts.learningRate./10;
        end
    end
else
    if(labelNum>10) 
        net.meta.trainOpts.learningRate=net.meta.trainOpts.learningRate./10;
        if(strcmp(model,'vgg-m'))
            net.meta.trainOpts.learningRate=net.meta.trainOpts.learningRate.*2;
        end
    end
end

opts.imageSize = net.meta.normalization.imageSize;
%% Prepare data
if exist(opts.imdbPath,'file')
    imdb=load(opts.imdbPath) ;
    if startsWith(model, 'vgg')
        load('dataMean_vgg.mat');
        imdb.meta.dataMean = dataMean_vgg;
    end
elseif(strcmp(Name_batch{1},'CelebA'))
        imdb=getImdb_CelebA_plus(Name_batch{1},conf,net.meta);
%     if(strcmp(Name_batch{1},'cub200'))
%         filename=['./mat/',Name_batch{1},'/imdb.mat'];
%         fprintf('generating set ');
%         fprintf(Name_batch{1});
%         fprintf('(%d/%d)',1,labelNum);
%         fprintf('...');
%         fprintf('\n');
%         if(~exist(filename,'file'))
%             IsTrain=true;
%             imdb=getImdb_cub200(Name_batch{i},conf,net.meta,IsTrain);
%         end
%         fprintf('set ');
%          IsTrain=true;fprintf(Name_batch{i});
%         fprintf(' generated \n');
%     else
%         for i=1:labelNum
%             filename=['./mat/',Name_batch{i},'/imdb.mat'];
%             fprintf('generating set ');
%             fprintf(Name_batch{i});
%             fprintf('(%d/%d)',i,labelNum);
%             fprintf('...');
%             fprintf('\n');
%             if(~exist(filename,'file'))
%                 IsTrain=true;
%                 if(strcmp(Name_batch{i},'cub200'))
%                     break;
%                 else
%                     imdb=getImdb(Name_batch{i},conf,net.meta,IsTrain);
%                 end
%                 mkdir(['./mat/',Name_batch{i}]);
%                 save(filename,'-struct','imdb');
%                 clear imdb;
%                 fprintf('set ');
%                 fprintf(Name_batch{i});
%                 fprintf(' generated \n');
%             end
%         end
%     imdb=produceIMDB(labelNum,net,Name_batch,conf);
%     end
%     fprintf('saving imdb...');
    mkdir(opts.expDir);
    save(opts.imdbPath,'-struct','imdb');
    fprintf('imdb saved');
else
      for i=1:labelNum
        filename=['./mat/',Name_batch{i},'/imdb.mat'];
        if(~exist(filename,'file'))
            IsTrain=true;
            if(strcmp(Name_batch{i},'cub200'))
                imdb=getImdb_cub200(Name_batch{i},conf,net.meta,IsTrain);
            else
                imdb=getImdb(Name_batch{i},conf,net.meta,IsTrain);
            end
            mkdir(['./mat/',Name_batch{i}]);
            save(filename,'-struct','imdb');
            clear imdb;
        end
    end
    imdb=produceIMDB(labelNum,net,Name_batch,conf);
    mkdir(opts.expDir);
    save(opts.imdbPath,'-struct','imdb');
end  

net.meta.classes.name=imdb.meta.classes(:)';

%% Train

[net,info]=our_cnn_train(net,imdb,getBatch(opts),'expDir',opts.expDir,net.meta.trainOpts,opts.train,'val',find(imdb.images.set==2),'prefetch',true);
end


function fn = getBatch(opts)
opts.numGpus = numel(opts.train.gpus) ;
fn = @(x,y,z) getSimpleBatch(opts,x,y,z) ;
end


function [images,labels]=getSimpleBatch(opts, imdb, batch, isPrefetch)
if isfield(imdb.images,'data')
    images = imdb.images.data(:,:,:,batch);
    if opts.numGpus > 0
      images = gpuArray(images) ;
    end
else
    images_dir = imdb.images.dir(batch);
    if ~isPrefetch
    images = vl_imreadjpeg(images_dir, ...
           'NumThreads', 10, ...
           'Pack', ...
           'GPU', ...
           'Interpolation', 'bilinear', ...
           'Resize', opts.imageSize(1:2),...
           'SubtractAverage', imdb.meta.dataMean...
       );
    images = images{1};
    else
    vl_imreadjpeg(images_dir, ...
           'NumThreads', 10, ...
           'Pack', ...
           'GPU', ...
           'Interpolation', 'bilinear', ...
           'Resize', opts.imageSize(1:2),...
           'SubtractAverage', imdb.meta.dataMean,...
           'Prefetch' ...
       );
    images = [];
    end
end
labels = reshape(imdb.images.labels(:,batch),[1,1,size(imdb.images.labels,1),numel(batch)]);
end

function imdb=produceIMDB(labelNum,net,Name_batch,theConf)
trainRate=0.9;
if(labelNum>10)
    maxSampleNum=400;
    minSampleNum=100;
else
    maxSampleNum=1000000;
    minSampleNum=1500;
end
for i=1:labelNum
    tmpimdb=load(sprintf('./mat/%s/imdb.mat',Name_batch{i}));
    tmpimdb.images.data=tmpimdb.images.data(:,:,:,tmpimdb.images.order);
    tmpimdb.images.labels=tmpimdb.images.labels(:,tmpimdb.images.order);
    list=find(tmpimdb.images.labels==1);
    objset=readAnnotation(Name_batch{i},theConf);
    list=reshape(list(1:numel(objset)*2),[1,numel(objset)*2]);
    if(numel(list)<minSampleNum)
        list=repmat(list,[1,ceil(minSampleNum/numel(list))]);
        list=list(1:minSampleNum);
    end
    list=list(1:min(numel(list),maxSampleNum));
    tmpimdb.images.data=tmpimdb.images.data(:,:,:,list);
    tmpimdb.images.labels=tmpimdb.images.labels(:,list);
    imgnum=numel(list);
    tmpimdb.images.data=bsxfun(@plus,tmpimdb.images.data,repmat(tmpimdb.meta.dataMean,[1,1,1,imgnum]));
    if(i==1)
        imdb=tmpimdb;
        imdb.images.labels=ones(labelNum,imgnum,'single').*(-1);
        imdb.images.labels(1,:)=tmpimdb.images.labels;
    else
        imdb.images.data(:,:,:,end+1:end+imgnum)=tmpimdb.images.data(1:227,1:227,:,:);
        imdb.images.labels(:,end+1:end+imgnum)=-1;
        imdb.images.labels(i,end-imgnum+1:end)=tmpimdb.images.labels;
    end
    clear tmpimdb;
    disp(i);
end
num=size(imdb.images.data,4);
imdb.images.set=zeros(1,num,'uint8');
list_train=round(linspace(1,num,round(num*trainRate)));
imdb.images.set(1:end)=2;
imdb.images.set(list_train)=1;
dataMean=mean(imdb.images.data(:,:,:,imdb.images.set==1),4);
imdb.images.data=bsxfun(@minus,imdb.images.data,dataMean);
imdb.images.order=1:num;
imdb.meta.classes=Name_batch;
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
