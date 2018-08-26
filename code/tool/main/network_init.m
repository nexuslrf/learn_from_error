function net = network_init(labelNum,model,dropoutRate,varargin)
lossWeight=0.02; %0.002;

% CNN_IMAGENET_INIT  Initialize a standard CNN for ImageNet
opts.scale = 1 ;
opts.initBias = 0 ;
opts.weightDecay = 1 ;
%opts.weightInitMethod = 'xavierimproved' ;
opts.weightInitMethod = 'gaussian' ;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1024 ; % 1GB
opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts = vl_argparse(opts, varargin) ;
opts.model=model;

% Define layers
switch opts.model
  case 'alexnet'
    net = alexnet(labelNum,lossWeight,opts) ;
    bs = 128;
    lrMag=0.001*labelNum;
  case 'vgg-vd-16'
    net = vgg_vd_16(labelNum,lossWeight,opts) ;
    bs = 32;
    lrMag = 1*labelNum;
  case 'vgg-m'
    net.meta.normalization.imageSize = [224, 224, 3] ;
    net = vgg_m(labelNum,lossWeight,opts) ;
    bs = 196 ;
    lrMag=30*labelNum;
  case 'vgg-s'
    net.meta.normalization.imageSize = [224, 224, 3] ;
    net = vgg_s(labelNum,lossWeight,opts) ;
    bs = 128 ;
    lrMag=10*labelNum;
  otherwise
    error('Unknown model ''%s''', opts.model) ;
end

% final touches
switch lower(opts.weightInitMethod)
  case {'xavier', 'xavierimproved'}
    net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end

net.layers{end+1} = struct('type', 'ourloss_logistic', 'name', 'loss') ;
% Meta parameters
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
net.meta.normalization.averageImage = opts.averageImage ;
net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions;
net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
net.meta.augmentation.jitterAspect = [2/3, 3/2] ;

if ~opts.batchNormalization
  lr = [logspace(-2, -4, 60),ones(1,40)*1e-4] ;
else
  lr = logspace(-1, -4, 20) ;
end

if(labelNum==1)
    lr=[logspace(-4,-5,10),logspace(-5,-5,50)];
    lr=lr(1:50)./lrMag; %% / changed to *
else
    lr=logspace(-4,-5,60)./lrMag;
end

net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = bs ;
net.meta.trainOpts.weightDecay = 0.0005 ;

for layerID=1:numel(net.layers)
    if(strcmp(net.layers{layerID}.type,'dropout'))
        net.layers{layerID}.rate=dropoutRate;
    end
end

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = aNet_fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err',dagnn.Loss('loss','binaryerror'),{'prediction','label'},'binaryerr'); %%%%%%%%%%%%%%%%%%%%%%
  otherwise
    assert(false) ;
end
