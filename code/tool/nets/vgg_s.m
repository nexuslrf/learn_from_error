function net = vgg_s(labelNum,lossWeight,opts)
channel_num=512;
mag=0.1;
netname='vgg-s';

partRate=1;
textureRate=0;
output_num=labelNum;
net=load('../nets/imagenet-vgg-s.mat');

net.layers=net.layers(1:16);
net = add_dropout(net, opts, '6') ;
net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, output_num, 1, 0) ;
net.layers(end) = [] ;
net.meta.classes.name={'pos'};
net.meta.classes.description={'pos'};
net.meta.normalization.imageSize=net.meta.normalization.imageSize(1:3);
net.meta.netname=netname;
end
