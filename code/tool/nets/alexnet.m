function net = alexnet(labelNum,lossWeight,opts)
netname='alexnet';
output_num=labelNum;
alexnet=load('../nets/imagenet-caffe-alex.mat');

net.meta=alexnet.meta;
for i = 1 : 17 %%%%%%%%%%%%%%%%%%%%%%%before fc7
    net.layers{i}=alexnet.layers{i};
end
net = add_dropout(net, opts, 'fc--6') ;
net = add_blockfc(net, opts, '7', 1, 1, 4096,  4096, 1, 0) ;
% origin = [13,16,18]; %%%%%%%%%%%%%%%%% conv5, fc6,7,8;
% add = [13,16,19];
%   for i= 1:3
%         net.layers{add(i)}.alpha = 1;
%         net.layers{add(i)}.type='our_conv';
%         net.layers{add(i)}.weights=alexnet.layers{origin(i)}.weights;
%         ts=alexnet.layers{origin(i)}.size;
%         %tmp_weights_added={init_weight(opts, ts(1),ts(2),ts(3),ts(4) , 'single'), ...
%         %                      ones(net.layers{i}.size(4), 1, 'single')*opts.initBias};
%         tmp_weights_added = net.layers{add(i)}.weights;
%         net.layers{add(i)}.weights = {cat(3,net.layers{add(i)}.weights{1},tmp_weights_added{1}),...
%                               cat(2,net.layers{add(i)}.weights{2},tmp_weights_added{2})};
%         net.layers{add(i)}.addnum = 1;
%   end
net = add_dropout(net, opts, 'fc--7') ;
net = add_blockfc(net, opts, '8', 1, 1, 4096,  output_num, 1, 0) ;

net.layers(end) = [] ;
%net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
net.meta.classes.name={'pos'};
net.meta.classes.description={'pos'};
net.meta.normalization.imageSize=net.meta.normalization.imageSize(1:3);
net.meta.netname=netname;
clear alexnet tmp_weights_added;
end
