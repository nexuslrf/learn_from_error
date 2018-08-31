
dataMean_vgg = zeros(224,224,3);
for i = 1: 512 : 162770
    fprintf('%d/162770\n',i)
    if i+511 <= 162770
        imdirs = params.imdb.images.dir(subset(i:i+511));
    else
        imdirs = params.imdb.images.dir(subset(i:162770));
    end
pts = vl_imreadjpeg(imdirs, ...
           'NumThreads', 15, ...
           'Pack', ...
           'Interpolation', 'bilinear', ...
           'Resize', [224 224]...
       );
pts= pts{1};
dataMean_vgg = dataMean_vgg + sum(pts, 4);
end
dataMean_vgg = dataMean_vgg ./ 162770;
save('dataMean_vgg.mat','dataMean_vgg');
