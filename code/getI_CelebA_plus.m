function [I_dir,I_patch,I]=getI_CelebA_plus(i,theConf,tarSize,bbox,IsFlip)
filename=[theConf.data.imgdir,'/',sprintf('%06d.jpg',i)];
filecrop=[theConf.data.imgcrop,'/',sprintf('%06d.jpg',i)];
try
    I=imread(filename);
catch
    disp(filename);
    error('cannot read the above image.');
end
[h,w,d]=size(I);
if(d==1)
    I=repmat(reshape(I,[h,w,1]),[1,1,3]);
end
xmin=max(bbox(1),1);
xmax=min(bbox(3),w);
ymin=max(bbox(2),1);
ymax=min(bbox(4),h);
I_patch=I(ymin:ymax,xmin:xmax,:);
I_patch=single(I_patch); % note: 0-255 range
I_patch=imresize(I_patch,tarSize,'bilinear');
if(IsFlip)
    I_patch=I_patch(:,end:-1:1,:);
    I=I(:,end:-1:1,:);
end
imwrite(uint8(I_patch),filecrop);
I_dir = filecrop;
