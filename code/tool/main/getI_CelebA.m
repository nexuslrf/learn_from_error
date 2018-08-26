function [I_patch,I]=getI_CelebA(i,theConf,tarSize,IsFlip)
filename=[theConf.data.imgdir,'/',sprintf('%06d.jpg',i)];
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
xmin=1;
xmax=w;
ymin=1;
ymax=h;
I_patch=I(ymin:ymax,xmin:xmax,:);
I_patch=single(I_patch); % note: 0-255 range
I_patch=imresize(I_patch,tarSize,'bilinear');
if(IsFlip)
    I_patch=I_patch(:,end:-1:1,:);
    I=I(:,end:-1:1,:);
end
