function objset=readAnnotation(Name_batch,theConf)
MaxObjNum=300000;

minArea=theConf.data.minArea;
objset(MaxObjNum).folder=[];
objset(MaxObjNum).filename=[];
objset(MaxObjNum).name=[];
objset(MaxObjNum).bndbox=[];
objset(MaxObjNum).ID=[];
files=dir(theConf.data.imgdir);
for i=1:length(files)-2
    folder=files;
    filename=sprintf('%05d.jpg',i);
    [h,w,~]=size(imread([files, '/', filename]));
    xmin=1;
    ymin=1;
    xmax=w;
    ymax=h;
    bndbox.xmin=int2str(xmin);
    bndbox.xmax=int2str(xmax);
    bndbox.ymin=int2str(ymin);
    bndbox.ymax=int2str(ymax);
    if(~IsAreaValid(bndbox,minArea))
        continue;
    end
    objset(i).folder=folder;
    objset(i).filename=filename;
    objset(i).name=filename;
    objset(i).bndbox=bndbox;
    objset(i).ID=i;
end
objset=objset(1:length(files)-2);
end


function pd=IsAreaValid(bndbox,minArea)
xmin=str2double(bndbox.xmin);
xmax=str2double(bndbox.xmax);
ymin=str2double(bndbox.ymin);
ymax=str2double(bndbox.ymax);
if((xmax-xmin+1)*(ymax-ymin+1)>=minArea)
    pd=true;
else
    pd=false;
end
end


