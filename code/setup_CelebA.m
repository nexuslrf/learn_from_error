function setup_CelebA()
%% download images

%% settings
conf.data.catedir='../data/CelebA/';
conf.data.imgdir='../data/CelebA/Img/img_celeba';
conf.data.readCode='./data_input/data_input_CelebA/';
conf.data.imgcrop='../data/CelebA/Img/img_crop_celeba';
conf.data.imgmat='../data/CelebA/Img/img_mat_celeba';
if ~exist(conf.data.imgcrop)
    mkdir(conf.data.imgcrop)
end
if ~exist(conf.data.imgmat)
    mkdir(conf.data.imgmat)
end
conf.data.minArea=50^2;
conf.output.dir='./mat/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Name_batch='CelebA';
mkdir([conf.output.dir,Name_batch]);
save([conf.output.dir,Name_batch,'/conf.mat'],'conf');
addpath(genpath('./tool'));
addpath(conf.data.readCode);
end

