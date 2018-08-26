function setup_CelebA()
%% download images
% if(~exist('../data/CUB_200_2011','dir'))
%     system('wget -O ../CUB.tgz --no-check-certificate http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz');
%     system('tar -xvzf ../CUB.tgz -C ../data/');
%     system('rm ../CUB.tgz');
% end

%% settings
conf.data.catedir='../data/CelebA/';
conf.data.imgdir='../data/CelebA/Img/img_align_celeba';
conf.data.readCode='./data_input/data_input_CelebA/';
conf.data.minArea=50^2;
conf.output.dir='./mat/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Name_batch='CelebA';
mkdir([conf.output.dir,Name_batch]);
save([conf.output.dir,Name_batch,'/conf.mat'],'conf');
addpath(genpath('./tool'));
addpath(conf.data.readCode);
end

