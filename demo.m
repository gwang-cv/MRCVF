% This is a demo for MRCVF 
% Gang Wang 
% //2016

clear all;
close all;

normalize = 1;

filenames1 = 'save_chinese_outlier_2_1.mat';
filenames2 = 'save_chinese_outlier_2_2.mat';

D1=load(filenames1);
D2=load(filenames2);
X=D1.y2a;
Y=D2.y2a;

normal.xm=0; 
normal.ym=0;
normal.xscale=1; 
normal.yscale=1;
if normalize
    [nX, nY, normal]=norm2s(X,Y); 
end
conf = MRCVF_init(conf);
CVF=MRCVF(nX, nY-nX, conf.gamma, conf.beta, conf.lambda, conf.lambda2, conf.theta, conf.a, conf.MaxIter, conf.ecr, conf.minP);
% inliers
CVF.Index