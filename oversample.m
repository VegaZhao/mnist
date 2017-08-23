%%============================================================%%
% content:1.oversample train set with rotate
%           after that total train data = 64000
%         2.oversample train set with rotate and mirror
%          data = 96000
%         3.oversample train set with rotate and mirror
%         data = 128000
%         4.oversample train set with darker
%         after that total train data = 64000
%
%         then shuffle,when you develop the data which you want 
%         noticing comment and uncomment the relevant codes
% date:2017.8.4
% auther:zwj
%%============================================================%%
clear all;

TRAIN = importdata('train.csv');
temp = TRAIN.data;
label_all = temp(:,1);
train_all = temp(:,2:end);
train_all = train_all./255;

train_d = train_all(1:32000,:);
train_d_flat = reshape(train_d',1,[]);

train_data = reshape(train_d_flat,28,28,1,32000);
train_data_t = permute(train_data,[2 1 3 4]);
train_label = label_all(1:32000)';
%%1.rotate------------------------------------------
% data_cat = cat(4,train_data,train_data_t);
% label_cat = cat(2,train_label,train_label);
%%-----------------------------------------------end

%%2.rotate+mirror-------------------------------------
% i = 1:32000;
% train_data_mr = fliplr(train_data_t(:,:,:,i));
% data_cat = cat(4,train_data_t,train_data,train_data_mr);
% label_cat = cat(2,train_label,train_label,train_label);
%%-----------------------------------------------end

%%3.rotate+mirror+darker----------------------------
% i = 1:32000;
% train_data_mr = fliplr(train_data_t(:,:,:,i));
% train_all_dk = temp(:,2:end);
% train_all_dk = train_all_dk - 15;
% train_all_dk(train_all_dk<0) = 0;
% train_all_dk = train_all_dk./255;
% train_d_dk = train_all_dk(1:32000,:);
% train_d_flat_dk = reshape(train_d_dk',1,[]);
% 
% train_data_dk = reshape(train_d_flat_dk,28,28,1,32000);
% train_data_dk_t = permute(train_data_dk,[2 1 3 4]);
% 
% data_cat = cat(4,train_data_t,train_data,train_data_mr,train_data_dk_t);
% label_cat = cat(2,train_label,train_label,train_label,train_label);
%%-----------------------------------------------end

%%4.darker-----------------------------------------
train_all_dk = temp(:,2:end);
train_all_dk = train_all_dk - 50;
train_all_dk(train_all_dk<0) = 0;
train_all_dk = train_all_dk./255;
train_d_dk = train_all_dk(1:32000,:);
train_d_flat_dk = reshape(train_d_dk',1,[]);

train_data_dk = reshape(train_d_flat_dk,28,28,1,32000);
train_data_dk_t = permute(train_data_dk,[2 1 3 4]);

data_cat = cat(4,train_data_t,train_data_dk_t);
label_cat = cat(2,train_label,train_label);
%%-----------------------------------------------end

%%=======================shuffle============================%%
m=size(label_cat,2);
order = randperm(m);
data = data_cat(:,:,:,order);
label = label_cat(:,order);

%save('train_data_oversample_ot.mat','data','label');
%save('train_data_oversample_ot_mr.mat','data','label');
%save('train_data_oversample_ot_mr_dk.mat','data','label');
save('train_data_oversample_dk.mat','data','label');
