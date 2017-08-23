clear all;

TEST = importdata('test.csv');
test_d = TEST.data;
test_d = test_d./255;  %scaling the data to 0~1
test_d_flat = reshape(test_d',1,[]);
test_data = reshape(test_d_flat,28,28,1,28000);
data = permute(test_data,[2 1 3 4]);
save('test_data.mat','data');

clear data;


%%=============================================%%

TRAIN = importdata('train.csv');
temp = TRAIN.data;
label_all = temp(:,1);
train_all = temp(:,2:end);
train_all = train_all./255;

train_d = train_all(1:32000,:);
train_d_flat = reshape(train_d',1,[]);
train_data = reshape(train_d_flat,28,28,1,32000);
data = permute(train_data,[2 1 3 4]);
label = label_all(1:32000)';
save('train_data.mat','data','label');

clear data;
clear label;

%%============================================%%

val_d = train_all(32000+1:end,:);
val_d_flat = reshape(val_d',1,[]);
val_data = reshape(val_d_flat,28,28,1,10000);
data = permute(val_data,[2 1 3 4]);
label = label_all(32000+1:end)';
save('val_data.mat','data','label');

