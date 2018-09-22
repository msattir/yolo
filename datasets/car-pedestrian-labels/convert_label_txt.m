clear all;
img_dir = '../../labeldata/pos/';

%Get file names
files = dir([img_dir '/L1_*.mat']);

for i=1:size(files)
    L=load([files(i).folder '\' files(i).name]);
    csvwrite([files(i).name(1:end-4) '.txt'],L.L);
end

