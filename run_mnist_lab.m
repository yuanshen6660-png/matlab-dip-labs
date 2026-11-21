function run_mnist_lab(epochs)
% Compare one and two convolution blocks on the official MNIST split.
if nargin < 1, epochs = 3; end
validateattributes(epochs,{'numeric'},{'scalar','integer','positive'});
root = fileparts(mfilename('fullpath'));
out = fullfile(root,'results','lab06');
data = fullfile(root,'data','MNIST');
if ~isfolder(out), mkdir(out); end
if ~isfolder(data), mkdir(data); end
classes = string(0:9);
X = read_idx(data,'train-images-idx3-ubyte',2051,60000);
T = categorical(read_idx(data,'train-labels-idx1-ubyte',2049,60000),0:9,cellstr(classes));
XTest = read_idx(data,'t10k-images-idx3-ubyte',2051,10000);
TTest = categorical(read_idx(data,'t10k-labels-idx1-ubyte',2049,10000),0:9,cellstr(classes));
rng(2403,'twister');
order = randperm(numel(T));
val = order(1:6000); train = order(6001:end);
batch = 128;
options = trainingOptions('adam',MaxEpochs=epochs,MiniBatchSize=batch, ...
    InitialLearnRate=0.001,Shuffle='every-epoch',ExecutionEnvironment='cpu', ...
    ValidationData={X(:,:,:,val),T(val)}, ...
    ValidationFrequency=ceil(numel(train)/batch),Metrics='accuracy', ...
    Verbose=true,Plots='none',OutputNetwork='last-iteration');
training = cell(1,2); validation = cell(1,2); comparisons = cell(1,2);
for depth = 1:2
    rng(2403,'twister');
    layers = [imageInputLayer([28,28,1],Normalization='none')
              convolution2dLayer(3,8,Padding='same')
              reluLayer
              maxPooling2dLayer(2,Stride=2)];
    if depth == 2
        layers = [layers
                  convolution2dLayer(3,16,Padding='same')
                  reluLayer
                  maxPooling2dLayer(2,Stride=2)];
    end
    layers = [layers; fullyConnectedLayer(10); softmaxLayer];
    started = tic;
    [net,info] = trainnet(X(:,:,:,train),T(train),layers,'crossentropy',options);
    elapsed = toc(started);
    scores = minibatchpredict(net,XTest,MiniBatchSize=256,ExecutionEnvironment='cpu');
    predicted = scores2label(scores,classes);
    isRight = string(predicted(:)) == string(TTest(:));
    truthID = str2double(string(TTest(:)))+1;
    guessID = str2double(string(predicted(:)))+1;
    matrix = accumarray([truthID,guessID],1,[10,10]);
    assert(sum(matrix,'all')==numel(TTest));
    parameters = sum(cellfun(@numel,net.Learnables.Value));
    comparisons{depth} = struct('depth',depth,'test_accuracy_percent',100*mean(isRight), ...
        'parameters',parameters,'train_seconds',elapsed,'confusion',matrix);
    training{depth} = info.TrainingHistory;
    validation{depth} = info.ValidationHistory;
    writetable(training{depth},fullfile(out,sprintf('train_%d.csv',depth)));
    writetable(validation{depth},fullfile(out,sprintf('validation_%d.csv',depth)));
    save(fullfile(out,sprintf('cnn_%d.mat',depth)),'net');
    fprintf('CNN-%d | accuracy %.2f%% | %d parameters | %.2f s\n', ...
        depth,100*mean(isRight),parameters,elapsed);
end
plot_history(training,validation,out);
f = figure('Visible','off','Color','w','Position',[50 50 1000 480]);
tiledlayout(2,6,'Padding','compact','TileSpacing','compact');
mistakes = find(~isRight,6,'first');
right = find(isRight,6,'first');
for n = reshape([right;mistakes],1,[])
    nexttile; imshow(XTest(:,:,1,n));
    title(sprintf('True %d | Pred %d',truthID(n)-1,guessID(n)-1));
end
exportgraphics(f,fullfile(out,'predictions.png'),'Resolution',180); close(f);
result = struct('matlab_release',version('-release'),'backend','CPU', ...
    'seed',2403,'epochs',epochs,'train_samples',numel(train), ...
    'validation_samples',numel(val),'test_samples',numel(TTest), ...
    'batch_size',batch,'learning_rate',0.001,'models',[comparisons{:}]);
file = fopen(fullfile(out,'metrics.json'),'w','n','UTF-8');
assert(file >= 0, 'Could not create metrics.json.');
fprintf(file,'%s',jsonencode(result,'PrettyPrint',true)); fclose(file);
end

function data = read_idx(path,name,expectedMagic,expectedCount)
file = fullfile(path,name);
if ~isfile(file)
    compressed = [file '.gz'];
    if ~isfile(compressed)
        link = ['https://storage.googleapis.com/cvdf-datasets/mnist/' name '.gz'];
        websave(compressed,link,weboptions('Timeout',60));
    end
    gunzip(compressed,path);
end
fid = fopen(file,'r','ieee-be');
assert(fid>=0,'Could not read %s.',name);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
magic = fread(fid,1,'uint32'); n = fread(fid,1,'uint32');
assert(magic==expectedMagic && n==expectedCount,'Unexpected IDX header: %s.',name);
if magic==2051
    rows = fread(fid,1,'uint32'); cols = fread(fid,1,'uint32');
    assert(rows==28 && cols==28,'Unexpected image size.');
    pixels = fread(fid,rows*cols*n,'*uint8');
    assert(numel(pixels)==rows*cols*n,'Truncated image file.');
    data = single(permute(reshape(pixels,[cols,rows,1,n]),[2,1,3,4]))/255;
else
    data = fread(fid,n,'*uint8');
    assert(numel(data)==n && all(data<=9),'Invalid labels.');
end
assert(isempty(fread(fid,1,'*uint8')),'Unexpected bytes after IDX data.');
end

function plot_history(training,validation,out)
f = figure('Visible','off','Color','w','Position',[50 50 1080 700]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
for depth = 1:2
    tr = training{depth}; va = validation{depth};
    nexttile(depth); hold on;
    plot(tr.Iteration,table_column(tr,{'Loss','TrainingLoss'}),'DisplayName','Training');
    plot(va.Iteration,table_column(va,{'Loss','ValidationLoss'}),'-o','DisplayName','Validation');
    title(sprintf('CNN-%d loss',depth)); xlabel('Iteration'); legend; grid on;
    nexttile(depth+2); hold on;
    plot(tr.Iteration,table_column(tr,{'Accuracy','TrainingAccuracy'}),'DisplayName','Training');
    plot(va.Iteration,table_column(va,{'Accuracy','ValidationAccuracy'}),'-o','DisplayName','Validation');
    title(sprintf('CNN-%d accuracy',depth)); xlabel('Iteration'); ylabel('Accuracy (%)'); legend; grid on;
end
exportgraphics(f,fullfile(out,'training_curves.png'),'Resolution',180); close(f);
end

function values = table_column(data,candidates)
for k = 1:numel(candidates)
    if any(strcmp(data.Properties.VariableNames,candidates{k}))
        values = data.(candidates{k});
        return
    end
end
error('The MATLAB training history does not contain %s.',strjoin(candidates,', '));
end
