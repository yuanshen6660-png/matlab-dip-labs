function run_image_labs(labs, interactive)
% Run Digital Image Processing labs 1 to 5 in MATLAB.
% run_image_labs(1,true) opens an image picker for Lab 1.
if nargin < 1, labs = 1:5; end
if nargin < 2, interactive = false; end
validateattributes(labs, {'numeric'}, {'integer','>=',1,'<=',5});
root = fullfile(fileparts(mfilename('fullpath')), 'results');
steps = {@histogram_lab, @fourier_lab, @filter_lab, @edge_lab, @watershed_lab};
for n = labs(:)'
    output = fullfile(root, sprintf('lab%02d',n));
    if ~isfolder(output), mkdir(output); end
    metrics = steps{n}(output, interactive);
    file = fopen(fullfile(output,'metrics.json'),'w','n','UTF-8');
    assert(file >= 0, 'Could not create metrics.json.');
    fprintf(file,'%s',jsonencode(metrics,'PrettyPrint',true));
    fclose(file);
    fprintf('Lab %d finished: %s\n',n,output);
end
end

function metrics = histogram_lab(out, interactive)
source = 'moon.tif';
if interactive
    [name,folder] = uigetfile({'*.png;*.jpg;*.tif;*.bmp','Image files'}, ...
        'Choose an image, or cancel to use moon.tif');
    if ~isequal(name,0), source = fullfile(folder,name); end
end
[input, map] = imread(source);
if ~isempty(map), input = ind2gray(input,map); end
I = im2uint8(im2gray(input));
before = imhist(I);
cdf = cumsum(before)/numel(I);
first = find(before,1);
if cdf(first) < 1
    lookup = uint8(round(255*max(0,(cdf-cdf(first))/(1-cdf(first)))));
    J = reshape(lookup(double(I)+1),size(I));
else
    J = I;
end
reference = histeq(I,256);
after = imhist(J);
f = make_figure('Histogram equalization',interactive);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
show_image(I,'Input'); show_image(J,'CDF equalization'); show_image(reference,'MATLAB histeq');
nexttile; bar(0:255,before,1); xlim([0,255]); title('Input histogram'); xlabel('Gray level');
nexttile; bar(0:255,after,1); xlim([0,255]); title('CDF histogram'); xlabel('Gray level');
nexttile; bar(0:255,imhist(reference),1); xlim([0,255]); title('histeq histogram'); xlabel('Gray level');
imwrite(I,fullfile(out,'input.png')); imwrite(J,fullfile(out,'equalized.png'));
save_figure(f,out,'histograms.png',interactive);
metrics = struct('input',source,'input_range',double([min(I(:)),max(I(:))]), ...
    'equalized_range',double([min(J(:)),max(J(:))]), ...
    'input_std',std(double(I(:))),'equalized_std',std(double(J(:))), ...
    'input_levels',nnz(before),'equalized_levels',nnz(after));
end

function metrics = fourier_lab(out, interactive)
[x,y] = meshgrid(0:255);
inputs = {0.5+0.4*cos(2*pi*12*x/256), 0.5+0.4*cos(2*pi*20*y/256)};
names = {'Vertical stripes','Horizontal stripes'};
f = make_figure('Fourier transform',interactive);
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');
errors = zeros(1,2);
peaks = zeros(2,2);
for k = 1:2
    I = inputs{k}; F = fft2(I); Fc = fftshift(F);
    R = real(ifft2(ifftshift(Fc)));
    errors(k) = max(abs(R-I),[],'all');
    magnitude = abs(Fc); magnitude(129,129) = 0;
    [~,index] = max(magnitude(:)); [row,col] = ind2sub(size(Fc),index);
    peaks(k,:) = [col-129,row-129];
    show_image(I,names{k}); show_image(log1p(abs(F)),'Unshifted spectrum');
    show_image(log1p(abs(Fc)),'Centered spectrum'); show_image(R,'Inverse reconstruction');
    imwrite(I,fullfile(out,sprintf('input%d.png',k)));
end
save_figure(f,out,'spectra.png',interactive);
metrics = struct('image_size',[256,256],'periods',[12,20], ...
    'peak_offsets_xy',peaks,'max_abs_reconstruction_error',errors);
end

function metrics = filter_lab(out, interactive)
rng(83,'twister');
[x,y] = meshgrid(1:256);
I = 0.22+0.20*x/256;
I((x-87).^2+(y-85).^2<=43^2) = 0.75;
I(x>=145 & x<=224 & y>=145 & y<=221) = 0.82;
I(x>=38 & x<=115 & y>=158 & y<=222) = 0.43;
I(x>=38 & x<=115 & y>=158 & y<=222 & mod(x,10)<5) = 0.63;
inputs = {imnoise(I,'salt & pepper',0.06),imnoise(I,'gaussian',0,0.006)};
titles = {'Salt and pepper','Gaussian noise'};
smooth = fspecial('average',[5 5]);
f = make_figure('Image smoothing',interactive);
tiledlayout(2,5,'Padding','compact','TileSpacing','compact');
scores = zeros(2,4);
for k = 1:2
    N = inputs{k};
    outputs = {N,imfilter(N,smooth,'symmetric'),medfilt2(N,[5 5],'symmetric'), ...
        imgaussfilt(N,1,'FilterSize',5,'Padding','symmetric')};
    show_image(I,'Clean image');
    labels = {titles{k},'Mean 5x5','Median 5x5','Gaussian 5x5'};
    for j = 1:4
        scores(k,j) = 10*log10(1/mean((outputs{j}-I).^2,'all'));
        show_image(outputs{j},sprintf('%s | %.2f dB',labels{j},scores(k,j)));
    end
    imwrite(N,fullfile(out,sprintf('noisy%d.png',k)));
end
imwrite(I,fullfile(out,'input.png'));
save_figure(f,out,'filters.png',interactive);
metrics = struct('seed',83,'impulse_density',0.06,'gaussian_variance',0.006, ...
    'window_size',[5,5],'gaussian_sigma',1,'psnr_db',scores);
end

function metrics = edge_lab(out, interactive)
[x,y] = meshgrid(1:256);
I = 0.17+0.12*y/256;
I(x>=38 & x<=83 & y>=39 & y<=218) = 0.78;
I(abs(y-(0.83*x+32))<7 & x>=100 & x<=223) = 0.90;
I((x-178).^2+(y-83).^2<=37^2) = 0.58;
[R,tr] = edge(I,'Roberts'); [S,ts] = edge(I,'Sobel'); [P,tp] = edge(I,'Prewitt');
kernel = [0 1 0; 1 -4 1; 0 1 0];
[L,tl] = edge(I,'zerocross',[],kernel);
response = imfilter(I,kernel,'replicate','conv');
f = make_figure('Edge extraction',interactive);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
show_image(I,'Input'); show_image(R,'Roberts'); show_image(S,'Sobel');
show_image(P,'Prewitt'); show_image(L,'Laplacian zero crossings');
show_image(min(max(I-response,0),1),'Laplacian sharpening');
imwrite(I,fullfile(out,'input.png'));
save_figure(f,out,'edges.png',interactive);
metrics = struct('operator_order',{{'Roberts','Sobel','Prewitt','Laplacian'}}, ...
    'thresholds',[tr,ts,tp,tl],'edge_pixels',[nnz(R),nnz(S),nnz(P),nnz(L)], ...
    'laplacian_kernel',kernel);
end

function metrics = watershed_lab(out, interactive)
rng(52,'twister');
[x,y] = meshgrid(1:320,1:240);
circles = [88 78 38; 150 82 36; 205 163 43; 272 170 36];
mask = false(size(x));
for k = 1:size(circles,1)
    mask = mask | (x-circles(k,1)).^2+(y-circles(k,2)).^2 <= circles(k,3)^2;
end
I = imgaussfilt(0.14+0.10*x/320+0.62*double(mask),0.8);
I = min(max(I+0.013*randn(size(I)),0),1);
B = imopen(imbinarize(I),strel('disk',2));
D = bwdist(~B);
markers = imextendedmax(D,3) & B;
terrain = imimposemin(-D,markers | ~B);
W = watershed(terrain);
boundary = W==0 & B;
labels = W;
labels(~B) = 0; % Keep watershed labels only inside the objects.
foregroundIDs = unique(labels(B & ~boundary));
foregroundIDs(foregroundIDs==0) = [];
rgb = imoverlay(I,boundary,[1 0.1 0.1]);
f = make_figure('Marker-controlled watershed',interactive);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
show_image(I,'Input'); show_image(B,'Otsu + opening'); show_image(D,'Distance transform');
show_image(markers,'Foreground markers');
nexttile; imshow(label2rgb(labels,'jet','k','noshuffle')); title('Watershed labels');
nexttile; imshow(rgb); title('Boundary overlay');
imwrite(I,fullfile(out,'input.png')); imwrite(rgb,fullfile(out,'boundary.png'));
save_figure(f,out,'watershed.png',interactive);
before = bwconncomp(B); seeds = bwconncomp(markers);
metrics = struct('seed',52,'input_objects',size(circles,1),'components_before',before.NumObjects, ...
    'foreground_markers',seeds.NumObjects,'regions_after',numel(foregroundIDs), ...
    'extended_maxima_height',3,'boundary_pixels',nnz(boundary));
end

function f = make_figure(name, interactive)
visibility = 'off';
if interactive, visibility = 'on'; end
f = figure('Name',name,'Color','w','Visible',visibility,'Position',[80 80 1300 680]);
end

function show_image(I,label)
nexttile;
if islogical(I) || isa(I,'uint8') || max(I,[],'all')<=1
    imshow(I);
else
    imshow(I,[]);
end
title(label,'Interpreter','none');
end

function save_figure(f,folder,name,interactive)
exportgraphics(f,fullfile(folder,name),'Resolution',180);
if ~interactive, close(f); end
end
