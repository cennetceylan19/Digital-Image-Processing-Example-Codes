%% some basic intensity transformation functions
%gray image
OriginalImg = imread('\cat.jpg'); %kendi dosya uzantınızı eklemeyi unutmayın :)
GrayImg = rgb2gray(OriginalImg);
imshow(GrayImg)
%image negative
% s=  (L-1)-img
L=256;
Negativeimg = (L-1) - OriginalImg;
figure, subplot(2,2,1),imshow(OriginalImg),title('Original image')
subplot(2,2,2),imshow(Negativeimg),title('Negative Image')
%log transformation
% s = c*log(1+r)
r = double(OriginalImg)/255;
c = 2;
logimg= c*log(1+r);
subplot(2,2,3),imshow(logimg)
title('Log Transformation Image')
%power-law (gamma) transformation
% s = c*r^g
g = 0.7;
power_lawimg = c*(r.^g);
subplot(2,2,4),imshow(power_lawimg)
title('Power-Law Transformation Image')
%% Histogram processing
OriginalImg = imread('\cat.jpg');
figure, subplot(2,2,1),imshow(OriginalImg) 
subplot(2,2,2),imhist(OriginalImg)
hist_equation_img = histeq(OriginalImg);
subplot(2,2,3),imshow(hist_equation_img)
subplot(2,2,4), imhist(hist_equation_img)
%% Spatial Filters
% Smoothing(Lowpass) Filters
%median filter
OriginalImg = imread('\cat.jpg');
NoiseImg = imnoise(OriginalImg,'salt & pepper',0.2);

% filter each channel separately
r = medfilt2(NoiseImg(:, :, 1), [3 3]);
g = medfilt2(NoiseImg(:, :, 2), [3 3]);
b = medfilt2(NoiseImg(:, :, 3), [3 3]);

% reconstruct the image from r,g,b channels
MedImg = cat(3, r, g, b);

figure,subplot(1,3,1), imshow(OriginalImg), title('Original Image')
subplot(1,3,2), imshow(NoiseImg) , title('Noise Image')
subplot(1,3,3), imshow(MedImg) , title('Median Image')

%Mean(averaring) Filter
f = ones(5,5)/25;
MeanImg = imfilter(GrayImg,f);
figure,subplot(1,2,1), imshow(GrayImg) , title('Gray Image')
subplot(1,2,2), imshow(MeanImg) , title('Mean(averaring) Image')

% Gausian filter
Iblur1 = imgaussfilt(GrayImg,2);
Iblur2 = imgaussfilt(GrayImg,4);
Iblur3 = imgaussfilt(GrayImg,8);
figure, subplot(2,2,1),imshow(GrayImg)
subplot(2,2,2),imshow(Iblur1)
subplot(2,2,3), imshow(Iblur2)
subplot(2,2,4), imshow(Iblur3)

sigma = 1.5;
w = sigma * 6 + 1;
Gaussian_filter= fspecial('gaussian',[w w], sigma);
GaussianImg= imfilter(OriginalImg, Gaussian_filter, 'replicate');
figure, subplot(1,2,1), imshow(OriginalImg),title('Original Image')
subplot(1,2,2), imshow(GaussianImg),title('Gaussian Image')

%Sharpening(Highpass)Filters
SharpImg = imsharpen(OriginalImg);
SharpenedImg = imsharpen(OriginalImg,'Radius',4,'Amount',2);


figure,subplot(1,3,1), imshow(OriginalImg) , title('Original Image')
subplot(1,3,2), imshow(SharpImg) , title('Sharpened1 Image')
subplot(1,3,3), imshow(SharpenedImg) , title('Sharpened2 Image')

%Laplacian Filter
OriginalImg = imread('\cat.jpg');

Laplacian_filter1=[0 1 0; 1 -4 1; 0 1 0];
r1=conv2(OriginalImg(:, :,1),Laplacian_filter1,'same');
g1=conv2(OriginalImg(:, :,2),Laplacian_filter1,'same');
b1=conv2(OriginalImg(:, :,3),Laplacian_filter1,'same');
LaplacianImg1 = cat(3,r1,g1,b1);
LaplacianImg1=uint8(LaplacianImg1);

Laplacian_filter2 = [-1 -1 -1; -1 8 -1; -1 -1 -1];
r2=conv2(OriginalImg(:, :,1),Laplacian_filter2,'same');
g2=conv2(OriginalImg(:, :,2),Laplacian_filter2,'same');
b2=conv2(OriginalImg(:, :,3),Laplacian_filter2,'same');
LaplacianImg2 = cat(3,r2,g2,b2);
LaplacianImg2=uint8(LaplacianImg2);

figure,subplot(1,3,1), imshow(OriginalImg) , title('Original Image')
subplot(1,3,2),imshow(abs(OriginalImg-LaplacianImg1)), title('Laplacian Image1')
subplot(1,3,3),imshow(abs(OriginalImg-LaplacianImg2)), title('Laplacian Image2')

%Sharpen Image
Filter = [0 -1 0; -1 5 -1;0 -1 0];
Sharpened_image= imfilter(OriginalImg, Filter, 'replicate');

figure,subplot(1,2,1);imshow(OriginalImg),title('the original image');
subplot(1,2,2),imshow(Sharpened_image),title('the sharpen image');
%Image Gradiant 
%# apply sobel filter on I with replicate padding
% sobelX and sobelY
Hx = [1 0 -1; 2 0 -2; 1 0 -1] ./ 8.0;
Hy = [1 2 1;0 0 0;-1 -2 -1] ./ 8.0;

% replicate padding
Ix = [GrayImg(:,1) GrayImg(:,1:end) GrayImg(:,end)];
Ixy = [Ix(1,:);Ix(1:end,:); Ix(end,:)];

% apply sobel filter with 2d-convolution and return results
% http://www.mathworks.com/help/matlab/ref/conv2.html
Dx = conv2(Ixy, Hx);
Dx = Dx(3:end-2,3:end-2);
Dy = conv2(Ixy, Hy);
Dy = Dy(3:end-2,3:end-2);
Dxy = sqrt(Dx.^2 + Dy.^2); % look like edges, right?

%# plot
figure;
subplot(2,2,1),imshow(GrayImg),title('the original image');
subplot(2,2,2),imshow(Dx),title('the image gradient Dx');
subplot(2,2,3),imshow(Dy),title('the image gradient Dy');
subplot(2,2,4),imshow(Dxy),title('the image gradient Dxy');

%High Boost Filter
%Formula: 
%HPF = Original image - Low frequency components 
%LPF = Original image - High frequency components 
%HBF = A * Original image - Low frequency components 
%       = (A - 1) * Original image + [Original image - Low frequency components]
%        = (A - 1) * Original image + HPF 
w1 = fspecial('laplacian');
HP = imfilter(OriginalImg,w1,'replicate');
A=1.5;
HBP= (A-1)*(OriginalImg)- HP;
subplot(1,2,1),imshow(OriginalImg);title('Original Image');
subplot(1,2,2),imshow(HBP);title('HighBoost Image');