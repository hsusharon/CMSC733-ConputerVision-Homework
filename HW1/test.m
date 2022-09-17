
img_path = 'edge_detection_inputs/3096.jpg'
img = imread(img_path);
image(img)

blurred_img = imgaussfilt(img, 3);
R_comp = blurred_img(:,:,1);
G_comp = blurred_img(:,:,2);
B_comp = blurred_img(:,:,3);