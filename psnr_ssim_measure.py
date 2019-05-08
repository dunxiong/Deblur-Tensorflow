import os
from skimage.measure import compare_psnr, compare_ssim
from skimage import io
import numpy as np


results_path = 'E:\\yewei_workspace\\deblur\\SRN-Deblur\\imgs\\testing_res_DAVANet_32_3_3'
sharp_path = 'E:\\yewei_workspace\\deblur\\SRN-Deblur\\imgs\\sharp'
result_images = sorted(os.listdir(results_path))
sharp_images = sorted(os.listdir(sharp_path))

num_img = len(result_images)

PSNR = []
SSIM = []

for i in range(num_img):
    sharp_image = io.imread(os.path.join(sharp_path, sharp_images[i]))
    result_image = io.imread(os.path.join(results_path, result_images[i]))

    PSNR.append(compare_psnr(sharp_image, result_image))
    SSIM.append(compare_ssim(sharp_image, result_image, multichannel=True))

mean_psnr = np.mean(PSNR)
mean_ssim = np.mean(SSIM)

print('psnr:%0.4f'%mean_psnr)
print('ssim:%0.4f'%mean_ssim)