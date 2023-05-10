##Assignment 2
##

import filters
import metrics
import noise
import numpy as np
import matplotlib.pyplot as plt
import os

input_video="data/Saint_Barthelemy_2.mov"
noisy_video = "results/noisy_video.mp4"
bilateral_filtered="results/bilateral.mp4"
temporal_filtered="results/temporal.mp4"
median_filtered="results/median.mp4"
gaussian_filtered="results/gaussian.mp4"
RBLT_filtered="results/RBLT.mp4"

noise_type = "gaussian"  # Choose between "gaussian" or "sp"

if not os.path.exists(noisy_video):
    noise.add_noise_to_video(input_video,noisy_video, noise_type)

# Calculate PSNR
psnr_values, avg_psnr = metrics.calculate_psnr(noisy_video, input_video)
print("Average PSNR:", avg_psnr)

# Plotting
frame_indices = np.arange(len(psnr_values))
plt.plot(frame_indices, psnr_values)
plt.xlabel("Frame")
plt.ylabel("PSNR")
plt.title("PSNR Variation over Time")
plt.grid(True)
plt.show(block=True)

# Calculate SSIM
if not os.path.exists("results/ssim_values_noisy.txt"):
    ssim_values_noisy = np.array(metrics.calculate_video_ssim(noisy_video, input_video))
    np.savetxt("results/ssim_values_noisy.txt", ssim_values_noisy)
else:
    ssim_values_noisy = np.loadtxt("results/ssim_values_noisy.txt")

# Plotting SSIM values
frame_numbers = np.arange(len(ssim_values_noisy))
plt.plot(frame_numbers, ssim_values_noisy)
plt.xlabel('Frame')
plt.ylabel('SSIM')
plt.title('SSIM Values')
plt.show(block=False)

if not os.path.exists(bilateral_filtered):
    filters.bilateral_denoise_video(noisy_video, bilateral_filtered)

if not os.path.exists("results/ssim_values_bf.txt"):
    ssim_values_bf = np.array(metrics.calculate_video_ssim(bilateral_filtered, input_video))
    np.savetxt("results/ssim_values_bf.txt", ssim_values_bf)
else:
    ssim_values_bf = np.loadtxt("results/ssim_values_bf.txt")

frame_numbers = np.arange(len(ssim_values_bf))
plt.plot(frame_numbers, ssim_values_bf)
plt.xlabel('Frame')
plt.ylabel('SSIM')
plt.title('SSIM Values')
plt.show(block=False)

if not os.path.exists(temporal_filtered):
    filters.bilateral_denoise_video(noisy_video, temporal_filtered)

if not os.path.exists(median_filtered):
    filters.median_filter_to_video(noisy_video, median_filtered)

if not os.path.exists(gaussian_filtered):
    filters.gaussian_filter_to_video(noisy_video, gaussian_filtered)    

if not os.path.exists(RBLT_filtered):
    filters.RBLT(noisy_video, RBLT_filtered)

