##Assignment 2
##

import filters
import metrics
import noise
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time

#Video paths
input_video="data/Saint_Barthelemy_2.mov"
noisy_video = "results/noisy_video.mp4"
bilateral_filtered="results/bilateral.mp4"
temporal_filtered="results/temporal.mp4"
median_filtered="results/median.mp4"
gaussian_filtered="results/gaussian.mp4"
RBLT_filtered="results/RBLT.mp4"

# Add noise to video
noise_type = "gaussian"  # Choose between "gaussian" or "sp"
if not os.path.exists(noisy_video):
    noise.add_noise_to_video(input_video,noisy_video, noise_type)

# Calculate PSNR
psnr_values, avg_psnr = metrics.calculate_psnr(noisy_video, input_video)
print("Average PSNR:", avg_psnr)

# Plotting
# frame_indices = np.arange(len(psnr_values))
# plt.plot(frame_indices, psnr_values)
# plt.xlabel("Frame")
# plt.ylabel("PSNR")
# plt.title("PSNR Variation over Time")
# plt.grid(True)
# plt.show(block=False)

# Calculate SSIM
if not os.path.exists("results/ssim_values_noisy.txt"):
    ssim_values_noisy = np.array(metrics.calculate_video_ssim(noisy_video, input_video))
    np.savetxt("results/ssim_values_noisy.txt", ssim_values_noisy)
else:
    ssim_values_noisy = np.loadtxt("results/ssim_values_noisy.txt")

# Plotting SSIM values
# frame_numbers = np.arange(len(ssim_values_noisy))
# plt.plot(frame_numbers, ssim_values_noisy)
# plt.xlabel('Frame')
# plt.ylabel('SSIM')
# plt.title('SSIM Values')
# plt.show(block=False)

if not os.path.exists(bilateral_filtered):
    filters.bilateral_denoise_video(noisy_video, bilateral_filtered)

if not os.path.exists("results/ssim_values_bf.txt"):
    ssim_values_bf = np.array(metrics.calculate_video_ssim(bilateral_filtered, input_video))
    np.savetxt("results/ssim_values_bf.txt", ssim_values_bf)
else:
    ssim_values_bf = np.loadtxt("results/ssim_values_bf.txt")

# frame_numbers = np.arange(len(ssim_values_bf))
# plt.plot(frame_numbers, ssim_values_bf)
# plt.xlabel('Frame')
# plt.ylabel('SSIM')
# plt.title('SSIM Values')
# plt.show(block=False)

if not os.path.exists(temporal_filtered):
    filters.bilateral_denoise_video(noisy_video, temporal_filtered)

if not os.path.exists(median_filtered):
    filters.median_filter_to_video(noisy_video, median_filtered)

if not os.path.exists(gaussian_filtered):
    filters.gaussian_filter_to_video(noisy_video, gaussian_filtered) 

if not os.path.exists(temporal_filtered):
    filters.apply_temporal_filter(noisy_video, temporal_filtered)   

if not os.path.exists(bilateral_filtered):
    filters.bilateral_denoise_video(noisy_video, bilateral_filtered)

if not os.path.exists(RBLT_filtered):
    filters.RBLT_filter(noisy_video, RBLT_filtered)


#Metrics

#Noisy
    #PSNR
if not os.path.exists("results/psnr_values_noisy.txt"):
    psnr_values_noisy, avg_psnr_noisy = metrics.calculate_psnr(noisy_video, input_video)
    np.savetxt("results/psnr_values_noisy.txt", psnr_values_noisy)
else:
    psnr_values_noisy = np.loadtxt("results/psnr_values_noisy.txt")

    #SSIM
if not os.path.exists("results/ssim_values_noisy.txt"):
    ssim_values_noisy, avg_ssim_noisy = metrics.calculate_ssim(noisy_video, input_video)
    np.savetxt("results/ssim_values_noisy.txt", ssim_values_noisy)
else:
    ssim_values_noisy = np.loadtxt("results/ssim_values_noisy.txt")


#Median 

    #PSNR
if not os.path.exists("results/psnr_values_median.txt"):
    psnr_values_median, avg_psnr_median = metrics.calculate_psnr(median_filtered, input_video)
    np.savetxt("results/psnr_values_median.txt", psnr_values_median)
else:
    psnr_values_median = np.loadtxt("results/psnr_values_median.txt")

    #SSIM
if not os.path.exists("results/ssim_values_median.txt"):
    ssim_values_median, avg_ssim_median = metrics.calculate_ssim(median_filtered, input_video)
    np.savetxt("results/ssim_values_median.txt", ssim_values_median)
else:
    ssim_values_median = np.loadtxt("results/ssim_values_median.txt")


#Gauss

    #PSNR
if not os.path.exists("results/psnr_values_gaussian.txt"):
    psnr_values_gaussian, avg_psnr_gaussian = metrics.calculate_psnr(gaussian_filtered, input_video)
    np.savetxt("results/psnr_values_gaussian.txt", psnr_values_gaussian)
else:
    psnr_values_gaussian = np.loadtxt("results/psnr_values_gaussian.txt")

    #SSIM
if not os.path.exists("results/ssim_values_gaussian.txt"):
    ssim_values_gaussian, avg_ssim_gaussian = metrics.calculate_ssim(gaussian_filtered, input_video)
    np.savetxt("results/ssim_values_gaussian.txt", ssim_values_gaussian)
else:
    ssim_values_gaussian = np.loadtxt("results/ssim_values_gaussian.txt")

#Temp
    #PSNR
if not os.path.exists("results/psnr_values_temporal.txt"):
    psnr_values_temporal, avg_psnr_temporal = metrics.calculate_psnr(temporal_filtered, input_video)
    np.savetxt("results/psnr_values_temporal.txt", psnr_values_temporal)
else:
    psnr_values_temporal = np.loadtxt("results/psnr_values_temporal.txt")

    #SSIM
if not os.path.exists("results/ssim_values_temporal.txt"):
    ssim_values_temporal, avg_ssim_temporal = metrics.calculate_ssim(temporal_filtered, input_video)
    np.savetxt("results/ssim_values_temporal.txt", ssim_values_temporal)
else:
    ssim_values_temporal = np.loadtxt("results/ssim_values_temporal.txt")

#Bilat
    #PSNR
if not os.path.exists("results/psnr_values_bilateral.txt"):
    psnr_values_bilateral, avg_psnr_bilateral = metrics.calculate_psnr(bilateral_filtered, input_video)
    np.savetxt("results/psnr_values_bilateral.txt", psnr_values_bilateral)
else:
    psnr_values_bilateral = np.loadtxt("results/psnr_values_bilateral.txt")

    #SSIM
if not os.path.exists("results/ssim_values_bilateral.txt"):
    ssim_values_bilateral, avg_ssim_bilateral = metrics.calculate_ssim(bilateral_filtered, input_video)
    np.savetxt("results/ssim_values_bilateral.txt", ssim_values_bilateral)
else:
    ssim_values_bilateral= np.loadtxt("results/ssim_values_bilateral.txt")

#RBLT
    #PSNR
if not os.path.exists("results/psnr_values_rblt.txt"):
    psnr_values_rblt, avg_psnr_rblt = metrics.calculate_psnr(RBLT_filtered, input_video)
    np.savetxt("results/psnr_values_rblt.txt", psnr_values_rblt)
else:
    psnr_values_rblt = np.loadtxt("results/psnr_values_rblt.txt")

    #SSIM
if not os.path.exists("results/ssim_values_rblt.txt"):
    ssim_values_rblt, avg_ssim_rblt = metrics.calculate_ssim(RBLT_filtered, input_video)
    np.savetxt("results/ssim_values_rblt.txt", ssim_values_rblt)
else:
    ssim_values_rblt = np.loadtxt("results/ssim_values_rblt.txt")


video = cv2.VideoCapture(noisy_video)
_, img = video.read()

before = time.time()
result = filters.rbl(img,3)
print('finished in ' + str(time.time()-before) + 'seconds')
cv2.imshow('amogus', result)
cv2.waitKey(0)
# if cv2.waitKey(0) == ord('q'):
#     cv2.destroyAllWindows()
#     exit()
