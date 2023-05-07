##Assignment 2
##

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def noisy_gauss(frame, mean=0, std_dev=20):
    """
    Add Gaussian noise to an image.

    Parameters:
    image (numpy.ndarray): The image to add noise to.
    sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
    numpy.ndarray: The noisy image.
    """
    # Create a copy of the original image
    noisy_image = np.copy(frame)

    # Generate Gaussian noise with mean and standard deviation 
    noise = np.random.normal(mean, std_dev, size=frame.shape)

    # Add the noise to the image
    noisy_image = noisy_image.astype(np.float32) + noise.astype(np.float32)

    # Clip the pixel values to the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def noisy_sp(image, prob=0.1):
    """
        Add salt and pepper noise to an image.

        Parameters:
        image (numpy.ndarray): The image to add noise to.
        prob (float): Probability of a pixel being either salt or pepper.

        Returns:
        numpy.ndarray: The noisy image.
        """

    noisy_image = np.copy(image)

    # Generate a random matrix with the same shape as the image
    random_matrix = np.random.uniform(0, 1, size=image.shape)

    # Set the pixels to salt or pepper noise based on the probability
    noisy_image[random_matrix < prob/2] = 0
    noisy_image[random_matrix > 1 - prob/2] = 255

    return noisy_image

###Add noise to video
def add_noise_to_video(video_path, output_path, noise_type="gaussian"):
    # Load video
    video_capture = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video frames
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Apply noise to frame
        if noise_type == "gaussian":
            noisy_frame = noisy_gauss(frame)
        elif noise_type == "sp":
            noisy_frame = noisy_sp(frame)
        else:
            raise ValueError("Invalid noise type specified.")

        # Write noisy frame to output video
        output_video.write(noisy_frame)

        cv2.imshow('Noisy Video', noisy_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()


def calculate_psnr(video_path, reference_path):
    # Load videos
    video_capture = cv2.VideoCapture(video_path)
    reference_capture = cv2.VideoCapture(reference_path)

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate PSNR for each frame
    psnr_values = []
    while video_capture.isOpened() and reference_capture.isOpened():
        ret, frame = video_capture.read()
        ret_ref, frame_ref = reference_capture.read()
        if not ret or not ret_ref:
            break

        # Convert frames to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_ref_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)

        # Calculate MSE (Mean Squared Error)
        mse = np.mean((frame_gray - frame_ref_gray) ** 2)

        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        if mse == 0:
            psnr = float('inf')
        else:
            max_intensity = np.max(frame_gray)
            psnr = 10 * np.log10((max_intensity ** 2) / mse)

        psnr_values.append(psnr)

    # Release resources
    video_capture.release()
    reference_capture.release()

    # Calculate average PSNR
    avg_psnr = np.mean(psnr_values)

    return psnr_values, avg_psnr

# SSIM metric
def calculate_video_ssim(video_path, reference_path):
    cap_video = cv2.VideoCapture(video_path)
    cap_reference = cv2.VideoCapture(reference_path)
    num_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    ssim_values = []

    for _ in range(num_frames):
        ret_video, frame_video = cap_video.read()
        ret_reference, frame_reference = cap_reference.read()

        if not (ret_video and ret_reference):
            break

        frame_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)
        frame_reference = cv2.cvtColor(frame_reference, cv2.COLOR_BGR2RGB)

        ssim_value = ssim(frame_video, frame_reference, multichannel=True)
        ssim_values.append(ssim_value)

    cap_video.release()
    cap_reference.release()

    return ssim_values

# Apply biletaral filter to video
def bilateral_denoise_video(input_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(input_path)

    # Get the video's properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the denoised video
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Loop through each frame of the video
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Apply bilateral filtering for noise reduction
        denoised_frame = cv2.bilateralFilter(frame, d, 20, 30)

        # Write the denoised frame to the output video
        output.write(denoised_frame)

        # Display the denoised frame
        cv2.imshow('Denoised Frame', denoised_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    video.release()
    output.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

#Appy temporal filter to video
def apply_temporal_filter(video_path, output_path, window_size):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        if len(frames) > window_size:
            frames.pop(0)
        
        if len(frames) == window_size:
            filtered_frame = cv2.medianBlur(frames[window_size // 2], window_size)
            cv2.imshow("Filtered Frame", filtered_frame)
            cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()



# Add noise to video
#input_video = "210329_06A_Bali_4K_004.mp4"
#input_video="mlky_6.mp4"
input_video="Saint_Barthelemy_2.mov"
noisy_video = "noisy_video.mp4"
bilateral_filtered="bilateral.mp4"

noise_type = "gaussian"  # Choose between "gaussian" or "sp"

add_noise_to_video(input_video,noisy_video, noise_type)



# Calculate PSNR
psnr_values, avg_psnr = calculate_psnr(noisy_video, input_video)
print("Average PSNR:", avg_psnr)

# Plotting
frame_indices = np.arange(len(psnr_values))
plt.plot(frame_indices, psnr_values)
plt.xlabel("Frame")
plt.ylabel("PSNR")
plt.title("PSNR Variation over Time")
plt.grid(True)
plt.show(block=False)


#Calculate SSIM
# Example usage
#ssim_values = calculate_video_ssim(noisy_video_path, reference_video_path)


# Plotting SSIM values --needs fixing
#frame_numbers = np.arange(len(ssim_values))
#plt.plot(frame_numbers, ssim_values)
#plt.xlabel('Frame')
#plt.ylabel('SSIM')
#plt.title('SSIM Values')
#plt.show(block=False)

# Example for filter 
d = 10
sigma_color = 75
sigma_space = 75

bilateral_denoise_video(noisy_video, bilateral_filtered)


#Example window size for temporal filter
#window_size = 5