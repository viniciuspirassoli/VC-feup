import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

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
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_ref_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)

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

        frame_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2GRAY)
        frame_reference = cv2.cvtColor(frame_reference, cv2.COLOR_BGR2GRAY)

        ssim_value = ssim(frame_video, frame_reference, multichannel=True)
        ssim_values.append(ssim_value)

    cap_video.release()
    cap_reference.release()

    return ssim_values