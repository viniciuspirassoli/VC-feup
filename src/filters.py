import cv2

# Apply bilateral filter to video
def bilateral_denoise_video(input_path, output_path, d = 10, sigmaColor = 20, sigmaSpace = 30):
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
        denoised_frame = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)

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

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        output.write(filtered_frame)
    
    output.release()
    cap.release()
    cv2.destroyAllWindows()

def filtro_do_mano():
    return None

import cv2
import numpy as np

def RBLT(input_filename, output_filename, alpha=1.0, beta=1.0/30, gamma=0.01, epsilon=0.01, d=15, sigma_color=75, sigma_space=75, m_estimator='charbonnier'):
    # Load input video
    cap = cv2.VideoCapture(input_filename)

    # Initialize variables
    prev_frame = None
    filtered_frames = []

    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply the filter to the current frame
        if prev_frame is not None:
            # Compute the spatial and temporal differences
            dx = cv2.bilateralFilter(gray, d, sigma_color, sigma_space) - gray
            dt = gray.astype(np.float32) - prev_frame.astype(np.float32)

            # Compute the error norm and weight function
            if m_estimator == 'charbonnier':
                # Compute the Charbonnier error norm
                e = np.sqrt(dx**2 + gamma**2) / epsilon

                # Compute the M-estimator weight function
                w = 1 / (e + 1)
            elif m_estimator == 'geman-mcclure':
                # Compute the Geman-McClure error norm
                e = (dx**2 + gamma**2) / (epsilon**2 + dx**2 + gamma**2)

                # Compute the M-estimator weight function
                w = epsilon**2 / (e + epsilon**2)

            # Apply the filter to the current frame
            filtered = gray + alpha * w * dx + beta * w * dt

            # Clip the filtered values to the range [0, 255]
            filtered = np.clip(filtered, 0, 255)

            # Convert the filtered frame back to uint8
            filtered = filtered.astype(np.uint8)

            # Add the filtered frame to the list of filtered frames
            filtered_frames.append(filtered)

        # Update the previous frame
        prev_frame = gray

    # Release the video capture object
    cap.release()

    # Save the filtered video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30, (gray.shape[1], gray.shape[0]))
    for frame in filtered_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    out.release()
