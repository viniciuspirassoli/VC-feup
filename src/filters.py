import cv2
import numpy as np
import m_estimators as m

print("cuda devices available: " + str(cv2.cuda.getCudaEnabledDeviceCount()))

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

def median_filter_to_video(input_path,output_path, window_size=3):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get the frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height), isColor=True)

    # Read the first frame
    ret, frame = cap.read()

    # Apply median filter to each frame and write it to the output video
    while ret:
        # Apply median filter
        frame_filtered = cv2.medianBlur(frame, window_size)

        # Write the filtered frame to the output video
        out.write(frame_filtered)

        # Read the next frame
        ret, frame = cap.read()

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    # Print a message when the filtering is done
    print('Median filter applied to the video!')


def gaussian_filter_to_video(input_path, output_path, kernel_size=5, sigmaX=1):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get the frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height), isColor=True)

    # Read the first frame
    ret, frame = cap.read()

    # Apply Gaussian filter to each frame and write it to the output video
    while ret:
        # Apply Gaussian filter
        frame_filtered = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigmaX)

        # Write the filtered frame to the output video
        out.write(frame_filtered)

        # Read the next frame
        ret, frame = cap.read()

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    # Print a message when the filtering is done
    print('Gaussian filter applied to the video!')

def RBLT_filter(input_filename: str, output_filename: str, m_estimator: str = 'charbonnier',
                temporal_window_size: int = 3, spatial_window_size: int = 3):

    video = cv2.VideoCapture(input_filename)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_buffer = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height), isColor=True)

    spatial_kernel = np.zeros((spatial_window_size, spatial_window_size, 3), dtype=float)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        # convert frame to YCrCb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        # add border to frame for convolution
        border_size = (int)((spatial_window_size - 1)/2)
        bordered_frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)

        while (len(frame_buffer) < spatial_window_size):
            frame_buffer.append(bordered_frame)

        print(len(frame_buffer))

        frame_copy = cv2.copyTo(src=bordered_frame, mask=None)
        # if buffer is sufficiently large, start filtering
            # for each pixel in our current frame:
                # generate spatial kernel
                # generate time weights
                # convolve using spatial kernel
                # add time contribution (weighted sum)

        for x in range(border_size, frame.shape[0]):
            for y in range(border_size, frame.shape[1]):

                # generate spatial kernel
                for i in range(spatial_window_size):
                    for j in range(spatial_window_size):
                        # m-estimator(difference) * gaussian(distance)
                        err = bordered_frame[x][y] - bordered_frame[x-border_size+i][y-border_size+j]
                        distance = cv2.norm(bordered_frame[x][y], bordered_frame[x-border_size+i][y-border_size+j])
                        if m_estimator == 'charbonnier':
                            spatial_kernel[i][j] = m.charbonnier(err) * m.gaussian(distance, spatial_window_size)
                        elif m_estimator == 'geman-mcclure':
                            spatial_kernel[i][j] = m.geman_mcclure(err) * m.gaussian(distance, spatial_window_size)

                spatial_kernel = cv2.normalize(spatial_kernel, None, alpha=1, beta=0, norm_type=cv2.NORM_L1)/2

                # generate time weights
                temporal_window = np.zeros((temporal_window_size, 3), dtype=float)
                for n in range(temporal_window_size):
                    if m_estimator == 'charbonnier':
                        temporal_window[n] = m.gaussian(temporal_window_size-n, temporal_window_size) * m.charbonnier(bordered_frame[x][y] - frame_buffer[n][x][y])
                    elif m_estimator == 'geman-mcclure':
                        temporal_window[n]=m.gaussian(temporal_window_size-n, temporal_window_size)*m.geman_mcclure(bordered_frame[x][y][0] - frame_buffer[n][x][y][0])

                temporal_window = cv2.normalize(temporal_window, None, alpha=1, beta=0, norm_type=cv2.NORM_L1)/2

                # apply spatial weights
                pixel = [0, 0, 0]
                for i in range(spatial_kernel.shape[0]):
                    for j in range(spatial_kernel.shape[1]):
                        pixel += spatial_kernel[i][j] * bordered_frame[x-border_size+i][y-border_size+j]
                
                # apply temporal weights
                for n in range(len(temporal_window)):
                    pixel += temporal_window[n] * frame_buffer[n][x][y]

                frame_copy[x][y] = pixel

        out.write(frame_copy)
        frame_buffer.append(bordered_frame)
        frame_buffer.pop(0)
    
    out.release()
    video.release()
    
                            

def robust_bilateral_filter(img, d, sigma_color, sigma_space, sigma_r):
    # Convert image to float32
    img = img.astype(np.float32)

    # Initialize output image
    out = np.zeros_like(img)

    # Compute range kernel
    x, y = np.meshgrid(np.arange(-d, d+1), np.arange(-d, d+1))
    range_kernel = np.exp(-0.5 * (x**2 + y**2) / sigma_r**2)

    # Pad image
    img_padded = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_REFLECT_101)

    # Loop over pixels in output image
    for i in range(d, img.shape[0] + d):
        for j in range(d, img.shape[1] + d):
            # Extract local region
            region = img_padded[i-d:i+d+1, j-d:j+d+1]

            # Compute spatial kernel
            x, y = np.meshgrid(np.arange(-d, d+1), np.arange(-d, d+1))
            spatial_kernel = np.exp(-0.5 * (x**2 + y**2) / sigma_space**2)

            # Compute photometric similarity
            photometric_similarity = np.exp(-0.5 * ((region - img[i-d:i+d+1, j-d:j+d+1])**2) / sigma_color**2)

            # Compute geometric closeness
            geometric_closeness = np.sqrt((x**2 + y**2)) * spatial_kernel

            # Compute weights
            weights = photometric_similarity * geometric_closeness * range_kernel

            # Normalize weights
            weights /= np.sum(weights)

            # Compute filtered pixel value
            out[i-d, j-d] = np.sum(region * weights)

    # Convert output image to uint8
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out




# def RBLT_filter(input_filename: str, output_filename: str, temporal_window_size: int = 3, 
#                 spatial_window_size: int = 3):
#     input = cv2.VideoCapture(input_filename)

#     frame_buffer = []
#     filtered_frames = []

#     while True:
#         ret, frame = input.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
#         frame_buffer.append(frame)
#         if len(frame_buffer) == temporal_window_size +1:
#             # if there are enough frames in the buffer, start filtering
#             break
    
    
#     img = cv2.copyTo(frame_buffer[-1], mask=None)
#     border_size = (int)((spatial_window_size - 1) / 2)
#     img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)

#     for i in range(border_size, img.shape[0] - border_size):
#         for j in range(border_size, img.shape[1] - border_size):
#             kernel = np.zeros((spatial_window_size, spatial_window_size, 3))

#             for x in range(-border_size, border_size+1):
#                 for y in range(-border_size, border_size+1):
#                     diff = img[i,j] - img[i+x, j+y]
#                     distance = cv2.norm(img[i,j], img[i+x, j+y], cv2.NORM_L2)
#                     kernel[x+border_size][y+border_size] = m.charbonnier(diff) + m.gaussian(distance, border_size)
            
#             kernel = cv2.normalize(kernel, None, alpha=1, beta=0, norm_type=cv2.NORM_L1)

#     print(kernel)
            # sum = 0
            # for x in range(kernel.shape[0]):
            #     for y in range(kernel.shape[1]):
                    
            

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





def bilateral_filter(image, diameter, sigma_color, sigma_space):
    "Chat GPT's implementation without using bilateral in opencv"
    # Convert image to grayscale if it's colored
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a padded version of the image for filtering
    padded_image = cv2.copyMakeBorder(image, diameter // 2, diameter // 2, diameter // 2, diameter // 2, cv2.BORDER_REFLECT)

    # Create a meshgrid of coordinates
    x, y = np.meshgrid(np.arange(-diameter // 2, diameter // 2 + 1), np.arange(-diameter // 2, diameter // 2 + 1))

    # Compute the spatial Gaussian component
    spatial_component = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_space ** 2))

    # Initialize the filtered image
    filtered_image = np.zeros_like(image, dtype=np.float32)

    # Apply bilateral filtering
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the local region
            region = padded_image[i:i + diameter, j:j + diameter]

            # Compute the intensity Gaussian component
            intensity_diff = region - image[i, j]
            intensity_component = np.exp(-(intensity_diff ** 2) / (2 * sigma_color ** 2))

            # Compute the bilateral filter response
            response = intensity_component * spatial_component

            # Normalize the response
            normalized_response = response / np.sum(response)

            # Compute the filtered pixel value
            filtered_pixel = np.sum(normalized_response * region)

            # Assign the filtered value to the output image
            filtered_image[i, j] = filtered_pixel

    # Convert the filtered image back to the original data type
    filtered_image = filtered_image.astype(image.dtype)

    return filtered_image
