import cv2
import numpy as np
import m_estimators as m
import time

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
            
def RBLT(inPath: str, outPath: str, spatialKernelSize: int = 3, sigma: float = 1.0, timeWindowSize: int = 5):
    startTime = time.time()
    cap = cv2.VideoCapture(inPath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outPath, fourcc, 30.0, (width, height), isColor=True)
    borderSize = int((spatialKernelSize-1)/2)
    gaussKernel = np.array(m.gaussianKernel(spatialKernelSize, sigma)) #3-channel gaussian kernel
    ret, frame = cap.read()
    # print("border size: " + str(borderSize))
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        bordered_frame = cv2.copyMakeBorder(frame, borderSize, borderSize, borderSize, borderSize, borderType=cv2.BORDER_REFLECT_101)

        for y in range(borderSize, bordered_frame.shape[1]-borderSize):
            for x in range(borderSize, bordered_frame.shape[0]-borderSize):
                # print("x: "+str(x)+" y: "+str(y))
                roi = np.array(bordered_frame[x-borderSize:x+borderSize+1, y-borderSize:y+borderSize+1])
                # print(roi.shape)
                kernel = np.array(m.charbonnier(roi - bordered_frame[x, y])) * gaussKernel
                kernel /= np.sum(kernel, axis=(0, 1))
                frame[x-borderSize, y-borderSize] = np.sum(roi * kernel, axis=(0, 1))

        frame = cv2.cvtColor(frame, cv2.COLOR_YCR_CB2BGR)
        out.write(frame)
        ret, frame = cap.read()
        
    print(time.time()-startTime)
    cap.release()
    out.release()  

