import cv2
import numpy as np
import m_estimators as m
import time

# Apply bilateral filter to video
def bilateral(input_path, output_path, d = 10, sigmaColor = 20, sigmaSpace = 30):
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

    # Release the video capture and writer objects
    video.release()
    output.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

#Appy temporal filter to video
def temporal(video_path, output_path, window_size):
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
            output.write(filtered_frame)
    
    output.release()
    cap.release()
    cv2.destroyAllWindows()

def median(input_path,output_path, window_size=3):
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


def gaussian(input_path, output_path, kernel_size=5, sigmaX=1):
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

def createBuffer(bufferSize: int, height: int, width: int):
    buffer = np.zeros((bufferSize, height, width, 3), dtype=np.uint8)

    def addToBuffer(frame):
        nonlocal buffer
        buffer = np.roll(buffer, 1, axis=0)
        buffer[0] = frame
    
    def getBuffer():
        return buffer

    return addToBuffer, getBuffer 

def RBLT(inPath: str, outPath: str, spatialKernelSize: int = 3, spaceSigma: float = 1.0, beta: float = 1.0, timeWindowSize: int = 5, timeSigma: float = 1.0):
    startTime = time.time()
    cap = cv2.VideoCapture(inPath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outPath, fourcc, 30.0, (width, height))
    borderSize = int((spatialKernelSize-1)/2)
    gaussKernel = np.array(m.gaussianKernel(spatialKernelSize, spaceSigma)) #3-channel gaussian kernel
    
    addToBuffer, getBuffer = createBuffer(timeWindowSize, height, width)
    timeMult = m.timeArray(timeWindowSize, timeSigma, beta)
    ret, frame = cap.read()

    for i in range(timeWindowSize):
        if not ret:
            break
        addToBuffer(frame)
        ret, frame = cap.read()

    counter = 0
    while ret and counter < 6:
        frameCopy = cv2.copyTo(frame, mask=None)
        borderedFrame = cv2.copyMakeBorder(frame, borderSize, borderSize, borderSize, borderSize, borderType=cv2.BORDER_REFLECT_101)
        
        for y in range(borderSize, borderedFrame.shape[1]-borderSize):
            for x in range(borderSize, borderedFrame.shape[0]-borderSize):
                roi = np.array(borderedFrame[x-borderSize:x+borderSize+1, y-borderSize:y+borderSize+1])
                kernel = np.array(m.charbonnier(roi - borderedFrame[x, y], beta)) * gaussKernel
                kernel /= np.sum(kernel, axis=(0, 1))
                spatialContrib = np.sum(roi * kernel, axis=(0, 1))
                temporalContrib = np.sum(getBuffer()[:, x-borderSize, y-borderSize]*timeMult, axis=0)
                frame[x-borderSize, y-borderSize] = np.uint8(spatialContrib/2 + temporalContrib/2)

        addToBuffer(frameCopy)
        out.write(frame)
        ret, frame = cap.read()
        counter += 1

    print(time.time()-startTime)
    # cv2.imshow("frame",frame)
    # cv2.waitKey(0)
    
    cap.release()
    out.release()  

def rgb_to_ycbcr(input_path, output_path):

    video = cv2.VideoCapture(input_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height), isColor=True)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        ycbcr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        out.write(ycbcr_frame)

    video.release()
    out.release()

def ycbcr_to_rgb(input_path, output_path):
    video = cv2.VideoCapture(input_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height), isColor=True)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2RGB)
        out.write(rgb_frame)

    video.release()
    out.release()
