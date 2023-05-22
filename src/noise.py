import cv2
import numpy as np

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

    # Release resources
    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()