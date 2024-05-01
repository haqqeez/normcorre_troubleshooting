import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import variance
from scipy.signal import savgol_filter
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
from math import log10, sqrt
import os
import multiprocessing



def someinfo():
    ### information about video stability and how to calculate it
    '''
    Video stability can be calculated using various techniques, many of which involve artificial intelligence (AI) and sophisticated algorithms. Here's a brief overview:

    1. **AI-Based Video Stabilization**: AI has significantly impacted video stabilization by enabling advanced algorithms to analyze, understand, and correct camera movements². Traditional stabilization techniques often rely on basic motion tracking, while AI-based methods go beyond that, harnessing the power of machine learning and pattern recognition².

    2. **Algorithms**: AI-based video stabilization algorithms employ sophisticated techniques to analyze each video frame and determine the optimal stabilization parameters². These algorithms use a combination of image analysis, motion estimation, and compensation methods to counteract camera shakes and vibrations². One widely used algorithm is the optical flow analysis, which tracks the motion of distinct features in consecutive frames². By understanding the changes in position of these features, the algorithm can calculate the camera's motion and apply appropriate correction to stabilize the video².

    3. **Stable Video Diffusion**: This is a model released in the form of two image-to-video models, capable of generating 14 and 25 frames at customizable frame rates between 3 and 30 frames per second¹. It surpasses the leading closed models in user preference studies¹.

    Please note that the specific method for calculating video stability may vary depending on the specific requirements of your project and the tools you have available. It's also important to remember that video stabilization not only improves the aesthetic quality of the footage but also reduces motion sickness and discomfort for viewers².

    Source: Conversation with Bing, 4/1/2024
    (1) Video Stabilization Through AI: A Complete Guide. https://hivo.co/blog/ai-based-video-stabilisation-a-comprehensive-guide.
    (2) Stable Video — Stability AI. https://stability.ai/stable-video.
    (3) How to Quickly Determine Stability – A Simplfied Approach. https://mentoredengineer.com/determining-stability/.


    To measure the improvement in video stabilization, you can compare the following aspects between the original shaky video and the stabilized video:

    1. **Motion Estimation**: This is the primary goal of video stabilization. It estimates the camera's motion between consecutive frames¹. You can compare the estimated motion of the camera in the shaky and stabilized videos.

    2. **Divergence and Jitter**: These are measures of the remaining unintentional motion in the video³. A decrease in divergence and jitter in the stabilized video compared to the shaky video indicates an improvement.

    3. **Blurring**: Blurring can be measured using the point spread function (PSF)³. If the stabilized video has less blurring than the shaky video, it's an indication of improvement.

    4. **Trajectory Smoothness**: You can calculate the moving average of the trajectory in both videos¹. A smoother trajectory in the stabilized video indicates an improvement.

    5. **Viewer Experience**: Finally, subjective measures such as viewer comfort and reduction in motion sickness can also be used to assess improvement¹.

    Remember, the specific metrics you use may depend on the specific requirements of your project and the tools you have available. It's also important to note that video stabilization not only improves the aesthetic quality of the footage but also reduces motion sickness and discomfort for viewers¹.

    Source: Conversation with Bing, 4/1/2024
    (1) How to Implement Real-Time Video Stabilization Using OpenCV - MUO. https://www.makeuseof.com/opencv-real-time-video-stabilization-how-to-implement/.
    (2) VIDEO STABILIZATION PERFORMANCE ASSESSMENT. https://www.cecs.uci.edu/~papers/icme06/pdfs/0000405.pdf.
    (3) Video stabilization: A comprehensive survey - ScienceDirect. https://www.sciencedirect.com/science/article/pii/S092523122201270X.
    (4) Video Stabilization Through AI: A Complete Guide. https://hivo.co/blog/ai-based-video-stabilisation-a-comprehensive-guide.
    (5) Main steps for video stabilization. | Download Scientific Diagram. https://www.researchgate.net/figure/Main-steps-for-video-stabilization_fig2_346166370.

    The line of code you're asking about is using the `calcOpticalFlowFarneback` function from the OpenCV library to calculate the optical flow. Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the movement of object or camera. It's a 2D vector field where each vector is a displacement vector showing the movement of points from first frame to second.

    Here's what each parameter in the function does:

    - `prev_gray`: The first 8-bit single-channel input image.
    - `gray`: The second input image of the same size and the same type as `prev_gray`.
    - `None`: The computed flow image that has the same size as `prev_gray` and type CV_32FC2.
    - `0.5`: The image scale. This is the pyramid scale: <1 to use a Gaussian pyramid. Smaller values lead to finer-scale estimates.
    - `3`: The number of pyramid layers, including the initial image; levels=1 means that no extra layers are created and only the original images are used.
    - `15`: The averaging window size. Larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
    - `3`: The number of iterations the algorithm does at each pyramid level.
    - `5`: The size of the pixel neighborhood used to find polynomial expansion in each pixel.
    - `1.2`: The standard deviation of the Gaussian used to smooth derivatives used as a basis for the polynomial expansion.
    - `0`: Additional flags that can be set to 0 for now.

    So, in simpler terms, this function is comparing two grayscale images (`prev_gray` and `gray`) and calculating the apparent motion of pixels from `prev_gray` to `gray`. The other parameters are used to fine-tune the calculation. The result, `flow`, is a 2D vector field of the same size as the input images, where each vector shows the motion of a pixel from `prev_gray` to `gray`. This information can then be used to estimate the camera's motion and stabilize the video.


    The Peak Signal-to-Noise Ratio (PSNR) is a metric that measures the quality of a reconstructed or compressed image or video compared to the original. In the context of video stabilization, you can use PSNR to compare the quality of the stabilized video to the original shaky video.

    Here's how it works:

    1. **High PSNR**: If the PSNR between the original and stabilized videos is high (typically above 30 dB for 8-bit depth videos), this indicates that the stabilized video is very similar to the original. This means your motion-correction algorithm has likely done a good job of stabilizing the video without introducing significant distortions.

    2. **Low PSNR**: On the other hand, if the PSNR is low, this suggests that there are significant differences between the original and stabilized videos. This could be due to distortions introduced by the stabilization process, such as blurring, ghosting, or cropping.

    However, it's important to note that PSNR is just one of many metrics that can be used to evaluate video stabilization. While a high PSNR indicates that the stabilized video is similar to the original, it doesn't necessarily mean that the stabilization is visually pleasing or effective. Other factors, such as the smoothness of the camera motion and the absence of jitter, are also important to consider.

    Moreover, PSNR is a purely mathematical metric and does not always align with human perception of quality. Therefore, visual inspection of the videos and user studies can also be valuable tools for evaluating the effectiveness of video stabilization algorithms.


    The Peak Signal-to-Noise Ratio (PSNR) is a common metric used to measure the quality of reconstructed or compressed images or videos. It's calculated using the Mean Squared Error (MSE) between the original and the reconstructed image or video.

    Here's the step-by-step process:

    1. **Calculate the Mean Squared Error (MSE)**: The MSE is the average squared difference between the pixel intensities of the original and the reconstructed images or videos. For an 8-bit grayscale image, it's calculated as follows:

        $$\text{MSE} = \frac{1}{MN}\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}[I(m, n) - K(m, n)]^2$$

        where $I(m, n)$ is the pixel intensity at location $(m, n)$ in the original image, $K(m, n)$ is the pixel intensity at location $(m, n)$ in the reconstructed image, and $M$ and $N$ are the dimensions of the images.

    2. **Calculate the PSNR**: Once you have the MSE, you can calculate the PSNR using the following formula:

        $$\text{PSNR} = 20 \cdot \log_{10}\left(\frac{\text{MAX}_I}{\sqrt{\text{MSE}}}\right)$$

        where $\text{MAX}_I$ is the maximum possible pixel value of the image. For an 8-bit grayscale image, this is 255.

    The PSNR is usually expressed in decibels (dB), which is a logarithmic scale. The higher the PSNR, the better the quality of the reconstructed or compressed image or video. However, it's worth noting that the PSNR is not always an accurate reflection of perceptual image quality. Other factors, such as the viewing conditions and the viewer's own visual system, can also play a role.


    In the context of video stabilization:

    1. **Blur**: Ideally, the amount of blur should **decrease** after stabilization. Blur in a video is usually caused by rapid movement or shaking of the camera. A good stabilization algorithm should be able to reduce this shaking, resulting in less blur.

    2. **Trajectory Smoothness**: The smoothness of the trajectory should **increase** after stabilization. A smooth trajectory means that the motion of the camera is steady and does not have sudden jumps or jitters. A higher smoothness value indicates a more stable video.

    So, in summary, for a well-stabilized video, blur should go down and trajectory smoothness should go up. However, these are just two of many possible metrics for video stabilization, and the specific requirements may vary depending on the application. It's always a good idea to visually inspect the videos as well.

    The trajectory over time plot shows the cumulative motion of the camera over the course of the video. Here's how to interpret it:

    - The **x-axis** represents the **frame number** or time, and the **y-axis** represents the **cumulative sum of the optical flow**, which is a measure of the camera's motion.

    - A **steady increase or decrease** in the trajectory indicates a **steady motion** of the camera in one direction. This could be a panning or tilting motion, for example.

    - A **flat section** of the trajectory indicates a period where the camera is **stationary**.

    - Any **sharp changes** in the trajectory indicate **sudden movements** or shakes of the camera.

    - In the context of video stabilization, you would expect the trajectory of the stabilized video to be smoother and have fewer sharp changes compared to the original video. This would indicate that the stabilization algorithm has successfully reduced the camera shake.

    Remember, the specific interpretation can depend on the specific requirements of your project and the nature of the videos you are working with. It's always a good idea to visually inspect the videos as well.

    In the context of video stabilization:

    - **Divergence** refers to the amount of variation or dispersion in the motion vectors of a video. A high divergence means there's a lot of inconsistency in the motion, which usually indicates a shaky or unstable video. In other words, divergence measures the degree to which the video deviates from a smooth path.

    - **Jitter** refers to the small, rapid variations in a signal or movement, often caused by noise or instability. In terms of video, jitter can be seen as abrupt changes in the motion from one frame to another. High jitter can make a video appear jerky or jumpy.

    So, while both divergence and jitter are measures of instability in a video, they capture different aspects of it. Divergence is more about the overall inconsistency in the motion, while jitter is about the abrupt changes from frame to frame. A good video stabilization algorithm should aim to minimize both.



    '''
    pass

def load_video(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    frames = []
    while video.isOpened():
        # Read the next frame from the video
        ret, frame = video.read()

        # If the frame was read correctly, add it to the list
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)
        else:
            # If no frame could be read, break the loop
            break

    # Release the video file
    video.release()

    return frames

def calculate_metrics(original_video_path, stabilized_video_path):
    # Load the videos
    original_video = cv2.VideoCapture(original_video_path)
    stabilized_video = cv2.VideoCapture(stabilized_video_path)

    # Initialize metrics
    original_motion = []
    stabilized_motion = []

    # Calculate motion between frames for both videos
    for video, motion, video_name in [(original_video, original_motion, "Original Video"), (stabilized_video, stabilized_motion, "Stabilized Video")]:
        # print(f"Processing {video_name}...")
        ret, prev_frame = video.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Get the total number of frames
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Process each frame
        for _ in tqdm(range(total_frames), desc=f"Processing {video_name}"):
            ret, frame = video.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Calculate magnitude and angle of 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Calculate mean magnitude of vectors, which represents the motion
            mean_magnitude = np.mean(magnitude)

            # Append the mean magnitude to the motion list
            motion.append(mean_magnitude)

            prev_gray = gray

    # Calculate divergence and jitter
    original_divergence = np.std(original_motion)
    stabilized_divergence = np.std(stabilized_motion)

    original_jitter = np.mean(np.diff(original_motion))
    stabilized_jitter = np.mean(np.diff(stabilized_motion))

    # Calculate improvement in divergence and jitter
    divergence_improvement = original_divergence - stabilized_divergence
    jitter_improvement = original_jitter - stabilized_jitter

    # Print the results
    print(f'Results for {stabilized_video_path}')
    print(f"Divergence Improvement: {divergence_improvement}")
    print(f"Jitter Improvement: {jitter_improvement}")


def calculate_psnr(stabilized_video_path):
    # Load the videos
    # original_video = cv2.VideoCapture(original_video_path)
    stabilized_video = cv2.VideoCapture(stabilized_video_path)

    # Initialize PSNR list
    psnr_list = []

    # Get the total number of frames
    total_frames = int(stabilized_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # print(f"Processing videos...")

    # Process each frame
    for i in tqdm(range(total_frames - 1), desc="Calculating PSNR"):
        # Read two consecutive frames from the videos
        if i == 0:
            ret1, frame1 = stabilized_video.read()
            ret2, frame2 = stabilized_video.read()
            if not ret1 or not ret2:
                break
        else:
            frame1 = frame2
            ret2, frame2 = stabilized_video.read()
            # Break the loop if video ends
            if not ret2:
                break

        # Calculate MSE
        mse = np.mean((frame1 - frame2) ** 2)

        # Avoid division by zero
        if mse == 0:
            return float('inf')

        # Calculate PSNR
        # find max pixel based on bitrate??
        # bitrate = stabilized_video.get(cv2.CAP_PROP_BITRATE)
        # if bitrate == 0:
            # max_pixel = 255.0
        # else:
            # max_pixel = 2 ** (bitrate / 8) - 1
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))

        # Append PSNR to the list
        psnr_list.append(psnr)

    # Calculate average PSNR
    average_psnr = np.mean(psnr_list)

    print(f"The average PSNR between consecutive frames in {stabilized_video_path} video is {average_psnr} dB.")

    # return average_psnr


def calculate_ssim_consecutive_frames(video):
    '''
    This function calculates the SSIM (structural similarity index) between each pair of consecutive frames in the video, then calculates the average SSIM. 
    This average SSIM can give you an idea of how much the video changes from frame to frame: 
    a lower average SSIM means the video changes more, while a higher average SSIM means the video is more stable.
    
    '''
    # Load the video
    video_frames = load_video(video)

    frame = video_frames[0]
    min_dim = min(frame.shape)
    win_size = min(min_dim | 1, 7)
    # print(min_dim)

    # Calculate the SSIM for each pair of consecutive frames
    ssim_values = [compare_ssim(video_frames[i], video_frames[i+1], win_size=win_size) for i in tqdm(range(len(video_frames)-1))]

    # Calculate the average SSIM
    average_ssim = sum(ssim_values) / len(ssim_values)

    print(f"The average SSIM between consecutive frames in the {video} video is {average_ssim}.")

def estimate_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return variance(cv2.Laplacian(gray, cv2.CV_64F))

def calculate_trajectory(video_path):
    video = cv2.VideoCapture(video_path)
    trajectory = []
    ret, prev_frame = video.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), desc="Calculating Trajectory"):
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        trajectory.append(np.sum(flow))
        prev_gray = gray
    return trajectory

def calculate_smoothness(trajectory):
    '''
    The savgol_filter function from the scipy.signal module applies a Savitzky-Golay filter to a 1D array. This filter is used to smooth the data without greatly distorting the signal.

    The function takes three main arguments:

    The input data array.
    The window length: This is the length of the window for which the filter will compute a high-degree polynomial to fit the data. It must be a positive odd integer. In your case, it's 51.
    The polynomial order: This is the order of the polynomial used to fit the data. It must be less than the window length. In your case, it's 3.
    So, savgol_filter(derivative, 51, 3) applies a Savitzky-Golay filter to the derivative array, using a window length of 51 and a polynomial order of 3. 
    The choice of these parameters depends on the specific characteristics of your data and the amount of smoothing you want to apply.

    Here's a more detailed explanation of the process:

    Fitting a polynomial of a certain order to a window of data points: 
        The filter starts by selecting a subset of the data, which is defined by the window size. For example,
        if the window size is 5, it would select the first 5 data points. 
        It then fits a polynomial of a certain order (degree) to these data points. 
        The order of the polynomial determines the complexity of the shapes that can be fitted. 
        For example, a 1st order polynomial is a straight line, a 2nd order polynomial can be a curve (parabola), 
        a 3rd order polynomial can fit more complex shapes, and so on.

    Using the value of the polynomial at the center of the window as the smoothed value for the center point: 
        Once the polynomial is fitted to the data points in the window, the value of the polynomial at the center of the window is calculated. 
        This value is used as the smoothed value for the center data point. 
        This means that the original value of the center data point is replaced with the value of the polynomial at that point, 
        which is likely to be closer to the underlying trend in the data.

    Moving the window along the data points and repeating the process: 
        After calculating the smoothed value for the center point of the current window, the window is moved 
        one data point to the right, and the process is repeated. This means that a new polynomial is fitted to the new window of data points, 
        a new smoothed value is calculated for the new center point, and so on. This continues until the window reaches the end of the data.

    The result of this process is a smoothed version of the original data, 
    where the smoothed value of each data point is determined by the values of the surrounding data points 
    (as defined by the window size) and the shape of the polynomial that can best fit those points (as defined by the polynomial order).

    In your code, savgol_filter(derivative, 51, 3) is fitting a 3rd order (cubic) polynomial to a window of 51 data points at a time. 
    This means it's able to capture more complex trends in the data compared to a lower order polynomial, 
    but without overfitting to the noise as a higher order polynomial might do. 
    The choice of polynomial order depends on the characteristics of your data and the amount of smoothing you want to apply.
    '''

    derivative = np.diff(trajectory)
    smoothed_derivative = savgol_filter(derivative, 51, 3) # why these values?
    difference = derivative - smoothed_derivative
    return np.var(difference)

def compare_videos(original_video_path, stabilized_video_path):
    original_video = cv2.VideoCapture(original_video_path)
    stabilized_video = cv2.VideoCapture(stabilized_video_path)
    original_blur = []
    stabilized_blur = []
    total_frames = int(original_video.get(cv2.CAP_PROP_FRAME_COUNT))
    for video, blur, video_name in [(original_video, original_blur, "Original Video"), (stabilized_video, stabilized_blur, "Stabilized Video")]:
        print(f"Processing {video_name}...")
        for _ in tqdm(range(total_frames), desc=f"Calculating Blur for {video_name}"):
            ret, frame = video.read()
            if not ret:
                break
            blur.append(estimate_blur(frame))

    average_original_blur = np.mean(original_blur)
    average_stabilized_blur = np.mean(stabilized_blur)

    print("Calculating trajectories...")
    original_trajectory = calculate_trajectory(original_video_path)
    stabilized_trajectory = calculate_trajectory(stabilized_video_path)

    print("Calculating smoothness...")
    original_smoothness = calculate_smoothness(original_trajectory)
    stabilized_smoothness = calculate_smoothness(stabilized_trajectory)

    print(f'Results for {stabilized_video_path}')
    print(f"Average Original Blur: {average_original_blur}")
    print(f"Average Stabilized Blur: {average_stabilized_blur}")
    print(f"Original Smoothness: {original_smoothness}")
    print(f"Stabilized Smoothness: {stabilized_smoothness}")

    # plt.figure(figsize=(12, 6))
    # plt.plot(original_blur, label="Original Video")
    # plt.plot(stabilized_blur, label="Stabilized Video")
    # plt.xlabel("Frame")
    # plt.ylabel("Blur (Variance of Laplacian)")
    # plt.legend()
    # plt.title("Blur Over Time")
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(original_trajectory, label="Original Video")
    # plt.plot(stabilized_trajectory, label="Stabilized Video")
    # plt.xlabel("Frame")
    # plt.ylabel("Trajectory (Sum of Optical Flow)")
    # plt.legend()
    # plt.title("Trajectory Over Time")
    # plt.show()


def find_unique_videos(original_video):
    # Get the list of video files in the directory
    video_files = os.listdir()

    # check that file ends with .avi
    video_files = [file for file in video_files if file.endswith('.avi')]

    # Remove '0.avi' from the list
    video_files.remove(original_video)

    # Find unique video files
    unique_videos = list(set(video_files))

    return unique_videos


# Define a function to compare videos and calculate metrics
def compare_and_calculate(original_video, video):
    print(f"Comparing original with {video}...")
    compare_videos(original_video, video)
    calculate_metrics(original_video, video)
    calculate_ssim_consecutive_frames(video)
    # calculate_psnr(video)
    
if __name__ == "__main__":
    directory = r'C:\Users\Zeeshan\Documents\PAL_Project\pythonpath\corrected_videos'
    os.chdir(directory)

    original_video = '0.avi'

    # Find unique videos
    unique_videos = find_unique_videos(original_video)
    print(unique_videos)

    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))

    # Apply the compare_and_calculate function to each video in parallel
    results = []
    for video in unique_videos:
        result = pool.apply_async(compare_and_calculate, (original_video, video))
        results.append(result)

    # Wait for all processes to finish
    pool.close()
    pool.join()

    # Get the results
    for result in results:
        result.get()
    
