from datetime import datetime
import numpy as np
import pandas as pd
import psutil
import cv2
import os

from CaptureManager import FrameStream, VideoStream, PicameraStream
from WindowManager import Window
from BSManager import Detector
from TimeManager import Timer

OS_NICE_PRIORITY_LEVEL = 0
X11_XAUTHORITY_PATH = '/home/sid/.Xauthority'

### VIDEO CAPTURE SOURCE
#   Input a directory path to feed in a series of frame images,
#   named as an integer denoting the frame number and must be
#   in PNG format.
#
#   Input a file path to feed in a video file.
#
#   Input integer '0' to use Picamera2 capture_array() to capture
#   feed frame-by-frame.
# VIDEO_CAPTURE_SOURCE = 0
# VIDEO_CAPTURE_SOURCE = "./import_04-08-2024-14-35-53/"
# VIDEO_CAPTURE_SOURCE = "../Tank Recordings/recording_04-22-2024-14-36-49.mkv"
VIDEO_CAPTURE_SOURCE = "../Tank Recordings/recording_04-22-2024-14-16-08.mkv"

### VIDEO CAPTURE PARAMETERS
#   WIDTH & HEIGHT:
#       These are the recording parameters which dictate capture
#       resolution.
#
#       When wanting to use the frame-by-frame output from the
#       bubble-backscatter-simulation program, set these values
#       to the same as the ones input to that program (800x600).
#
#       When wanting to use a pre-recorded video source, these
#       values will be updated to match the correct resolution
#       of the video. Ensure they are similar to avoid confusion.
#
#       Want wanting to use the Pi Camera feed, these values will
#       be used when configuring the camera resolution parameters,
#       however, the camera will align the stream size to force
#       optimal alignment, so the resolution may be slightly
#       different.
#
#   CROP WIDTH & HEIGHT:
#       Apply a crop to isolate the region of interest, mitigating
#       the submersible housing offsets
VIDEO_CAPTURE_WIDTH = 800
VIDEO_CAPTURE_HEIGHT = 600
VIDEO_CAPTURE_CROP_HEIGHT = (0, 320)
VIDEO_CAPTURE_CROP_WIDTH = (430, 700)

### PREVIEW WINDOW NAMES
#   These constants store the names for each GUI window
INPUT_PREVIEW_WINDOW_NAME = "Input Feed"
SEGMENTATION_PREVIEW_WINDOW_NAME = "Backscatter Segmentation"
PROJECTOR_PREVIEW_WINDOW_NAME = "Projected Light Pattern"

### BSMANAGER PARAMETERS
#   CANNY_THRESHOLD_SIGMA: Threshold for the zero-parameter
#   Canny implementation - (https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
#
#   BS_MANAGER_DEBUG_WINDOWS: Whether or not to display the intermediate
#   step visualisation.
CANNY_THRESHOLD_SIGMA = 0
BS_MANAGER_HISTOGRAM_EQUALISATION = True
BS_MANAGER_DEBUG_WINDOWS = True
BINARY_THRESHOLD_THRESH = (200, 255)


if __name__ == "__main__":
    # Required to run X11 forwarding as sudo
    os.environ['XAUTHORITY'] = X11_XAUTHORITY_PATH

    # Set the OS priority level
    p = psutil.Process(os.getpid())
    print("Current OS priority: ", p.nice())
    p.nice(OS_NICE_PRIORITY_LEVEL)
    print("New OS priority: ", p.nice())

    # Generate CSV export filename
    export_filename_csv = "export_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + ".csv"

    # Initialise capture stream
    match VIDEO_CAPTURE_SOURCE:
        # If int 0 then set up Pi camera stream
        case 0:
            # Initialise FrameStream
            stream = PicameraStream(VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT)
        # If not int 0 then check if it is a valid path
        case _:
            # If the path is a directory, then FrameStream
            if os.path.isdir(VIDEO_CAPTURE_SOURCE):
                # Initialise FrameStream
                stream = FrameStream(VIDEO_CAPTURE_SOURCE)
            # If the path is a file, then VideoStream
            elif os.path.isfile(VIDEO_CAPTURE_SOURCE):
                # Initialise VideoStream
                stream = VideoStream(
                    source=VIDEO_CAPTURE_SOURCE,
                    crop_w=VIDEO_CAPTURE_CROP_WIDTH,
                    crop_h=VIDEO_CAPTURE_CROP_HEIGHT,
                    loop=True
                )
    
    # Initialise window to display the input
    input_feed_window = Window(INPUT_PREVIEW_WINDOW_NAME)

    # Initialise window to display the particle segmentation
    segmentation_window = Window(SEGMENTATION_PREVIEW_WINDOW_NAME)

    # Initialise window to display the projector output
    projector_window = Window(PROJECTOR_PREVIEW_WINDOW_NAME)

    # Initialise the backscatter detector
    detector = Detector(
        canny_threshold=CANNY_THRESHOLD_SIGMA,
        thresh_threshold=BINARY_THRESHOLD_THRESH,
        debug_windows=BS_MANAGER_DEBUG_WINDOWS
    )

    # Initialise variable to track frame processing duratione (sec)
    total_frame_processing_time = 0

    # Initialise a Pandas DataFrame to log real-time metrics
    rt_metrics_df = pd.DataFrame(
        columns=[
            'Capture Duration (s)',
            'Greyscale Conversion Duration (s)',
            'Gaussian Blur Duration (s)',
            'Histogram Equalisation Duration (s)',
            'Binary Thresholdind Duration (s)',
            'Canny Algorithm Duration (s)',
            'CV2 findContours() Duration (s)',
            'CV2 minEnclosingCircle() Duration (s)',
            'Number of MECs on screen',
            'Total frame processing time (s)'
        ]
    )
    
    while True:
        # Start timer to calculate capture duration
        timer = Timer()

        # Read a frame
        frame = stream.read()

        # Stop timer
        capture_duration = timer.stop()

        # Detect keypress
        keypress = cv2.waitKey(1)

        # Exit if the 'e' key is pressed
        if keypress == ord('e'):
            break

        # While there are frames...
        if frame is not None:
            # Update the input visualisation window
            input_feed_window.update(frame)

            # Start timer for the total frame processing duration
            timer = Timer()

            # Detect the particles and retrieve real-time metrics
            particles, metrics = detector.detect(frame)

            # Stop the total frame processing duration timer
            frame_processing_time = timer.stop()

            # Create a black mask for the segmentation preview
            particle_mask = np.copy(frame)

            # Draw white circles on the black mask for each MEC
            for particle in particles:
                cv2.circle(
                    particle_mask,
                    particle[0],
                    particle[1],
                    (0, 0, 255),
                    1
                )

            # Display the black mask with white circles
            segmentation_window.update(particle_mask)

            # Create a white mask for the projector preview
            projector_mask = np.ones_like(frame) * 255

            # Draw black circles on the black mask for each MEC
            for particle in particles:
                cv2.circle(
                    projector_mask,
                    particle[0],
                    particle[1],
                    (0, 0, 0),
                    -1
                )

            # Display the white mask with black circles
            projector_window.update(projector_mask)

            # Prepend the capture time to the metrics
            metrics = [capture_duration] + metrics

            # Append metrics list to end of dataframe
            rt_metrics_df.loc[len(rt_metrics_df)] = metrics
        else:
            # break out of the while loop when there are no more frames
            break
    
    # Export dataframe as CSV
    rt_metrics_df.to_csv(
        path_or_buf=export_filename_csv,
        encoding='utf-8'
    )

    cv2.destroyAllWindows()
