from multiprocessing import Process, Queue
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import psutil
import cv2
import os

from CaptureManager import FrameStream, VideoStream, PicameraStream
from WindowManager import Window
from TimeManager import Timer

OS_NICE_PRIORITY_LEVEL = -20
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
VIDEO_CAPTURE_SOURCE = "./import_04-08-2024-14-35-53/"

### VIDEO CAPTURE RESOLUTION
#   These are the recording parameters which dictate capture
#   resolution.
#
#   When wanting to use the frame-by-frame output from the
#   bubble-backscatter-simulation program, set these values
#   to the same as the ones input to that program (800x600).
#
#   When wanting to use a pre-recorded video source, these
#   values will be updated to match the correct resolution
#   of the video. Ensure they are similar to avoid confusion.
#
#   Want wanting to use the Pi Camera feed, these values will
#   be used when configuring the camera resolution parameters,
#   however, the camera will align the stream size to force
#   optimal alignment, so the resolution may be slightly
#   different.
VIDEO_CAPTURE_WIDTH = 800
VIDEO_CAPTURE_HEIGHT = 600

### PREVIEW/DEBUG WINDOW NAMES
#   These constants store the names for each GUI window
INPUT_PREVIEW_WINDOW_NAME = "Input Feed"
PROJECTOR_PREVIEW_WINDOW_NAME = "Projected Light Pattern"
GREYSCALE_DEBUG_WINDOW_NAME = "BSDetector Debug: Greyscale"
GAUSBLUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Gaussian Blur"
CANNY_DEBUG_WINDOW_NAME = "BSDetector Debug: Canny Algorithm"
CONTOUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Detected Contours"
HISTEQU_DEBUG_WINDOW_NAME = "BSDetector Debug: Histogram Equalisation"

### BSMANAGER PARAMETERS
#   CANNY_THRESHOLD_SIGMA: Threshold for the zero-parameter
#   Canny implementation - (https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
#
#   BS_MANAGER_HISTOGRAM_EQUALISATION: Whether or not to carry
#   out the histogram equalisation step.
#
#   BS_MANAGER_DEBUG_WINDOWS: Whether or not to display the intermediate
#   step visualisation.
CANNY_THRESHOLD_SIGMA = 0.33
BS_DEBUG_WINDOWS = False

LOGGING_FILEPATH = ""
LOGGING_FILEPATH_S1 = "export_log-s1.csv"
LOGGING_FILEPATH_S2 = "export_log-s2.csv"
LOGGING_FILEPATH_S3 = "export_log-s3.csv"
LOGGING_FILEPATH_S4 = "export_log-s4.csv"
LOGGING_FILEPATH_S5 = "export_log-s5.csv"
LOGGING_FILEPATH_S6 = "export_log-s6.csv"
LOGGING_FILEPATH_S7 = "export_log-s7.csv"
LOGGING_FILEPATH_S8 = "export_log-s8.csv"

PROCESS_QUEUE_QUIT_SIGNAL = "QUIT"
PROCESS_QUEUE_FQUIT_SIGNAL = "FQUIT"

class S1_Capture(Process):
    """ Acquires and enqueues frames from the capture source. """

    def __init__(self, output_q):
        """
        output_q: Queue to store captured frames and metrics.\n
        """
        super().__init__()
        self.output_q = output_q
        self.capture_duration = 0

    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S1_Capture,%(message)s',
            filename=LOGGING_FILEPATH + LOGGING_FILEPATH_S1,
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise capture stream
        match VIDEO_CAPTURE_SOURCE:
            # If int 0 then set up Pi camera stream
            case 0:
                # Log
                logging.debug("Initialising PicameraStream")
                # Initialise FrameStream
                stream = PicameraStream(VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT)
                # Log
                logging.debug("PicameraStream initialised")
            # If not int 0 then check if it is a valid path
            case _:
                # If the path is a directory, then FrameStream
                if os.path.isdir(VIDEO_CAPTURE_SOURCE):
                    # Log
                    logging.debug("Initialising FrameStream")
                    # Initialise FrameStream
                    stream = FrameStream(VIDEO_CAPTURE_SOURCE)
                    # Log
                    logging.debug("FrameStream initialised")
                # If the path is a file, then VideoStream
                elif os.path.isfile(VIDEO_CAPTURE_SOURCE):
                    # Log
                    logging.debug("Initialising VideoStream")
                    # Initialise VideoStream
                    stream = VideoStream(VIDEO_CAPTURE_SOURCE)
                    # Log
                    logging.debug("VideoStream initialised")

        # Initialise window to display the input
        input_feed_window = Window(INPUT_PREVIEW_WINDOW_NAME)

        # Counter to track frames that have been read
        frame_count = 0

        # Exit status
        exit = False

        while not stream.empty():
            # Log frame retrieval
            logging.debug("Retrieving frame %d", frame_count)

            # Track capture duration
            timer = Timer()
            # Capture the frame
            frame, _ = stream.read()
            # Stop the timer
            duration = timer.stop()

            # Log the retrieval
            logging.debug("Retrieved frame %d", frame_count)

            # Enqueue frame and metrics
            self.output_q.put_nowait((frame, [duration]))
            
            # Log the frame enqueue
            logging.debug("Enqueued frame %d", frame_count)
            
            # Update the preview window
            keypress = input_feed_window.update(frame)

            # Increment frame count
            frame_count += 1
            
            # Forcefully exit when the 'e' key is pressed
            if keypress == ord('e'):
                exit = True
                break

        # Handle event where capture has finished!
        stream.exit()
        if exit:
            logging.info("User pressed exit key - sending quit signal")
            self.output_q.put((PROCESS_QUEUE_FQUIT_SIGNAL, None), block=True, timeout=None)
        else:
            logging.info("No frames left to capture - sending quit signal")
            self.output_q.put((PROCESS_QUEUE_QUIT_SIGNAL, None), block=True, timeout=None)

class S2_Greyscale(Process):
    """ Process that applies a greyscale filter to an input. """

    def __init__(self, input_q, output_q):
        """
        input_q: Queue that stores frames that require greyscaling prev. stage metrics.\n
        output_q: Queue that stores greyscaled frames and metrics.
        """
        super().__init__()
        self.input_q = input_q
        self.output_q = output_q


    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S2_Greyscale,%(message)s',
            filename=LOGGING_FILEPATH + LOGGING_FILEPATH_S2,
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        # Initialise window to display debug preview
        if BS_DEBUG_WINDOWS:
            greyscale_window = Window(GREYSCALE_DEBUG_WINDOW_NAME)

        while True:
            # Log input backlog
            # logging.debug("Input backlog of %d", self.input_q.qsize())

            # Log frame retrieval
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve frame
            frame, metrics = self.input_q.get(block=True)
            # Log the retrieval
            logging.debug("Retrieved frame %d", frame_count)

            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame, None))
                    break

            # Track capture duration
            timer = Timer()
            # Apply the single-channel conversion with greyscale filter
            greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Stop the timer
            duration = timer.stop()

            # Compile metrics
            metrics.append(duration)

            # Enqueue frame and metrics
            self.output_q.put_nowait((greyscale, metrics))
            
            # Log the frame enqueue
            logging.debug("Processed and enqueued frame %d", frame_count)
            
            # Update the preview window
            if BS_DEBUG_WINDOWS:
                greyscale_window.update(greyscale)

            # Increment frame count
            frame_count += 1

class S3_HistogramEqualisation(Process):
    """ Process that applies a histogram equalisation to an input. """

    def __init__(self, input_q, output_q):
        """
        input_q: Queue that stores frames that require hist. equ. and prev. frame metrics\n
        output_q: Queue that stores hist. equalised frames and metrics.
        """
        super().__init__()
        self.input_q = input_q
        self.output_q = output_q


    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S3_HistogramEqualisation,%(message)s',
            filename=LOGGING_FILEPATH + LOGGING_FILEPATH_S3,
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        # Initialise window to display debug preview
        if BS_DEBUG_WINDOWS:
            histequ_window = Window(HISTEQU_DEBUG_WINDOW_NAME)

        while True:
            # Log input backlog
            # logging.debug("Input backlog of %d", self.input_q.qsize())

            # Log frame retrieval
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve frame
            frame, metrics = self.input_q.get(block=True)
            # Log the retrieval
            logging.debug("Retrieved frame %d", frame_count)

            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame, None))
                    break

            # Track capture duration
            timer = Timer()
            # Apply the histogram equalisation
            histequ = cv2.equalizeHist(frame)
            # Stop the timer
            duration = timer.stop()

            # Compile metrics
            metrics.append(duration)

            # Enqueue frame and metrics
            self.output_q.put_nowait((histequ, metrics))
            
            # Log the frame enqueue
            logging.debug("Processed and enqueued frame %d", frame_count)
            
            # Update the preview window
            if BS_DEBUG_WINDOWS:
                histequ_window.update(histequ)

            # Increment frame count
            frame_count += 1

class S4_GaussianBlur(Process):
    """ Process that applies a Gaussian blur to an input. """

    def __init__(self, input_q, output_q):
        """
        input_q: Queue that stores frames that require Gaus. blurring and prev. frame metrics\n
        output_q: Queue that stores Gaussian blurred frames and metrics.
        """
        super().__init__()
        self.input_q = input_q
        self.output_q = output_q


    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S4_GaussianBlur,%(message)s',
            filename=LOGGING_FILEPATH + LOGGING_FILEPATH_S4,
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        # Initialise window to display debug preview
        if BS_DEBUG_WINDOWS:
            gausblur_window = Window(GAUSBLUR_DEBUG_WINDOW_NAME)

        while True:
            # Log input backlog
            # logging.debug("Input backlog of %d", self.input_q.qsize())

            # Log frame retrieval
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve frame
            frame, metrics = self.input_q.get(block=True)
            # Log the retrieval
            logging.debug("Retrieved frame %d", frame_count)

            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame, None))
                    break

            # Track capture duration
            timer = Timer()
            # Apply the gaussian blur
            gausblur = cv2.GaussianBlur(frame, (5,5), 0)
            # Stop the timer
            duration = timer.stop()

            # Compile metrics
            metrics.append(duration)

            # Enqueue frame and metrics
            self.output_q.put_nowait((gausblur, metrics))
            
            # Log the frame enqueue
            logging.debug("Processed and enqueued frame %d", frame_count)
            
            # Update the preview window
            if BS_DEBUG_WINDOWS:
                gausblur_window.update(gausblur)

            # Increment frame count
            frame_count += 1

class S5_Canny(Process):
    """ Process that applies the Canny algorithm to an input. """

    def __init__(self, input_q, output_q):
        """
        input_q: Queue that stores frames to apply Canny with and prev. frame metrics\n
        output_q: Queue that stores the frame, Canny output edges, and metrics.
        """
        super().__init__()
        self.input_q = input_q
        self.output_q = output_q


    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S5_Canny,%(message)s',
            filename=LOGGING_FILEPATH + LOGGING_FILEPATH_S5,
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        # Initialise window to display debug preview
        if BS_DEBUG_WINDOWS:
            canny_window = Window(CANNY_DEBUG_WINDOW_NAME)

        while True:
            # Log input backlog
            # logging.debug("Input backlog of %d", self.input_q.qsize())

            # Log frame retrieval
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve frame
            frame, metrics = self.input_q.get(block=True)
            # Log the retrieval
            logging.debug("Retrieved frame %d", frame_count)

            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame, None, None))
                    break

            # Track capture duration
            timer = Timer()
            # compute the median single-channel pixel intensities
            gaus_median = np.median(frame)
            # compute threshold values for canny using single parameter Canny
            lower_threshold = int(max(0, (1.0 - CANNY_THRESHOLD_SIGMA) * gaus_median))
            upper_threshold = int(min(255, (1.0 + CANNY_THRESHOLD_SIGMA) * gaus_median))
            # perform Canny edge detection
            edges = cv2.Canny(
                frame,
                lower_threshold,
                upper_threshold
            )
            # Stop the timer
            duration = timer.stop()

            # Compile metrics
            metrics.append(duration)

            # Enqueue frame and metrics
            self.output_q.put_nowait((frame, edges, metrics))
            
            # Log the frame enqueue
            logging.debug("Processed and enqueued frame %d", frame_count)
            
            # Update the preview window
            if BS_DEBUG_WINDOWS:
                canny_window.update(edges)

            # Increment frame count
            frame_count += 1

class S6_Segmentation(Process):
    """ Process that segments edges using minimum enclosing circles (MECs). """

    def __init__(self, input_q, output_q):
        """
        input_q: Queue that stores Canny detected edges and prev. frame metrics\n
        output_q: Queue that stores Canny output edges and metrics.
        """
        super().__init__()
        self.input_q = input_q
        self.output_q = output_q


    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S6_Segmentation,%(message)s',
            filename=LOGGING_FILEPATH + LOGGING_FILEPATH_S6,
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        # Initialise window to display debug preview
        if BS_DEBUG_WINDOWS:
            contour_window = Window(CONTOUR_DEBUG_WINDOW_NAME)

        while True:
            # Log input backlog
            # logging.debug("Input backlog of %d", self.input_q.qsize())

            # Log frame retrieval
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve frame
            frame, edges, metrics = self.input_q.get(block=True)
            # Log the retrieval
            logging.debug("Retrieved frame %d", frame_count)

            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame, None, None))
                    break

            # Track find contours process
            timer = Timer()
            # 01 - Find the contours
            # RETR_EXTERNAL only retrieves the extreme outer contours
            # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and
            #   diagonal segments and leaves only their end points
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,       # RetrievalModes
                cv2.CHAIN_APPROX_SIMPLE  # ContourApproximationModes
            )
            # Stop the timer
            duration = timer.stop()

            # Compile metrics
            metrics.append(duration)

            # Track find contours process
            timer = Timer()
            # List to store the particle information (centre coords + radius)
            particles = []
            # 02 - Find minimum enclosuing circles
            for contour in contours:
                # Find the minimum enclosing circle for each contour
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                # Find the centre
                centre = (int(x), int(y))
                # Find the radius
                radius = int(radius)
                # Store the information
                particles.append((centre, radius))
            # Stop the timer
            duration = timer.stop()

            # Compile metrics
            metrics.append(duration)

            # Enqueue frame and metrics
            self.output_q.put_nowait((frame, particles, metrics))
            
            # Log the frame enqueue
            logging.debug("Processed and enqueued frame %d", frame_count)
            
            # Update the preview window
            if BS_DEBUG_WINDOWS:
                # create a black mask
                mask = np.zeros_like(edges)
                # draw contours white white fill
                cv2.drawContours(mask, contours, -1, (255), cv2.FILLED)
                # display window
                self.contour_window.update(mask)

            # Increment frame count
            frame_count += 1

class S7_Project(Process):
    """ Project the backscatter-cancelling light patterns. """

    def __init__(self, input_q, output_q):
        """
        input_q: Queue that stores computed backscatter particles and prev. frame metrics\n
        output_q: Queue that stores metrics.
        """
        super().__init__()
        self.input_q = input_q
        self.output_q = output_q


    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S7_Project,%(message)s',
            filename=LOGGING_FILEPATH + LOGGING_FILEPATH_S7,
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        # Initialise window to display debug preview
        projector_window = Window(PROJECTOR_PREVIEW_WINDOW_NAME)

        while True:
            # Log input backlog
            # logging.debug("Input backlog of %d", self.input_q.qsize())

            # Log frame retrieval
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve frame
            frame, particles, metrics = self.input_q.get(block=True)
            # Log the retrieval
            logging.debug("Retrieved frame %d", frame_count)

            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame))
                    break

            # Track capture duration
            # timer = Timer()
            # Create a white mask for the projector preview
            projector_mask = np.ones_like(frame) * 255
            # Process the particles:
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
            # Stop the timer
            # duration = timer.stop()

            # Compile metrics
            metrics.append(len(particles))

            # Enqueue frame and metrics
            self.output_q.put_nowait(metrics)
            
            # Log the frame enqueue
            logging.debug("Projected frame %d", frame_count)

            # Increment frame count
            frame_count += 1

class S8_Logging(Process):
    """ Project the backscatter-cancelling light patterns. """

    def __init__(self, input_q):
        """
        input_q: Queue that stores each frame's metrics.\n
        """
        super().__init__()
        self.input_q = input_q


    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S8_Logging,%(message)s',
            filename=LOGGING_FILEPATH + LOGGING_FILEPATH_S8,
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        # Generate CSV export filename
        export_filename_csv = "export_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + ".csv"

        # Initialise a Pandas DataFrame to log real-time metrics
        rt_metrics_df = pd.DataFrame(
            columns=[
                'Capture Duration (s)',
                'Greyscale Conversion Duration (s)',
                'Histogram Equalisation Duration (s)',
                'Gaussian Blur Duration (s)',
                'Canny Algorithm Duration (s)',
                'CV2 findContours() Duration (s)',
                'CV2 minEnclosingCircle() Duration (s)',
                'Number of MECs on screen',
            ]
        )

        while True:
            # Log input backlog
            # logging.debug("Input backlog of %d", self.input_q.qsize())

            # Log frame retrieval
            logging.debug("Retrieving metrics for frame %d", frame_count)
            # Retrieve frame
            metrics = self.input_q.get(block=True)
            # Log the retrieval
            logging.debug("Retrieved metrics for frame %d", frame_count)

            # Check if the input is a quit signal (check whether it's a string first!)
            if type(metrics) == str:
                if metrics == PROCESS_QUEUE_QUIT_SIGNAL:
                    # If it is then export to CSV and break
                    logging.info("Quit signal received - starting to export")
                    # Export dataframe as CSV
                    rt_metrics_df.to_csv(
                        path_or_buf=export_filename_csv,
                        encoding='utf-8'
                    )
                    logging.info("Successfully exported - now exiting")
                    break
                elif metrics == PROCESS_QUEUE_FQUIT_SIGNAL:
                    logging.info("Force quit signal received - I am now quitting")
                    break

            # Log the particles retrieval 
            logging.debug("Logging data for frame %d", frame_count)
            # Add this frame's metrics to the end of the dataframe
            rt_metrics_df.loc[len(rt_metrics_df)] = metrics
            # Log the particle retrieval and processing 
            logging.debug("Logged data for frame %d", frame_count)
            # Increment frame count
            frame_count += 1

if __name__ == "__main__":
    # Required to run X11 forwarding as sudo
    os.environ['XAUTHORITY'] = X11_XAUTHORITY_PATH

    # Set the OS priority level
    p = psutil.Process(os.getpid())
    print("Current OS priority: ", p.nice())
    p.nice(OS_NICE_PRIORITY_LEVEL)
    print("New OS priority: ", p.nice())

    q1_2 = Queue()  # Queue between stages 1 and 2
    q2_3 = Queue()  # Queue between stages 2 and 3
    q3_4 = Queue()  # Queue between stages 3 and 4
    q4_5 = Queue()  # Queue between stages 4 and 5
    q5_6 = Queue()  # Queue between stages 5 and 6
    q6_7 = Queue()  # Queue between stages 6 and 7
    q7_8 = Queue()  # Queue between stages 7 and 8

    stages = [
        S1_Capture(output_q=q1_2),
        S2_Greyscale(input_q=q1_2, output_q=q2_3),
        S3_HistogramEqualisation(input_q=q2_3, output_q=q3_4),
        S4_GaussianBlur(input_q=q3_4, output_q=q4_5),
        S5_Canny(input_q=q4_5, output_q=q5_6),
        S6_Segmentation(input_q=q5_6, output_q=q6_7),
        S7_Project(input_q=q6_7, output_q=q7_8),
        S8_Logging(input_q=q7_8)
    ]

    # Start the stages
    for stage in stages:
        stage.start()

    # Wait for stages to finish
    for stage in stages:
        stage.join()

    cv2.destroyAllWindows()