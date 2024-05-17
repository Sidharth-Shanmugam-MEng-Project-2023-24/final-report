import os
import cv2
import logging
from multiprocessing import Queue
from natsort import natsorted

from TimeManager import Timer

class PicameraStream:
    """ Streams from a Picamera frame-by-frame. """

    def __init__(self, width, height):
        # Try to import and use Picamera2 library - this will fail on non-RPi platforms
        try:
            # This line shows a warning on IDEs on non-RPi systems to due to the package absence (Suppress!)
            from picamera2 import Picamera2 # type: ignore

            # Initialise Pi camera instance
            self.picam2 = Picamera2()

            # Pi camera configuration:
            self.picam2.still_configuration.size = (width, height)      # Capture stills at defined resolution/size
            self.picam2.still_configuration.format = 'BGR888'           # Use this format for OpenCV compatibility
            self.picam2.still_configuration.align()                     # Align stream size (THIS CHANGES THE RES SLIGHTLY!)
            
            # Apply the configuration to the Pi camera
            self.picam2.configure("video")
            
            # Print camera configurations to confirm correct set up
            logging.debug(self.picam2.camera_configuration()['sensor'])
            logging.debug(self.picam2.camera_configuration()['main'])
            logging.debug(self.picam2.camera_configuration()['controls'])

            # Start the Pi camera
            self.picam2.start()

            # Return the aligned stream resolution
            (aligned_width, aligned_height) = self.picam2.camera_configuration()['main']['size']
            return (aligned_width, aligned_height)

        # If the Picamera2 module is not found, then the program is probably not running on a RPi
        except ImportError:
            logging.error("Picamera2 module not found. Make sure you are running this on a Raspberry Pi.")
        # No other exceptions should be encountered, if they are then log it
        except Exception as e:
            logging.error("Unforseen error encountered: ", e)

    def read(self):
        """ Capture camera sensor array for constructing a frame-by-frame feed. """

        timer = Timer()
        frame = True, self.picam2.capture_array("main")
        return frame, timer.stop()

    def exit(self):
        """ Gracefully stop the Pi camera instance. """

        # Stop the Pi camera instance.
        self.picam2.stop()

        # Log to console
        logging.info("Camera instance has been gracefully stopped.")

    def empty(self):
        """ This stream is never 'empty'. """
        return False

class FrameStream:
    """ Streams a video represented by a set of PNG images for each frame. """

    def __init__(self, path):
        # Initialise path to folder
        self.path = path

        # Initialise queue to store frames in memory
        self.q = Queue()

        # Populate queue with image frames
        self._load_frames()

    def _load_frames(self):
        """ Internal method that populates the image frame queue. """

        logging.debug("Beginning to populate frame queue...")

        # Sort all of the frame files inside the folder
        frames = natsorted(os.listdir(self.path))

        # For each file in the sorted list of frame files...
        for file in frames:
            # Ensure that the file is a PNG
            if file.endswith('.png'):
                # Extract the frame number from the filename
                frame_num = int(os.path.splitext(file)[0])
                # Generate the filepath
                frame_path = os.path.join(self.path, file)
                # Read in the file
                frame = cv2.imread(frame_path)
                # Store the file in the queue along with the frame number
                self.q.put(frame)

        # Log status
        logging.debug("Frame queue has been populated.")

    def read(self):
        """ Dequeues the next frame in the sequence. """

        timer = Timer()
        if not self.empty():
            return self.q.get(), timer.stop()
        else:
            return None, timer.stop()

    def exit(self):
        """ 
        Cleanly disposes the queue.\n
        Due to Python's garbage collection not properly closing the queue
        when it has items inside, the thread/process which the queue is
        used from is prevented from exiting. By closing then preventing
        an automatic join thread when the parent process exits, allowing
        for the main thread to manually join.
        """
        self.q.close()
        self.q.cancel_join_thread()

    def empty(self):
        """ Returns true if there are no more frames to stream. """

        return self.q.empty()

class VideoStream:
    """ Streams an input video file. """

    def __init__(self, source):
        # initialise the OpenCV stream
        self.capture = cv2.VideoCapture(source)
        
        # check if capture is accessible
        if not self.capture.isOpened():
            logging.error("Cannot open video stream!")
            raise Exception("Cannot open video stream!")
        
        # calculate FPS and FPT of the capture
        self.target_fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.target_fpt = (1 / self.target_fps) * 1000

    def read(self):
        timer = Timer()
        frame = self.capture.read()
        return frame, timer.stop()

    def exit(self):
        self.capture.release()

    def empty(self):
        # Check if there are no more frames that can be read
        return not self.capture.isOpened() or self.capture.get(cv2.CAP_PROP_POS_FRAMES) >= self.capture.get(cv2.CAP_PROP_FRAME_COUNT)