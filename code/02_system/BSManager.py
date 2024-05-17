import cv2
import numpy as np

from WindowManager import Window
from TimeManager import Timer

### DEBUG WINDOW NAMES
#   These constants store the names for each GUI window
GREYSCALE_DEBUG_WINDOW_NAME = "BSDetector Debug: Greyscale"
GAUSBLUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Gaussian Blur"
CANNY_DEBUG_WINDOW_NAME = "BSDetector Debug: Canny Algorithm"
HISTEQU_DEBUG_WINDOW_NAME = "BSDetector Debug: Histogram Equalisation"
TRESHOLD_DEBUG_WINDOW_NAME = "BSDetector Debug: Binary Thresholding"

class Detector:
    """ Backscatter detection logic (V1): (a) edges are detected using the Canny algorithm, (b) the detected edges are segmented using a simple method - minimum enclosing circle (MEC), (c) the centre coordinates and radius of the detected particles (MECs) are returned. """

    def __init__(self, canny_threshold, thresh_threshold, debug_windows=True):
        # Zero-parameter threshold for canny (https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
        self.canny_threshold = canny_threshold

        self.thresh_threshold = thresh_threshold
        
        # Whether or not to print the intermediate step visualisation
        self.debug_windows = debug_windows

        # Initialise the debug windows if enabled
        if self.debug_windows:
            self.greyscale_window = Window(GREYSCALE_DEBUG_WINDOW_NAME)
            self.gausblur_window = Window(GAUSBLUR_DEBUG_WINDOW_NAME)
            self.canny_window = Window(CANNY_DEBUG_WINDOW_NAME)
            # self.histequ_window = Window(HISTEQU_DEBUG_WINDOW_NAME)
            self.threshold_window = Window(TRESHOLD_DEBUG_WINDOW_NAME)

    def _greyscale(self, frame):
        """ (Internal) Applies a greyscale filter to the frame. """

        # Log start timestamp
        timer = Timer()

        # apply the single-channel conversion with greyscale filter
        greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate process duration
        duration = timer.stop()

        # output to the debug window if enabled
        if self.debug_windows:
            self.greyscale_window.update(greyscale)

        # return the greyscaled frame
        return greyscale, duration

    def _threshold(self, frame):
        """ (Internal) Applies a binary threshold to the frame. """

        # Log start timestamp
        timer = Timer()

        # apply binary threshold
        _, thresh = cv2.threshold(
            frame,
            self.thresh_threshold[0],
            self.thresh_threshold[1],
            cv2.THRESH_BINARY
        )

        # Calculate process duration
        duration = timer.stop()

        # output to the debug window if enabled
        if self.debug_windows:
            self.threshold_window.update(thresh)
        
        # return the thresholded
        return thresh, duration

    def _histequ(self, frame):
        """ (Internal) Applies a histogram equalisation to the frame. """

        # Log start timestamp
        timer = Timer()

        # apply histogram equalisation
        histequ = cv2.equalizeHist(frame)

        # Calculate process duration
        duration = timer.stop()

        # output to the debug window if enabled
        if self.debug_windows:
            self.histequ_window.update(histequ)
        
        # return the histogram equalised frame
        return histequ, duration

    def _gausblur(self, frame):
        """ (Internal) Applies a Gaussian blur to the frame. """

        # Log start timestamp
        timer = Timer()

        # apply the Gaussian blur
        gausblur = cv2.GaussianBlur(frame, (5,5), 0)

        # Calculate process duration
        duration = timer.stop()

        # output to the debug window if enabled
        if self.debug_windows:
            self.gausblur_window.update(gausblur)
        
        # return the Gaussian blurred frame
        return gausblur, duration

    def _canny(self, frame):
        """ (Internal) Applies the Canny algorithm to the frame. """
        # Log start timestamp of Canny process
        timer = Timer()

        # compute the median single-channel pixel intensities
        gaus_median = np.median(frame)
        # compute threshold values for canny using single parameter Canny
        lower_threshold = int(max(0, (1.0 - self.canny_threshold) * gaus_median))
        upper_threshold = int(min(255, (1.0 + self.canny_threshold) * gaus_median))

        # perform Canny edge detection
        edges = cv2.Canny(
            frame,
            lower_threshold,
            upper_threshold
        )

        # Calculate the Canny process duration
        duration = timer.stop()

        # output to the debug window if enabled
        if self.debug_windows:
            self.canny_window.update(edges)

        return edges, duration

    def _segmentation(self, edges):
        """ (Internal) Finds contours using cv2.findContours() then calculates the minumum enclosing circles using cv2.minEnclosingCircle(). """

        # Log start timestamp of findContours()
        timer_findContours = Timer()

        # RETR_EXTERNAL only retrieves the extreme outer contours
        # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and
        #   diagonal segments and leaves only their end points
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,       # RetrievalModes
            cv2.CHAIN_APPROX_SIMPLE  # ContourApproximationModes
        )

        # Calculate findContours process duration
        findContours_duration = timer_findContours.stop()

        # Log start timestamp of minEnclosingCircle()
        timer_segmentContours = Timer()

        # List to store the particle information (centre coords + radius)
        particles = []

        for contour in contours:
            # Find the minimum enclosing circle for each contour
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            # Find the centre
            centre = (int(x), int(y))
            # Find the radius
            radius = int(radius)
            # Store the information
            particles.append((centre, radius))

        # Calculate findContours process duration
        segmentContours_duration = timer_segmentContours.stop()

        # return the segmented particle information
        return particles, findContours_duration, segmentContours_duration

    def detect(self, input):
        """ Detects the backscatter particles. Returns the particle coordinates and radius, and the real-time metrics. """

        # single channel conversion using greyscaling
        frame, greyscale_duration = self._greyscale(input)

        # apply Gaussian blur noise reduction and smoothening, prep for Canny
        frame, gausblur_duration = self._gausblur(frame)
        
        # apply histogram equalisation to improve contrasts for better Canny
        # frame, histequ_duration = self._histequ(frame)

        # apply a binary threshold to isolate bright backscatter
        frame, thresh_duration = self._threshold(frame)

        # apply the Canny algorithm to the frame
        edges, canny_duration = self._canny(frame)

        # segment the particles
        particles, findContours_duration, segmentContours_duration = self._segmentation(edges)

        # compile metrics into a list
        metrics = [
            greyscale_duration,
            gausblur_duration,
            # histequ_duration,
            0,
            thresh_duration,
            canny_duration,
            findContours_duration,
            segmentContours_duration
        ]

        # Get the total frame processing time, add it to metrics with total particles
        metrics = metrics + [len(particles)] + [sum(metrics)]

        # return detected particles
        return particles, metrics
