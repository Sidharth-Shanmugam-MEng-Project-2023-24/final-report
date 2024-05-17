import io
import os
import subprocess
from picamera2 import Picamera2
from picamera2.encoders import Encoder
from picamera2.outputs import FileOutput
from datetime import datetime

class Camera:
    """
    A wrapper to interface with the Pi camera with implementations to capture a frame for preview, start and stop a recording.
    This interface will record and capture the 'main' stream which is set up as the processed 'raw' feed.
    """

    def __init__(self, width, height, rate):

        # Initialise Pi camera instance
        self.picam2 = Picamera2()

        # Pi camera configuration:
        self.picam2.video_configuration.size = (width, height)      # Capture video at defined resolution/size
        self.picam2.video_configuration.controls.FrameRate = rate   # Capture video at defined framerate
        self.picam2.video_configuration.encode = 'main'             # Use 'main' stream for encoder
        self.picam2.video_configuration.format = 'BGR888'           # Use this format for OpenCV compatibility
        self.picam2.video_configuration.align()                     # Align stream size (THIS CHANGES THE RES SLIGHTLY!)

        # Apply the configuration to the Pi camera
        self.picam2.configure("video")
        
        # Print camera configurations to confirm correct set up
        print(self.picam2.camera_configuration()['sensor'])
        print(self.picam2.camera_configuration()['main'])
        print(self.picam2.camera_configuration()['controls'])

        # Start the Pi camera
        self.picam2.start()

        # Variable to store whether recording is in progress
        self.recording = False

        # Initialise variable to store recording filename and timestamps
        self.recording_filename = ""
        self.recording_start_ts = None
        self.recording_stop_ts = None

    def getStatus(self):
        """ Return whether or not recording is in progress. """
        return self.recording

    def captureFrame(self):
        """ Capture a frame from the camera for constructing a frame-by-frame preview feed. """
        return self.picam2.capture_array("main")

    def startRecording(self):
        """ Start recording from the Pi camera to the memory buffer. """
        # Only start is not already recording
        if not self.recording:
            # Set recording status
            self.recording = True

            # Initialise a memory buffer to store video recording
            self.recording_membuff = io.BytesIO()
            self.recording_output = FileOutput(self.recording_membuff)

            # Generate starting timestamp
            self.recording_start_ts = datetime.now()
            self.recording_start_ts_str = self.recording_start_ts.strftime("%m-%d-%Y-%H-%M-%S")

            # Generate filename
            self.recording_filename = "recording_" + self.recording_start_ts_str

            # Reset recording end timestamp
            self.recording_stop_ts = None

            # Initialise a 'null' encoder to record without any encoding
            self.recording_encoder = Encoder()

            # Start the recording
            self.picam2.start_recording(
                encoder=self.recording_encoder,  # Null encoder for 'raw' footage
                output=self.recording_output     # Record directly to memory
            )

            # Log to console
            print("Started recording:", self.recording_filename)
        else:
            # Log to console if user requests recording start when already in progress
            print("Cannot start recording, existing recording is in progress!")

    def stopRecording(self):
        """ Stop recording, unload from mamory buffer to disk and convert to MKV with the lossless FFV1 codec. """
        # Only stop recording if already in progress
        if self.recording:
            # Stop the recording
            self.picam2.stop_recording()

            # stop_recording() entirely stops the Pi camera with stop()
            self.picam2.start()

            # Generate stopped timestamp
            self.recording_stop_ts = datetime.now()
            self.recording_stop_ts_str = self.recording_stop_ts.strftime("%m-%d-%Y-%H-%M-%S")

            # Log to console
            print("Stopped recording:"
                + self.recording_filename
                + " (Stopped at: "
                + self.recording_stop_ts_str
                + ")"
            )
            print("Please wait, unloading recording from memory buffer...")

            # Save to file
            with open(self.recording_filename, "xb") as file:
                file.write(self.recording_membuff.getbuffer())

            # Reset recording start timestamp
            self.recording_start_ts = None

            # Close recording memory buffer to discard data and clear RAM space 
            self.recording_membuff.close()

            # Generate encoded recording filename
            self.recording_filename_mkv = self.recording_filename + ".mkv"

            # Log to console
            print("Please wait, encoding raw recording to lossless mkv...: " + self.recording_filename_mkv)

            # Get the aligned resolution of the recording
            (aligned_width, aligned_height) = self.picam2.camera_configuration()['main']['size']

            # FFmpeg command to convert video from BGR888 to MKV (lossless FFV1 codec)
            ffmpeg_command = [
                'ffmpeg',                                   # call FFmpeg
                '-f', 'rawvideo',                           # force raw data type
                '-s', f'{aligned_width}x{aligned_height}',  # width and height
                '-pix_fmt', 'bgr24',                        # set BGR888 format
                '-i', self.recording_filename,              # filename of raw file
                '-c:v', 'ffv1',                             # lossless FFV1 codec
                '-y',                                       # overwrite output file if exists
                '-loglevel', 'error',                       # only show errors in console
                '-stats',                                   # show progress stats
                self.recording_filename_mkv                 # output filename
            ]

            # Run FFmpeg command as a subprocess
            subprocess.run(ffmpeg_command)

            # Delete raw file
            os.remove(self.recording_filename)

            # Log to console
            print("Write complete, it is safe to exit or record again: " + self.recording_filename_mkv)

            # Set the recording status to false to allow program exiting
            self.recording = False
        else:
            # If not already recording, log error to console
            print("Cannot stop recording, program is currently not recording anything!")

    def toggleRecording(self):
        """ Toggle the Pi camera recording. """
        # Turn off if already on
        if self.recording:
            self.stopRecording()
        # Turn on light if already off
        else:
            self.startRecording()

    def shutdown(self):
        """ Gracefully stop the Pi camera instance. """
        # If recording in progress, stop it
        if self.recording:
            print("Exit request received while recording - recording will be stopped.")
            self.stopRecording()

        # Stop the Pi camera instance.
        self.picam2.stop()

        # Log to console
        print("Camera instance has been gracefully stopped.")