import cv2 as cv

class Preview:
    """ Preview window which renders frame-by-frame to an OpenCV window. """

    def __init__(self):
        cv.namedWindow("Pi Camera Feed")

    def getKeypress(self):
        """ Waits 1ms for a keypress and returns it. """
        return cv.waitKey(1)

    def shutdown(self):
        """ Destroys all OpenCV windows for a clean exit. """
        cv.destroyAllWindows()

    def showFrame(self, frame, recording, light):
        """ Updates the window with the new frame, adds status text overlays. """

        # Generate a text-based status for the light source
        match light:
            case True:
                light_status = "LON"
            case False:
                light_status = "LOFF"

        # Generate a text-based status for recording
        match recording:
            case True:
                recording_status = "REC"
            case False:
                recording_status = ""

        # Add the two status text together
        overlay_text = light_status + "    " + recording_status

        # Overlay the text to the frame
        frame_overlay = cv.putText(
            img=frame,
            text=overlay_text,
            org=(10, 25),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255,0,0),
            thickness=1,
            lineType=cv.LINE_AA
        )

        # Render the frame
        cv.imshow("Pi Camera Feed", frame)