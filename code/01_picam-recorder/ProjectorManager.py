import numpy as np

class Projector:
    """
    A simple object which controls the framebuffer to drive a HDMI-interfaced light source.
    """

    def __init__(self, width, height, depth, status):
        self.width = width
        self.height = height
        self.depth = depth
        self.status = status

        # Initialise framebuffer outputs
        self.light_off_output = np.full(
            (self.height, self.width),  # fill into array of this size
            0,                      # fill array with this value (black/off = 0)
            dtype=np.uint16
        ).reshape(-1)               # flatten to 1d array for writing to framebuffer

        self.light_on_output = np.full(
            (self.height, self.width),  # fill into array of this size
            (2 ** self.depth - 1),    # Maximum value for given colour depth (white)
            dtype=np.uint16
        ).reshape(-1)               # flatten to 1d array for writing to framebuffer

        # Initialise the light to the given status
        if self.status:
             self.on()
        else:
             self.off()

    def getStatus(self):
        """ Return whether or not light is on. """
        return self.status

    def on(self):
        """ Turn on the light. """
        try:
            with open('/dev/fb0', 'wb') as buf:
                 buf.write(self.light_on_output.tobytes())
                 self.status = True
        except Exception as e:
            print("Error: couldn't turn on FB light:", e)

    def off(self):
        """ Turn off the light. """
        try:
            with open('/dev/fb0', 'wb') as buf:
                buf.write(self.light_off_output.tobytes())
                self.status = False
        except Exception as e:
            print("Error: couldn't turn off FB light:", e)

    def toggle(self):
        """ Toggle the light. """
        # Turn off if already on
        if self.status:
            self.off()
        # Turn on light if already off
        else:
            self.on()