import time

class Timer:
    """ Time durations accurately using the performance counter. """

    def __init__(self):
        """ Start the timer on initialisation. """
        self.start()

    def start(self):
        """ Starts the timer. """

        self.start_timestamp = time.perf_counter()
        self.end_timestamp = 0
        self.duration = 0

    def stop(self):
        """ Stops the timer. """

        self.end_timestamp = time.perf_counter()
        self.duration = self.end_timestamp - self.start_timestamp

        # return (self.start_timestamp, self.end_timestamp, self.duration)
        return self.duration


