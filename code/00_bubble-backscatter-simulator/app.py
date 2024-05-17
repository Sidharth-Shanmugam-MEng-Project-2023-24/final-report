import pygame
import random
import uuid
from datetime import datetime
import csv
import os

WINDOW_WIDTH = 800          # Width of simulation window
WINDOW_HEIGHT = 600         # Height of simulation window
SIMULATION_FPS = 60         # Simulation frame rate

MAX_VELOCITY_X = 50         # Maximum possible x-axis velocity for particles
MIN_VELOCITY_X = -50        # Minimum possible x-axis velocity for particles
MAX_VELOCITY_Y = 200        # Maximum possible y-axis velocity for particles
MIN_VELOCITY_Y = 50         # Minimum possible y-axis velocity for particles
VELOCITY_CHANGE_RATE = 0.1  # Rate at which velocity changes

MAX_RADIUS = 6          # Maximum possible spawn-time particle radius
MIN_RADIUS = 2          # Minimum possible spawn-time particle radius
HEIGHT_RADIUS_GROW_MULTIPLIER = 0.05    # Grow multiplier as particle rises

BG_COLOUR = (0, 0, 0)   # Background colour (default is black)
PARTICLE_COLOUR = (255, 255, 255)   # Particle colour (default is white)

MAX_PARTICLES = 300                     # Maximum particles on screen
CONSTANT_PARTICLE_GENERATION = True    # Whether to constantly maintain MAX_PARTICLES
PARTICLE_RANDOMISE_VELOCITY_PROB = 0.01 # Probability or particle movement randomisation

class Bubble:
    """ A bubble-based backscatter particle. """
    def __init__(self):
        self.id = uuid.uuid4()                      # Initialise ID
        self.x = random.randint(0, WINDOW_WIDTH)    # Randomly initialise x-axis spawn position
        self.y = WINDOW_HEIGHT                      # Initialise y-axis spawn position to bottom
        self.radius = random.randint(MIN_RADIUS, MAX_RADIUS)    # Randomly initialise radius
        self.randomiseVelocities()                  # Randomly initialise velocities
        self.velocity_x = self.target_velocity_x    # Randomly initialise velocities 
        self.velocity_y = self.target_velocity_y    # Randomly initialise velocities

    def randomiseVelocities(self):
        """ Randomly update x and y-axis particle velocities. """
        self.target_velocity_y = random.uniform(MIN_VELOCITY_Y, MAX_VELOCITY_Y)  # Update vertical velocity randomly
        self.target_velocity_x = random.uniform(MIN_VELOCITY_X, MAX_VELOCITY_X)  # Update horizontal velocity randomly

    def move(self, delta_t):
        """ Update position and radius based on velocity and position. """
        self.y -= self.velocity_y * delta_t     # Update vertical position
        self.x += self.velocity_x * delta_t     # Update horizontal position

        # Gradually adjust velocities towards target velocities
        self.velocity_y += (self.target_velocity_y - self.velocity_y) * VELOCITY_CHANGE_RATE
        self.velocity_x += (self.target_velocity_x - self.velocity_x) * VELOCITY_CHANGE_RATE

        # As a bubble travels upwards, it must get bigger
        height_factor = (WINDOW_HEIGHT - self.y) / WINDOW_HEIGHT        # Calculate height factor
        self.radius += HEIGHT_RADIUS_GROW_MULTIPLIER * height_factor    # Grow radius based on height factor and multiplier

    def draw(self, screen):
        """ Draw the bubble-based backscatter particle on screen. """
        pygame.draw.circle(screen, PARTICLE_COLOUR, (int(self.x), int(self.y)), self.radius)

if __name__ == "__main__":
    # Initialise pygame module
    pygame.init()

    # Initialise pygame window and set window title
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Bubble Simulation")

    # Create clock object to track time
    clock = pygame.time.Clock()

    # Initialise runtime variables
    running = True          # Stores whether the simulation is running
    delta_t = 0             # Stores the time in seconds since the last frame
    bubbles = []            # Array of bubbles which are on the screen
    bubble_positions = {}   # Dictionary storing individual bubble positions in each frame

    # Log timestamp when simulation starts
    simulation_start_ts = datetime.now()

    # Generate export filename
    export_filename_generic = "export_" + simulation_start_ts.strftime("%m-%d-%Y-%H-%M-%S")

    # Create folder for the frame image export
    if not os.path.exists(export_filename_generic):
        os.makedirs(export_filename_generic)

    if not CONSTANT_PARTICLE_GENERATION:
        while len(bubbles) < MAX_PARTICLES:
            bubbles.append(Bubble())

    # Initialise frame counter
    frame_num = 0

    while running:
        # Fill window with the background colour
        screen.fill(BG_COLOUR)

        # Generate new bubbles randomly and constantly
        if CONSTANT_PARTICLE_GENERATION:
            while len(bubbles) < MAX_PARTICLES:
                bubbles.append(Bubble())

        # Log bubble positions and radius in this frame
        for bubble in bubbles:
            # If this bubble has not been previously tracked...
            if bubble.id not in bubble_positions:
                # ... Initialise an empty list to track in subsequent frames
                bubble_positions[bubble.id] = {'positions': [], 'radius': []}
            # Append the list with the bubble's position in this frame
            bubble_positions[bubble.id]['positions'].append((bubble.x, bubble.y))
            bubble_positions[bubble.id]['radius'].append(bubble.radius)

        # Update and draw bubbles
        for bubble in bubbles:
            bubble.move(delta_t)
            bubble.draw(screen)

        # Randomly update bubble velocities
        if random.random() < PARTICLE_RANDOMISE_VELOCITY_PROB:
            for bubble in bubbles:
                bubble.randomiseVelocities()

        # Remove bubbles that are out of the screen
        bubbles = [bubble for bubble in bubbles if bubble.y > -bubble.radius]

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update the contents of the full display
        pygame.display.flip()

        # Export frame as png
        filename = os.path.join(export_filename_generic, f"{frame_num}.png")
        pygame.image.save(screen, filename)

        # Limit FPS and store time in seconds since last frame
        delta_t = clock.tick(SIMULATION_FPS) / 1000

        # Increment frame number counter
        frame_num += 1

    # Generate CSV dataset of simulated bubble positions
    export_filename_csv = export_filename_generic + ".csv"

    with open(export_filename_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Bubble ID', 'Frame #', 'X Position', 'Y Position', 'Radius'])
        # Write bubble positions and radius
        for bubble_id, data in bubble_positions.items():
            for frame, (x, y) in enumerate(data['positions']):
                radius = data['radius'][frame]
                writer.writerow([bubble_id, frame, x, y, radius])

    pygame.quit()