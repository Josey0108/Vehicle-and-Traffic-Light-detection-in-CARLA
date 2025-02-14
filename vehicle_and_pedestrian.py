import numpy as np
import pygame
import cv2
import sys
import glob
import os
import time
import threading
import math
from ultralytics import YOLO
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO(r'C:\Users\josey\Downloads\best.pt')  # Correct path to the YOLO model


# Constants for PyGame window
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Initialize PyGame
pygame.init()
display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("CARLA with YOLOv8 Detection")
clock = pygame.time.Clock()

# Global variables
yolo_model = None

def spawn_actors(world, num_vehicles=15, num_pedestrians=15):
    """Spawn vehicles, pedestrians, and traffic lights in the CARLA world."""
    blueprints = world.get_blueprint_library()

    # Spawn vehicles
    vehicle_bp = blueprints.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    vehicles = []
    for i in range(min(num_vehicles, len(spawn_points))):
        vehicle = world.try_spawn_actor(vehicle_bp[i % len(vehicle_bp)], spawn_points[i])
        if vehicle:
            vehicles.append(vehicle)
            vehicle.set_autopilot(True)

    # Spawn pedestrians
    walker_bp = blueprints.filter('walker.pedestrian.*')
    walkers = []
    for i in range(num_pedestrians):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_point.location = loc
            walker = world.try_spawn_actor(walker_bp[i % len(walker_bp)], spawn_point)
            if walker:
                walkers.append(walker)

    return vehicles, walkers

def process_image(image):
    """Process the image from CARLA's camera and detect objects using YOLOv8."""
    # Convert CARLA image to OpenCV format
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]

    # Run YOLO detection
    results = yolo_model(array)

    # Draw bounding boxes
    annotated_image = results[0].plot()

    # Convert back to PyGame format
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(annotated_image.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    pygame.display.flip()


def main():
    global yolo_model

   
    # Connect to the CARLA simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world=client.load_world('Town03')

    # Load YOLOv8 model
    yolo_model = YOLO(r'C:\Users\josey\Downloads\best.pt')  # Correct path to the YOLO model

    # Spawn actors
    vehicles, walkers = spawn_actors(world)

    # Set up a camera
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{WINDOW_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{WINDOW_HEIGHT}')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=2.5, z=2.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicles[0])

    # Start listening to the camera
    camera.listen(lambda image: process_image(image))

    try:
        while True:
            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            clock.tick(30)  # Limit to 30 FPS
    finally:
        print("Cleaning up actors...")
        camera.stop()
        camera.destroy()
        for vehicle in vehicles:
            vehicle.destroy()
        for walker in walkers:
            walker.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()
