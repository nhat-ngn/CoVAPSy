from high_level.src.programs.utils.driver import Driver
from typing import cast
import numpy as np
from controller import Camera, Lidar
from controller.touch_sensor import TouchSensor
from vehicle import Close_loop_Driver

# AlexNet Controller
MODEL_PATH = "f{high_level.models.model_CNN1DAlexNetExtractorMemory.onnx}"

Cl_driver = Close_loop_Driver()

basicTimeStep = int(Cl_driver.getBasicTimeStep())
sensorTime = basicTimeStep // 4

ai_driver = Driver()
ai_driver.load_model(MODEL_PATH)

lidar = Lidar("Hokuyo")
lidar.enable(sensorTime)
lidar.enablePointCloud()

camera = cast(Camera, Cl_driver.getDevice("RASPI_Camera_V2"))
camera.enable(sensorTime)

touch_sensor = cast(TouchSensor, Cl_driver.getDevice("touch_sensor"))
touch_sensor.enable(sensorTime)

# speed in km/h
speed = 0
maxSpeed = 28  # km/h

# steering angle
angle = 0
maxangle = 0.28  # rad (strange, the car is defined with a limit of 0.31 rad...

backwards_duration = 2000  # ms
stop_duration = 3000  # ms

death_count = 0

# reset speed and direction to zero
Cl_driver.setSteeringAngle(angle)
Cl_driver.setCruisingSpeed(speed)


def backwards(lidar_data, camera_data):
    for _ in range(backwards_duration // basicTimeStep):
        speed = -1
        avg_color = np.mean(camera_data, axis=0) / 255

        angle = -0.5 * avg_color[0] + 0.5 * avg_color[1]

        Cl_driver.setCruisingSpeed(speed)
        Cl_driver.setSteeringAngle(angle)
        Cl_driver.step()

    # makes sure it doesn't go backwards again because there is a wall behind the car
    speed = 1
    angle = 0
    for _ in range(10):
        Cl_driver.setCruisingSpeed(speed)
        Cl_driver.setSteeringAngle(angle)
        Cl_driver.step()


def stop():
    Cl_driver.setCruisingSpeed(0)
    Cl_driver.setSteeringAngle(0)
    for _ in range(stop_duration // basicTimeStep):
        Cl_driver.step()
    # will be reset by the controller_world_supervisor.py


while Cl_driver.step() != -1:
    lidar_data = np.nan_to_num(lidar.getRangeImage(), nan=0.0, posinf=30.0)
    camera_data = np.nan_to_num(camera.getImageArray(), nan=0.0, posinf=30.0).squeeze()
    sensor_data = touch_sensor.getValue()

    # goes backwards
    if sensor_data == 1:
        death_count += 1
        if death_count < 3:
            # print("backwards", driver.getName())
            backwards(lidar_data, camera_data)
        else:
            # print("stop", driver.getName())
            death_count = 0
            stop()

    # speed = 1  # km/h
    # # the steering angle is the difference between ray measurements
    # avg_color = np.mean(camera_data, axis=0) / 255

    # i = np.argmin(lidar_data)
    # m = lidar_data[i]
    # if m <= 0.7:
    #     angle = 0.3 if i <= 64 else -0.3
    # else:
    #     angle = 0.3 * avg_color[0] - 0.3 * avg_color[1]

    angle, speed = ai_driver.ai_update_lidar_camera(lidar_data, camera_data)
    speed = speed * 3.6 # km/h
    Cl_driver.setCruisingSpeed(speed)
    Cl_driver.setSteeringAngle(angle)
