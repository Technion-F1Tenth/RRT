#!/usr/bin/env python
import rospy, math
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

# Mode for always driving the car slowly
SLOW_MODE = True
# Servo steering angle limits
MAX_STEERING_ANGLE = 60
MIN_STEERING_ANGLE = -MAX_STEERING_ANGLE
# Tunable parameters
LOOKAHEAD_DISTANCE = 0.35
KP = 0.3

class PurePursuit(object):
    """ The class that handles the pure pursuit controller """
    def __init__(self, lookahead_distance=LOOKAHEAD_DISTANCE, kp=KP):
        self.lookahead_distance = lookahead_distance
        self.kp = kp

    def get_velocity(self, steering_angle):
        """ Given the desired steering angle, returns the appropriate velocity to publish to the car """
        if SLOW_MODE:
            return 0.8
        if abs(steering_angle) < np.deg2rad(10):
            velocity = 1.2
        elif abs(steering_angle) < np.deg2rad(20):
            velocity = 1.0
        else:
            velocity = 0.8
        return velocity
    
    def calc_eucl_distance(self, array1, array2):
        return math.sqrt(math.pow(array1[0] - array2[0], 2) + math.pow(array1[1] - array2[1], 2))

    def pursue(self, target_pts, current_position, current_orientation):
        # Choose next target point to pursue (lookahead)
        i = 0
        look = self.lookahead_distance
        while self.calc_eucl_distance(target_pts[i], current_position) < look:
            i += 1
            if (i > len(target_pts) - 1): # if we run out of waypoints to check, go back to the start and decrease the lookahead distance
                i = 0
                look -= 0.1
        goal_marker = self.create_waypoint_marker(target_pts[i], nearest_wp=True) # for visualization of the chosen target point

        eucl_d = self.calc_eucl_distance(target_pts[i], current_position) #[m]
        lookahead_angle = math.atan2(target_pts[i][1]-current_position[1], target_pts[i][0]-current_position[0]) #[rad]
        del_y = eucl_d * math.sin(lookahead_angle - current_orientation) #[m]

        curvature = 2.0*del_y / math.pow(eucl_d,2) #[rad]
        steering_angle = self.kp*curvature #[rad]

        while (steering_angle > np.pi/2) or (steering_angle < -np.pi/2):
            if steering_angle > np.pi/2:
                steering_angle -= np.pi
            elif steering_angle < -np.pi/2:
                steering_angle += np.pi

        # We need to wrap the steering angle to be between -90 and +90 degrees
        #steering_angle = (steering_angle + np.pi) % (2 * np.pi) - np.pi

        # We also need to set the limits for the steering angle based on the servo limits
        steering_angle = min(steering_angle, MAX_STEERING_ANGLE)
        steering_angle = max(steering_angle, MIN_STEERING_ANGLE)

        # Prepare the drive command for pursuing the target point
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle) #[rad]
        speed = self.get_velocity(steering_angle) #[m/s]
        drive_msg.drive.speed = speed
        print("Steering Angle: {:.3f} [deg], Speed: {:.3f} [m/s]".format(np.rad2deg(steering_angle), speed))

        return drive_msg, goal_marker

    #def create_waypoint_marker(self, waypoint_idx, nearest_wp=False):
    def create_waypoint_marker(self, waypoint_position, nearest_wp=False):
        """Given the index of the nearest waypoint, publishes the necessary Marker data to the 'wp_viz' topic for RViZ visualization"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = 2 # sphere
        marker.action = 0 # add the marker
        marker.pose.position.x = waypoint_position[0]
        marker.pose.position.y = waypoint_position[1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        
        # different color/size for the whole waypoint array and the nearest waypoint
        if nearest_wp:
            marker.scale.x *= 2
            marker.scale.y *= 2
            marker.scale.z *= 2
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        return marker