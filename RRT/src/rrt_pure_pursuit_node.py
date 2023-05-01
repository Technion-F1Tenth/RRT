#!/usr/bin/env python
import rospy, math, tf
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

# Mode for always driving the car slowly
SLOW_MODE = False
ADAPTIVE = True
CLOCKWISE = False
# Servo steering angle limits
MAX_STEERING_ANGLE = np.deg2rad(80)
MIN_STEERING_ANGLE = -MAX_STEERING_ANGLE
# Tunable parameters
MIN_LOOKAHEAD_DISTANCE = 1.0
MAX_LOOKAHEAD_DISTANCE = 1.5
KP = 0.325

MAX_VELOCITY = 2.0

class PurePursuit(object):
    """ The class that handles the pure pursuit controller """
    def __init__(self):
        self.min_lookahead_distance = MIN_LOOKAHEAD_DISTANCE
        self.max_lookahead_distance = MAX_LOOKAHEAD_DISTANCE
        self.kp = KP
        self.waypoints = None

        drive_topic = '/drive' # '/vesc/ackermann_cmd_mux/input/navigation', sending drive commands
        target_viz_topic = '/rrt/target_viz' # sending point visualization data
        odom_topic = '/odom' # '/pf/pose/odom', receiving odometry data
        wp_topic = '/rrt/waypoints'
        
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=10)
        self.target_pub = rospy.Publisher(target_viz_topic, Marker, queue_size=10)
        self.wp_sub = rospy.Subscriber(wp_topic, Float64MultiArray, self.wp_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=10)
        
    def wp_callback(self, wp_msg):
        self.waypoints = np.reshape(wp_msg.data, (wp_msg.layout.dim[0].size, wp_msg.layout.dim[1].size))
        #self.waypoints = [i[1:3] for i in self.waypoints]
        #print(self.waypoints)

    def odom_callback(self, odom_msg):
        current_position = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y] 
        euler_angles = tf.transformations.euler_from_quaternion([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])
        current_heading = euler_angles[2] # also known as the "heading"

        self.transformation_matrix = np.array([[np.cos(-current_heading), -np.sin(-current_heading), current_position[0]], [np.sin(-current_heading), np.cos(-current_heading), current_position[1]], [0, 0, 1]])

        #print(np.rad2deg(current_heading))
        if self.waypoints is not None:
            drive_msg, target_marker = self.pursue(self.waypoints, current_position, current_heading)
            self.drive_pub.publish(drive_msg)
            self.target_pub.publish(target_marker)

    def get_velocity(self, steering_angle):
        """ Given the desired steering angle, returns the appropriate velocity to publish to the car """
        if SLOW_MODE:
            return 0.8
        if ADAPTIVE:
            velocity = max(MAX_VELOCITY - abs(np.rad2deg(steering_angle))/50, 0.8) # Velocity varies smoothly with steering angle
            return velocity
        if abs(steering_angle) < np.deg2rad(10):
            velocity = 1.5
        elif abs(steering_angle) < np.deg2rad(20):
            velocity = 1.0
        else:
            velocity = 0.8
        return velocity
    
    def calc_eucl_distance(self, array1, array2):
        return math.sqrt(math.pow(array1[0] - array2[0], 2) + math.pow(array1[1] - array2[1], 2))

    def find_nearest_waypoint(self, current_position):
        """ Given the current XY position of the car, returns the index of the nearest waypoint """
        ranges = []
        for index in range(len(self.waypoints)):
            eucl_d = self.calc_eucl_distance(current_position, self.waypoints[index][:2])
            ranges.append(eucl_d)
        return ranges.index(min(ranges))

    def pursue(self, target_pts, current_position, current_heading):
        # Choose next target point to pursue (lookahead)
        nearest_waypoint_idx = self.find_nearest_waypoint(current_position)
        j = nearest_waypoint_idx
        """
        i = nearest_waypoint_idx #0
        
        #print(i)
        look = self.lookahead_distance
        relative_x = -1
        while self.calc_eucl_distance(target_pts[i][:2], current_position) < look or relative_x < 0:
            #print(self.calc_eucl_distance(target_pts[i][:2], current_position))
            target_car_frame = np.dot(self.transformation_matrix, [target_pts[i][0], target_pts[i][1], 1])
            #if target_car_frame[1] < 0:
            #    pass
            #print()
            #print(current_position)
            #print(list(target_pts[i][:2]), list(target_car_frame[:2]))
            relative_x = target_car_frame[0]
            #print(i)
            i += 1
            if (i > len(target_pts) - 1): # if we run out of waypoints to check, go back to the start and decrease the lookahead distance
                i = 0
                look -= 0.1

        """
        
        min_look = self.min_lookahead_distance
        max_look = self.max_lookahead_distance
        target_car_frame = np.dot(self.transformation_matrix, [target_pts[j][0], target_pts[j][1], 1])
        relative_x = target_car_frame[0]
        dist = self.calc_eucl_distance(target_pts[j][:2], current_position)
        while dist < min_look or dist > max_look or relative_x < 0:
            if CLOCKWISE:
                j -= 1                
                if j < 0: # if we run out of waypoints to check, go back to the start and decrease the lookahead distance
                    j = len(target_pts)-1
                    min_look -= 0.1
                    max_look += 0.1
            else:
                j += 1
                if j > len(target_pts) - 1: # if we run out of waypoints to check, go back to the start and decrease the lookahead distance
                    j = 0
                    min_look -= 0.1
                    max_look += 0.1

            dist = self.calc_eucl_distance(target_pts[j][:2], current_position)
            target_car_frame = np.dot(self.transformation_matrix, [target_pts[j][0], target_pts[j][1], 1])
            relative_x = target_car_frame[0]
        
        #print(target_pts[i][:2])
        #print(current_position)
        """
        if self.calc_eucl_distance(target_pts[j][:2], current_position) < self.calc_eucl_distance(target_pts[i][:2], current_position):
            target = target_pts[j][:2]
        else:
            target = target_pts[i][:2]
        """
        target = target_pts[j][:2]

        #print(relative_x)
        #print(current_position)
        #print(target)
        #print(target_car_frame[:2])
        

        target_marker = self.create_waypoint_marker(target, nearest_wp=True) # for visualization of the chosen target point

        eucl_d = self.calc_eucl_distance(target, current_position) #[m]
        lookahead_angle = math.atan2(target[1]-current_position[1], target[0]-current_position[0]) #[rad]
        del_y = eucl_d * math.sin(lookahead_angle - current_heading) #[m]

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

        return drive_msg, target_marker

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
    
def main():
    rospy.init_node('rrt_pp_node')
    pp = PurePursuit()
    while not rospy.is_shutdown():
        rospy.spin()
        #rospy.sleep(3)

if __name__ == '__main__':
    print("Pure Pursuit Controller Initialized...")
    main()