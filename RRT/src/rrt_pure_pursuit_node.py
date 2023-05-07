#!/usr/bin/env python
import rospy, math
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker# , MarkerArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64MultiArray
from rrt_auxiliary import *

NO_RRT = rospy.get_param("/rrt_pure_pursuit_node/NO_RRT")
DRIVE = rospy.get_param("/rrt_pure_pursuit_node/DRIVE")
SLOW_MODE = rospy.get_param("/rrt_pure_pursuit_node/SLOW_MODE") # Mode for always driving the car slowly
VERBOSE = rospy.get_param("/rrt_pure_pursuit_node/VERBOSE")
#CLOCKWISE = False

# Servo steering angle limits
MAX_STEERING_ANGLE = np.deg2rad(80)
MIN_STEERING_ANGLE = -MAX_STEERING_ANGLE

# Tunable parameters
LOOKAHEAD_IDX = rospy.get_param("/rrt_pure_pursuit_node/LOOKAHEAD_IDX")
#MIN_LOOKAHEAD_DISTANCE = 1.0
#MAX_LOOKAHEAD_DISTANCE = 2.0
KP = 0.5*rospy.get_param("/f1tenth_simulator/wheelbase") # 0.325 # wheelbase length (in meters)
MAX_VELOCITY = rospy.get_param("/rrt_pure_pursuit_node/MAX_VELOCITY") #[m/s]

# Choose map
LEVINE = rospy.get_param("/rrt_pure_pursuit_node/LEVINE")
SPIELBERG = rospy.get_param("/rrt_pure_pursuit_node/SPIELBERG")

class PurePursuit(object):
    """ The class that handles the pure pursuit controller """
    def __init__(self):
        #self.min_lookahead_distance = MIN_LOOKAHEAD_DISTANCE
        #self.max_lookahead_distance = MAX_LOOKAHEAD_DISTANCE
        self.waypoints = None
        if NO_RRT:
            if LEVINE:
                opt_waypoints = [i[1:3] for i in set_optimal_waypoints(file_name='levine_raceline')] # in the global frame
            elif SPIELBERG:
                opt_waypoints = [i[:2] for i in set_optimal_waypoints(file_name='Spielberg_raceline')] # in the global frame
            self.waypoints = [opt_waypoints[j] for j in range(len(opt_waypoints)) if j % 3 == 0]

        self.kp = KP
        self.last_wp_idx = None
        #self.flag = False

        drive_topic = rospy.get_param("/rrt_pure_pursuit_node/drive_topic") 
        odom_topic = rospy.get_param("/rrt_pure_pursuit_node/odom_topic")
        target_viz_topic = rospy.get_param("/rrt_pure_pursuit_node/target_viz_topic")
        
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size = 10)
        self.target_pub = rospy.Publisher(target_viz_topic, Marker, queue_size = 10)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=10)

        if not NO_RRT:
            wp_topic = rospy.get_param("/rrt_pure_pursuit_node/wp_topic")
            self.wp_sub = rospy.Subscriber(wp_topic, Float64MultiArray, self.wp_callback, queue_size=10)
        
    def wp_callback(self, wp_msg):
        self.waypoints = np.reshape(wp_msg.data, (wp_msg.layout.dim[0].size, wp_msg.layout.dim[1].size)) # laser frame

    def odom_callback(self, odom_msg):
        current_position = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y] 
        current_orientation = [odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
        #current_heading = get_heading(current_orientation)
        self.transformation_matrix, current_heading = calc_transf_matrix(current_position, current_orientation)
        
        if self.waypoints is not None:
            drive_msg, target_marker = self.pursue(self.waypoints, current_position, current_heading)
            if DRIVE:
                self.drive_pub.publish(drive_msg)
            self.target_pub.publish(target_marker)

    def get_velocity(self, steering_angle):
        """ Given the desired steering angle, returns the appropriate velocity to publish to the car """
        if SLOW_MODE:
            return 0.3 #0.8
        velocity = max(MAX_VELOCITY - abs(np.rad2deg(steering_angle))/50, 0.8) # Velocity varies smoothly with steering angle
        return velocity
    
    def pursue(self, target_pts, current_position, current_heading):
        """ Choose next target point to pursue (lookahead) """
        if NO_RRT:
            if self.last_wp_idx is None:
                nearest_waypoint_idx = find_nearest_waypoint(current_position, self.waypoints)
            else:
                nearest_waypoint_idx = self.last_wp_idx# % (len(self.waypoints)-1)
            if LEVINE:
                j = (nearest_waypoint_idx + LOOKAHEAD_IDX) % (len(self.waypoints))
            elif SPIELBERG:
                j = (nearest_waypoint_idx - LOOKAHEAD_IDX) % (len(self.waypoints))
            target = target_pts[j] #[:2]
        else:
            target = target_pts[1] #[:2]
        
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

        if self.flag:
            target_car_frame = np.dot(self.transformation_matrix, [target_pts[j+3][0], target_pts[j+3][1], 1])
            relative_x = target_car_frame[0]
            dist = self.calc_eucl_distance(target_pts[j+3], current_position)
            print("Checking flag...")
            print(current_position, target_pts[j+3])
            print(dist, relative_x)
            print()

        min_look = self.min_lookahead_distance
        max_look = self.max_lookahead_distance
        target_car_frame = np.dot(self.transformation_matrix, [target_pts[j][0], target_pts[j][1], 1])
        relative_x = target_car_frame[0]
        dist = self.calc_eucl_distance(target_pts[j], current_position) #[:2]
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

            # Problem! At some point, max_look becomes too big, causing the car to look too far ahead and crash!

            dist = self.calc_eucl_distance(target_pts[j], current_position) #[:2]
            target_car_frame = np.dot(self.transformation_matrix, [target_pts[j][0], target_pts[j][1], 1])
            relative_x = target_car_frame[0]
        self.last_wp_idx = j

        if dist > 3:
            self.flag = True
            print(current_position)
            print()

        if current_position[0] < -18:
            print(target_pts[j], dist)
            print()

        if self.calc_eucl_distance(target_pts[j][:2], current_position) < self.calc_eucl_distance(target_pts[i][:2], current_position):
            target = target_pts[j][:2]
        else:
            target = target_pts[i][:2]
        """

        #target_marker = self.create_waypoint_marker(target, nearest_wp=True) # for visualization of the chosen target point
        target_global = laser_to_global(self.transformation_matrix, target)
        target_marker = create_point_marker(target_global, goal=True)

        eucl_d = calc_eucl_distance(target, current_position) #[m]
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
        if VERBOSE:
            print("Steering Angle: {:.3f} [deg], Speed: {:.3f} [m/s]".format(np.rad2deg(steering_angle), speed))

        return drive_msg, target_marker
        
def main():
    rospy.init_node('rrt_pp_node')
    pp = PurePursuit()
    #rate = rospy.Rate(1) # ROS Rate at 5Hz
    #rospy.sleep(1)
    #try:
    #    rospy.spin()
    #except rospy.ROSInterruptException:
    #    rospy.Publisher(rospy.get_param("/rrt_pure_pursuit_node/brake_bool_topic"), Bool, queue_size = 10).publish(True)
        #rospy.loginfo("Termination requested by user...")
    
    while not rospy.is_shutdown():
        rospy.spin()
        #pp.wp_callback()
        #pp.odom_callback()
        #rate.sleep()
    #rospy.Publisher(rospy.get_param("/rrt_pure_pursuit_node/brake_bool_topic"), Bool, queue_size = 10).publish(True)

if __name__ == '__main__':
    print("Pure Pursuit Controller Initialized...")
    main()