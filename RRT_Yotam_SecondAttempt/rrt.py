import rospy, tf, os, csv
import numpy as np

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

from rrt_OccGrid import OccGrid
from rrt_MotionPlanner import MotionPlanner
from rrt_PurePursuit import PurePursuit

# Tunable parameters
REPLAN_TIME = 3 #[sec], time between re-planning attempts
LOOKAHEAD_DISTANCE = 0.35 #[m], for Pure Pursuit
KP = 0.3 # for Pure Pursuit

class RRT(object):
    def __init__(self):
        scan_topic = '/scan' # receiving LiDAR data
        odom_topic = '/odom' # '/pf/pose/odom', receiving odometry data
        drive_topic = '/drive' # '/vesc/ackermann_cmd_mux/input/navigation', sending drive commands
        occ_grid_topic = '/rrt/occ_grid'
        wp_viz_topic = '/wp_viz' # sending point visualization data

        self.scan_sub = rospy.Subscriber(scan_topic, LaserScan, self.scan_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=10)
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=10)
        self.occ_grid_pub = rospy.Publisher(occ_grid_topic, OccupancyGrid, queue_size=10)
        #self.nearest_wp_pub = rospy.Publisher(wp_viz_topic+'/nearest', Marker, queue_size=10)

        self.occ_grid = OccGrid()
        #self.planner = MotionPlanner()
        #self.controller = PurePursuit(lookahead_distance=LOOKAHEAD_DISTANCE, kp=KP)
        #self.opt_waypoints = self.set_optimal_waypoints(file_name='attempt1') # in the global frame

    def scan_callback(self, scan_msg):
        """ Records the latest LiDAR data (distances and angles) relative to the LiDAR frame """
        self.scan_ranges = list(scan_msg.ranges)
        self.scan_angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, num=len(self.scan_ranges)) # laser frame

    def odom_callback(self, odom_msg):
        """ Records the car's current position and orientation relative to the global frame, given odometry data """
        self.current_position = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y] 
        euler_angles = tf.transformations.euler_from_quaternion([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])
        self.current_orientation = euler_angles[2] # also called the heading

    def set_optimal_waypoints(self, file_name):
        """ Reads waypoint data from the .csv file produced by the raceline optimizer, inserts it into an array called 'opt_waypoints' """
        file_path = os.path.expanduser('~/f1tenth_ws/logs/{}.csv'.format(file_name))
        opt_waypoints = []
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for waypoint in csv_reader:
                opt_waypoints.append(waypoint)
        for index in range(0, len(opt_waypoints)):
            for point in range(0, len(opt_waypoints[index])):
                opt_waypoints[index][point] = float(opt_waypoints[index][point])
        return opt_waypoints
    
    def update_occ_grid(self):
        """ Updates the occupancy grid,using the car's current position and orientation relative to the global frame, as well as the laser scan data (in the local frame) """
        self.occ_grid.fill_grid(self.scan_ranges, self.scan_angles, self.current_position, self.current_orientation)
        return
    
    def run(self):
        while True:
            self.update_occ_grid()
            map_msg = self.occ_grid.fill_message()
            self.occ_grid_pub.publish(map_msg)

            new_target_pts = self.planner.plan(self.occ_grid, self.opt_waypoints) # we need to also send the transformation from global coordinates to local, since the optimal waypoints are given in global frame but we plan in local frame
            drive_msg, target_marker = self.controller.pursue(new_target_pts, self.current_position, self.current_orientation)
            self.drive_pub.publish(drive_msg)
            self.nearest_wp_pub.publish(target_marker)
            rospy.sleep(REPLAN_TIME) # tune the wait-time between iterations

if __name__ == '__main__':
    rospy.init_node('rrt_node')
    rrt = RRT()
    try:
        rrt.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Termination requested by user...")