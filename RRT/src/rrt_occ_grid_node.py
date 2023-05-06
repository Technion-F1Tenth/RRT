#!/usr/bin/env python
import numpy as np

import rospy, tf
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry

"""
NOTES: 

Occupancy Grid takes in length, width, resolution
length: Perpendicular to the Car's motion
width: In the direction of the Car's motion
resolution: Determines the size of each grid cell

Check Integer Bullshit on line 78
Need to change all x's to y's and all y's to x's
Switch all y's and x's back. 

"""

# Tunable parameters
WIDTH = 50 #[cells], grid-frame x-direction
LENGTH = 50 #[cells], grid-frame y-direction
RESOLUTION = 0.1 #0.05 #[m/cell]

LIDAR_X_OFFSET = 0.275 #[m]
GRID_UPDATE_TIME = 0.1 #[sec]

class OccGrid(object):
    def __init__(self):
        self.width = WIDTH
        self.length = LENGTH
        self.resolution = RESOLUTION
        self.grid = np.ndarray((self.width, self.length), buffer=np.zeros((self.width, self.length), dtype=np.int), dtype=np.int)
        self.grid.fill(int(-1))
        self.origin_x = 0
        self.origin_y = 0
        
        self.scan_ranges = None
        self.scan_angles = None
        self.current_position = None
        self.current_orientation = None

        scan_topic = '/scan' # receiving LiDAR data
        odom_topic = '/odom' # '/pf/pose/odom', receiving odometry data
        occ_grid_topic = '/rrt/occ_grid'
        
        self.scan_sub = rospy.Subscriber(scan_topic, LaserScan, self.scan_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=10)
        self.occ_grid_pub = rospy.Publisher(occ_grid_topic, OccupancyGrid, queue_size=10)
    
    def scan_callback(self, scan_msg):
        """ Records the latest LiDAR data (distances and angles) relative to the LiDAR frame """
        self.scan_ranges = list(scan_msg.ranges)
        self.scan_angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, num=len(self.scan_ranges)) # laser frame

    def odom_callback(self, odom_msg):
        """ Records the car's current position and orientation relative to the global frame, given odometry data """
        self.current_position = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y] 
        self.current_orientation = [odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
        euler_angles = tf.transformations.euler_from_quaternion(self.current_orientation)
        current_heading = euler_angles[2] # also known as the "heading"
        self.transformation_matrix = np.array([[np.cos(current_heading), -np.sin(current_heading), self.current_position[0]], [np.sin(current_heading), np.cos(current_heading), self.current_position[1]], [0, 0, 1]])
    
    def calc_transformation_matrix(self, position, heading):
        return np.array([[np.cos(heading), -np.sin(heading), position[0]], [np.sin(heading), np.cos(heading), position[1]], [0, 0, 1]])

    #Checks if an (x,y) point is in the occupancy grid
    def in_OccGrid(self, point):
        # point is in the LiDAR frame
        if point[0] <= self.resolution*self.width and point[0] >= 0:
            if abs(point[1]) <= self.resolution*self.length/2:
                return True
        return False
    
    #Returns the indices of where the point is found in the Occupancy Grid
    def get_OccGrid_indices(self, point):
        # point in laser frame
        x_index = point[0]//self.resolution
        y_index = (point[1]+(self.length*self.resolution)/2)//self.resolution
        return (x_index, y_index)
    
    #Fills the grid using the Lidar data and the current Pose of the car
    def fill_grid(self, scan_data, scan_angles, position, orientation):
        euler_angles = tf.transformations.euler_from_quaternion(orientation)
        heading = euler_angles[2]
        trans = self.calc_transformation_matrix(position, heading)
        
        grid_origin_global_frame = np.dot(trans, [LIDAR_X_OFFSET, -(self.length*self.resolution)/2, 1])
        self.origin_x = grid_origin_global_frame[0] #We use the Pose to determine the position of the central bottom point of the grid.
        self.origin_y = grid_origin_global_frame[1]

        for i in range(len(scan_data)):
            new_point = (scan_data[i]*np.cos(scan_angles[i]), scan_data[i]*np.sin(scan_angles[i])) #We use the Lidar data to determine where there are obstacles; in the LiDAR frame
            if self.in_OccGrid(new_point):
                y_index, x_index = self.get_OccGrid_indices(new_point) #converts point in LiDAR frame to indicies of grid
                self.grid[int(x_index)][int(y_index)] = int(100) #Obstacles are filled in the grid with a 1 (otherwise are 0). 
                
                """
                t = 0.1; j = 1
                while t*j <= scan_data[i]: # create free cells
                    x = (scan_data[i]-t*j)*np.cos(scan_angles[i]) # x coordinate in lidar frame
                    y = (scan_data[i]-t*j)*np.sin(scan_angles[i]) # y coordinate in lidar frame
                    #x -= lidar_position # x coordinate in base link frame
                    #y -= lidar_position[1] # y coordinate in base link frame
                    #possible_free_cell = np.dot(self.rotMatrix, np.array([0, - self.height // 2])) + [y//self.resolution, x//self.resolution]
                    
                    possible_free_cell = self.get_OccGrid_indices([x, y])
                    #possible_free_cell = [y//self.resolution - self.length//2, x//self.resolution]
                    if int(possible_free_cell[0]) < self.width and int(possible_free_cell[1]) < self.length:
                        if self.grid[int(possible_free_cell[0]), int(possible_free_cell[1])] < int(1):
                            self.grid[int(possible_free_cell[0]), int(possible_free_cell[1])] = int(0)
                    j += 1

                """

                """
                m = (new_point[0]-position[0])/(new_point[1]-position[1]) #adding unreachable points using straight line formula
                c = new_point[0]-m*new_point[1]
                
                for j in range(int(y_index), self.width):
                    x_value = (j*self.resolution-c)/m
                    check_point = (x_value, j*self.resolution)
                    if(self.in_OccGrid(check_point)):
                         block_x_index, block_y_index = self.get_OccGrid_indices(check_point)
                         self.grid[int(block_x_index)][int(block_y_index)] = 1
                """

    #Puts all the data from the Occupancy grid into a ROS message   
    def fill_message(self):
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = 'map'
        map_msg.header.stamp = rospy.Time.now()
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.width
        map_msg.info.height = self.length
        map_msg.info.origin.position.x = self.origin_x
        map_msg.info.origin.position.y = self.origin_y
        map_msg.info.origin.orientation.x = self.current_orientation[0]
        map_msg.info.origin.orientation.y = self.current_orientation[1] 
        map_msg.info.origin.orientation.z = self.current_orientation[2] 
        map_msg.info.origin.orientation.w = self.current_orientation[3]
        map_msg.data = self.grid.flatten()
        return map_msg
    
    def run(self):
        if self.scan_ranges is not None and self.current_position is not None:
            self.fill_grid(self.scan_ranges, self.scan_angles, self.current_position, self.current_orientation)
            map_msg = self.fill_message()
            self.occ_grid_pub.publish(map_msg)
        rospy.sleep(GRID_UPDATE_TIME) # tune the wait-time between grid updates
    
def main():
    rospy.init_node('rrt_occ_grid_node')
    occ_grid = OccGrid()
    while not rospy.is_shutdown():
        occ_grid.grid.fill(int(-1))
        occ_grid.run()

if __name__ == '__main__':
    print("Occupancy Grid Constructor Initialized...")
    main()