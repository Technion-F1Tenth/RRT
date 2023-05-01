#!/usr/bin/env python
import numpy as np
#from numpy import linalg as LA
#import math, time

import rospy #, tf
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
#from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
#from geometry_msgs.msg import PointStamped
#from geometry_msgs.msg import Pose
#from geometry_msgs.msg import Point
#from ackermann_msgs.msg import AckermannDriveStamped

"""
NOTES: 

Occupancy Grid takes in width, length, resolution
width: Perpendicular to the Car's motion
length: In the direction of the Car's motion
resolution: Determines the size of each grid cell

Check Integer Bullshit on line 78
Need to change all x's to y's and all y's to x's
Switch all y's and x's back. 

"""

# Tunable parameters
LENGTH = 40
WIDTH = 40
RESOLUTION = 0.05
GRID_UPDATE_TIME = 3 #[sec]

class OccGrid(object):
    def __init__(self):
        self.length = LENGTH
        self.width = WIDTH
        self.resolution = RESOLUTION
        self.grid = np.ndarray((self.length, self.width), buffer=np.zeros((self.length, self.width), dtype=np.int), dtype=np.int)
        self.grid.fill(int(0))
        self.origin_y = 0
        self.origin_x = 0
        self.orientation_x = 0
        self.orientation_y = 0
        self.orientation_z = 0
        self.orientation_w = 0

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
    
    #Checks if an (x,y) point is in the occupancy grid
    def in_OccGrid(self, point):
        if (point[0] <= self.origin_y + self.resolution*self.width and point[0] >= self.origin_y):
            if (point[1] <= self.origin_x + self.resolution*self.length and point[1] >= self.origin_x):
                return True
            
        return False 
    
    #Returns the indices of where the point is found in the Occupancy Grid
    def get_OccGrid_indices(self, point):
        y_index = (point[0]-self.origin_y)//self.resolution
        x_index = (point[1]-self.origin_x)//self.resolution
        return (y_index, x_index)
    
    #Fills the grid using the Lidar data and the current Pose of the car
    def fill_grid(self, scan_data, scan_angles, position, orientation):
        
        self.origin_y = position[0]-(self.width*self.resolution)/2 #We use the Pose to determine the position of the central bottom point of the grid.
        self.origin_x = position[1] 
        self.orientation_x = orientation[0]
        self.orientation_y = orientation[1]
        self.orientation_z = orientation[2]
        self.orientation_w = orientation[3]
        
        for i in range(len(scan_data)):
            new_point = (scan_data[i]*np.cos(scan_angles[i]), scan_data[i]*np.sin(scan_angles[i])) #We use the Lidar data to determine where there are obstacles.
            if(self.in_OccGrid(new_point)):
                y_index, x_index = self.get_OccGrid_indices(new_point)
                self.grid[int(y_index)][int(x_index)] = 1 #Obstacles are filled in the grid with a 1 (otherwise are 0). 
                
                m = (new_point[0]-position[0])/(new_point[1]-position[1]) #adding unreachable points using straight line formula
                c = new_point[0]-m*new_point[1]
                
                for j in range(int(x_index), self.length):
                    y_value = (j*self.resolution-c)/m
                    check_point = (y_value, j*self.resolution)
                    if(self.in_OccGrid(check_point)):
                         block_y_index, block_x_index = self.get_OccGrid_indices(check_point)
                         self.grid[int(block_y_index)][int(block_x_index)] = 1
        
    #Puts all the data from the Occupancy grid into a ROS message   
    def fill_message(self):
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = 'map'
        map_msg.header.stamp = rospy.Time.now()
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.length
        map_msg.info.height = self.width
        map_msg.info.origin.position.x = self.origin_y #check this (what does the message see as the origin?)
        map_msg.info.origin.position.y = self.origin_x #check this (what does the message see as the origin?)
        map_msg.info.origin.orientation.x = self.orientation_x 
        map_msg.info.origin.orientation.y = self.orientation_y 
        map_msg.info.origin.orientation.z = self.orientation_z 
        map_msg.info.origin.orientation.w = self.orientation_w 
        map_msg.data = self.grid.flatten()
        return map_msg
    
    def run(self):
        #while True:
        if self.scan_ranges is not None and self.current_position is not None:
            #return #continue
            #print(self.scan_ranges, self.scan_angles, self.current_position, self.current_orientation)
            #print('yo')
            self.fill_grid(self.scan_ranges, self.scan_angles, self.current_position, self.current_orientation)
            map_msg = self.fill_message()
            self.occ_grid_pub.publish(map_msg)
        #rospy.sleep(GRID_UPDATE_TIME) # tune the wait-time between grid updates
    
def main():
    rospy.init_node('rrt_occ_grid_node')
    occ_grid = OccGrid()
    while not rospy.is_shutdown():
        occ_grid.run()

if __name__ == '__main__':
    print("Occupancy Grid Constructor Initialized...")
    main()