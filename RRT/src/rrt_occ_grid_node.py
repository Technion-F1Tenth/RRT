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
WIDTH = 15 #[cells], grid-frame x-direction
LENGTH = 15 #[cells], grid-frame y-direction
RESOLUTION = 0.2 #0.05 #[m/cell]
#NUMBER_OF_POINTS = 10

LIDAR_X_OFFSET = 0.275 #[m]
GRID_UPDATE_TIME = 0.5 #[sec]

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

        #self.transform_bc = tf.TransformBroadcaster()
    
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
        self.transformation_matrix = self.calc_transformation_matrix(self.current_position, current_heading)
    
    def calc_transformation_matrix(self, position, heading):
        return np.array([[np.cos(heading), -np.sin(heading), position[0]], [np.sin(heading), np.cos(heading), position[1]], [0, 0, 1]])

    def in_OccGrid(self, point):
        """ Checks if an (x,y) point, given in the LiDAR frame, is in the occupancy grid """
        if point[0] <= self.resolution*self.width and point[0] >= 0:
            if abs(point[1]) <= self.resolution*self.length/2:
                return True
        return False
    
    def get_OccGrid_indices(self, point):
        """ Returns the indices of where the point is found in the Occupancy Grid; the point is in the laser frame;
        grid index directions are swapped (grid x is LiDAR frame y) """
        x_index = (point[1]+(self.length*self.resolution)/2)//self.resolution
        y_index = point[0]//self.resolution
        return [int(x_index), int(y_index)]
    
    def get_pt_from_idx(self, grid_cell):
        return [grid_cell[0]*self.width*self.resolution, (grid_cell[1]-0.5)*self.length*self.resolution]
    
    def get_grid_indices(self):
        cells_to_check = []
        for i in range(self.width):
            point = (i, 0)
            point2 = (i, self.length-1)
            cells_to_check.append(point)
            cells_to_check.append(point2)
        for j in range(1, self.length-1):
            point3 = (0, j)
            point4 = (self.width-1, j)
            cells_to_check.append(point3)
            cells_to_check.append(point4)
        return cells_to_check
    
    def check_edge_collision(self, point_1, point_2):
        line = Line(point_1, point_2) # Need line between the two points
        points_on_line = [] # Sample points on the line
        
        t = self.resolution #/2 # resolution parameter
        i = 1; dist = 0
        while dist < line.dist:
            dist = np.linalg.norm(np.array([0,0]) - np.array(line.path(i*t)))
            i += 1
            points_on_line.append(line.path(i*t))

        obstacle_seen = False
        for point in points_on_line:
            grid_point = self.get_OccGrid_indices(point)
            if grid_point[0] >= self.width or grid_point[1] >= self.length:
                continue
            if self.grid[grid_point[0], grid_point[1]] == int(100):
                obstacle_seen = True
            if not obstacle_seen and self.grid[grid_point[0], grid_point[1]] < int(100):
                self.grid[grid_point[0], grid_point[1]] = int(0)
        return
    
    def fill_grid(self, scan_data, scan_angles, position, orientation):
        """ Fills the grid using the LiDAR data and the current pose of the car """
        euler_angles = tf.transformations.euler_from_quaternion(orientation)
        heading = euler_angles[2]
        trans = self.calc_transformation_matrix(position, heading)
        
        grid_origin_global_frame = np.dot(trans, [LIDAR_X_OFFSET, -(self.length*self.resolution)/2, 1])
        self.origin_x = grid_origin_global_frame[0] #We use the Pose to determine the position of the central bottom point of the grid.
        self.origin_y = grid_origin_global_frame[1]

        for i in range(len(scan_data)):
            new_point = (scan_data[i]*np.cos(scan_angles[i]), scan_data[i]*np.sin(scan_angles[i])) #We use the Lidar data to determine where there are obstacles; in the LiDAR frame
            if self.in_OccGrid(new_point):
                x_index, y_index = self.get_OccGrid_indices(new_point) # converts point in LiDAR frame to indices of grid
                self.grid[int(x_index)][int(y_index)] = int(100) # obstacles are filled in the grid with a 100 (otherwise are -1) 

        outer_grids_idxs = self.get_grid_indices()
        for o in outer_grids_idxs:
            point = self.get_pt_from_idx(o)
            self.check_edge_collision([0,0], point)
        return

    def fill_message(self):
        """ Puts all the data from the occupancy grid into a ROS message  """
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

#  The regular vector definition of a line
class Line():
    def __init__(self, p0, p1):
        self.p = np.array(p0)
        self.dirn = np.array(p1) - np.array(p0) #direction of line
        self.dist = np.linalg.norm(self.dirn) #magnitude of line
        self.dirn /= self.dist # normalize

    def path(self, t):
        return self.p + t * self.dirn

def main():
    rospy.init_node('rrt_occ_grid_node')
    occ_grid = OccGrid()
    while not rospy.is_shutdown():
        occ_grid.grid.fill(int(-1))
        occ_grid.run()

if __name__ == '__main__':
    print("Occupancy Grid Constructor Initialized...")
    main()