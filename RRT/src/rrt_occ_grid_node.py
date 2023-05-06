#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from rrt_auxiliary import calc_transf_matrix, get_OccGrid_idx, check_edge_collision, in_OccGrid

# Parameters
WIDTH = rospy.get_param("/rrt_occ_grid_node/WIDTH") #[cells], number of cells in the grid-frame's x-direction
LENGTH = rospy.get_param("/rrt_occ_grid_node/LENGTH") #[cells], number of cells in the grid-frame's y-direction
RESOLUTION = rospy.get_param("/rrt_occ_grid_node/RESOLUTION") #[m/cell], size of the cells
#LIDAR_X_OFFSET = rospy.get_param("/rrt_occ_grid_node/LIDAR_X_OFFSET") #[m], LiDAR frame translation from car frame
LIDAR_X_OFFSET = rospy.get_param("/f1tenth_simulator/scan_distance_to_base_link") #[m], LiDAR frame translation from car frame
GRID_UPDATE_TIME = rospy.get_param("/rrt_occ_grid_node/GRID_UPDATE_TIME") #[sec]

class OccGrid(object):
    def __init__(self):
        self.width, self.length, self.resolution = WIDTH, LENGTH, RESOLUTION
        self.grid = np.ndarray((self.width, self.length), buffer=np.zeros((self.width, self.length), dtype=np.int), dtype=np.int)
        self.grid.fill(int(-1))
        self.origin_x, self.origin_y = 0, 0
        self.scan_ranges, self.scan_angles, self.current_position, self.current_orientation = None, None, None, None

        scan_topic = rospy.get_param("/rrt_occ_grid_node/scan_topic")
        odom_topic = rospy.get_param("/rrt_occ_grid_node/odom_topic")
        occ_grid_topic = rospy.get_param("/rrt_occ_grid_node/occ_grid_topic")
        
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
        self.transformation_matrix, _ = calc_transf_matrix(self.current_position, self.current_orientation)    
    
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
    
    def fill_grid(self, scan_data, scan_angles, position, orientation):
        """ Fills the grid using the LiDAR data and the current pose of the car """
        trans, _ = calc_transf_matrix(position, orientation)
        grid_origin_global_frame = np.dot(trans, [LIDAR_X_OFFSET, -(self.length*self.resolution)/2, 1])
        self.origin_x = grid_origin_global_frame[0] #We use the Pose to determine the position of the central bottom point of the grid
        self.origin_y = grid_origin_global_frame[1]

        for i in range(len(scan_data)):
            new_point = (scan_data[i]*np.cos(scan_angles[i]), scan_data[i]*np.sin(scan_angles[i])) #We use the Lidar data to determine where there are obstacles; in the LiDAR frame
            if in_OccGrid(new_point, self.resolution, self.width, self.length):
                x_index, y_index = get_OccGrid_idx(new_point, self.resolution, self.length) # converts point in LiDAR frame to indices of grid
                self.grid[x_index][y_index] = int(100) # obstacles are filled in the grid with a 100 (otherwise are -1) 

        outer_grids_idxs = self.get_grid_indices()
        for o in outer_grids_idxs:
            point = self.get_pt_from_idx(o)
            step = self.resolution #/2 # resolution parameter
            self.grid = check_edge_collision([0,0], point, step, self.grid, self.resolution, self.width, self.length)
        return

    def fill_message(self):
        """ Puts all the data from the occupancy grid into a ROS message  """
        map_msg = OccupancyGrid()
        map_msg.header.frame_id, map_msg.header.stamp = 'map', rospy.Time.now()
        map_msg.info.resolution, map_msg.info.width, map_msg.info.height = self.resolution, self.width, self.length
        map_msg.info.origin.position.x, map_msg.info.origin.position.y = self.origin_x, self.origin_y
        map_msg.info.origin.orientation.x, map_msg.info.origin.orientation.y = self.current_orientation[0], self.current_orientation[1]
        map_msg.info.origin.orientation.z, map_msg.info.origin.orientation.w = self.current_orientation[2], self.current_orientation[3]
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