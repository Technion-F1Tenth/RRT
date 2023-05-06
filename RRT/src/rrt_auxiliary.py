#!/usr/bin/env python
import numpy as np
import rospy, tf, os, csv
from visualization_msgs.msg import Marker, MarkerArray

## Geometric functions

class Line():
    def __init__(self, p0, p1):
        self.p = np.array(p0)
        self.dirn = np.array(p1) - np.array(p0) #direction of line
        self.dist = np.linalg.norm(self.dirn) #magnitude of line
        self.dirn /= self.dist # normalize

    def path(self, t):
        return self.p + t * self.dirn

def calc_eucl_distance(array1, array2):
    return np.sqrt(np.power(array1[0] - array2[0], 2) + np.power(array1[1] - array2[1], 2))
 
def calc_transf_matrix(position, orientation):
    euler_angles = tf.transformations.euler_from_quaternion(orientation)
    heading = euler_angles[2]
    return np.array([[np.cos(heading), -np.sin(heading), position[0]], [np.sin(heading), np.cos(heading), position[1]], [0, 0, 1]]), heading

## Occupancy grid functions

def get_OccGrid_idx(point, grid_resolution, grid_length):
    """ Returns the indices of where the point is found in the Occupancy Grid; the point is in the laser frame;
        grid index directions are swapped (grid x is LiDAR frame y) """
    x_index = (point[1]+(grid_length*grid_resolution)/2)//grid_resolution
    y_index = point[0]//grid_resolution
    return [int(x_index), int(y_index)]

def in_OccGrid(point, grid_resolution, grid_width, grid_length):
    """ Checks if an (x,y) point, given in the LiDAR frame, is in the occupancy grid """
    if point[0] <= grid_resolution*grid_width and point[0] >= 0:
        if abs(point[1]) <= grid_resolution*grid_length/2:
            return True
    return False

def check_edge_collision(point_1, point_2, step, grid, grid_resolution, grid_width, grid_length, rrt=False):
        line = Line(point_1, point_2) # Need line between the two points
        points_on_line = [] # Sample points on the line
        
        i = 1; dist = 0
        while dist < line.dist:
            points_on_line.append(line.path(i*step))
            i += 1
            dist = np.linalg.norm(np.array(point_1) - np.array(line.path(i*step)))

        obstacle_seen = False
        for point in points_on_line:
            grid_point = get_OccGrid_idx(point, grid_resolution, grid_length)
            if grid_point[0] >= grid_width or grid_point[1] >= grid_length:
                continue

            if rrt: # edge collision detection for RRT tree building
                if grid[grid_point[0], grid_point[1]] != int(0):
                    return False
            else: # occupancy grid mapping
                if grid[grid_point[0], grid_point[1]] == int(100):
                    obstacle_seen = True
                if not obstacle_seen and grid[grid_point[0], grid_point[1]] < int(100):
                    grid[grid_point[0], grid_point[1]] = int(0)
        if rrt:
            return True
        else:
            return grid
        
## Waypoint functions

def set_optimal_waypoints(file_name):
    """ Reads waypoint data from the .csv file produced by the raceline optimizer, inserts it into an array called 'opt_waypoints' """
    file_path = os.path.expanduser('~/f1tenth_ws/logs/{}.csv'.format(file_name))
    opt_waypoints = []
    with open(file_path) as csv_file:
        #csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.reader(csv_file, delimiter=';')
        for waypoint in csv_reader:
            opt_waypoints.append(waypoint)
    for index in range(0, len(opt_waypoints)):
        for point in range(0, len(opt_waypoints[index])):
            opt_waypoints[index][point] = float(opt_waypoints[index][point])
    return opt_waypoints

def find_nearest_waypoint(current_position, waypoints):
    """ Given the current XY position of the car, returns the index of the nearest waypoint """
    ranges = []
    for index in range(len(waypoints)):
        eucl_d = calc_eucl_distance(current_position, waypoints[index])
        ranges.append(eucl_d)
    return ranges.index(min(ranges))

## Visualization functions

def create_point_marker(pos, goal=False):
    """ Given the position of a point, creates the necessary Marker data for RViZ visualization """
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.id = 0
    marker.type = 2 # sphere
    marker.action = 0 # add the marker
    marker.pose.position.x, marker.pose.position.y = pos[0], pos[1]
    marker.pose.position.z = 0.0
    marker.pose.orientation.x, marker.pose.orientation.y = 0.0, 0.0
    marker.pose.orientation.z, marker.pose.orientation.w = 0.0, 1.0
    marker.scale.x, marker.scale.y, marker.scale.z = 0.1, 0.1, 0.1
    marker.color.a, marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0, 1.0
    if goal:
        marker.scale.x, marker.scale.y, marker.scale.z = 0.2, 0.2, 0.2
        marker.color.g, marker.color.b = 1.0, 0.0
    return marker

def create_raceline_marker_array(opt_waypoints):
    markerArray = MarkerArray()
    for i in range(len(opt_waypoints)):
        marker = create_point_marker(opt_waypoints[i])
        markerArray.markers.append(marker)
    id = 0
    for m in markerArray.markers:
        m.id = id
        id += 1
    return markerArray