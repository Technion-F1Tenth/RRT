#!/usr/bin/env python
import rospy, os, csv, tf
import numpy as np

from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float64MultiArray, MultiArrayDimension #, MultiArrayLayout
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

#from f1tenth_ws.src.RRT.src.rrt_tree import 
from rrt_tree import RRTTree


# RRT parameters
GOAL_BIAS = 0.05

# Choose map
LEVINE = True
SPIELBERG = False

class MotionPlanner(object):
    def __init__(self):
        ## Occupancy grid stuff
        self.occ_grid = None
        self.grid_resolution = None
        self.grid_width = None
        self.grid_length = None

        ## Global waypoint stuff
        if LEVINE:
            opt_waypoints = [i[1:3] for i in self.set_optimal_waypoints(file_name='levine_raceline')] # in the global frame
        elif SPIELBERG:
            opt_waypoints = [i[:2] for i in self.set_optimal_waypoints(file_name='Spielberg_raceline')] # in the global frame
        self.opt_waypoints = [opt_waypoints[j] for j in range(len(opt_waypoints)) if j % 3 == 0]

        ## RRT Stuff
        self.goal_bias = GOAL_BIAS
        self.goal_pos = None
        self.tree = RRTTree()
        start_pos = [0,0] # LiDAR frame
        self.tree.add_vertex(start_pos, parent=None)

        ## Data Topics
        wp_topic = '/rrt/waypoints' # sending waypoints to this topic
        occ_grid_topic = '/rrt/occ_grid' # reading the occupancy grid from this topic
        odom_topic = '/odom' # '/pf/pose/odom', receiving odometry data

        self.wp_pub = rospy.Publisher(wp_topic, Float64MultiArray, queue_size=10)
        self.occ_grid_sub = rospy.Subscriber(occ_grid_topic, OccupancyGrid, self.occ_grid_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=10)
        
        ## Visualization Topics
        global_wp_viz_topic = '/rrt/global_wp_viz' # sending waypoint visualization data to this topic
        sample_viz_topic = '/rrt/sample_viz' # sending sample visualization data to this topic
        plan_viz_topic = '/rrt/plan_viz' # sending plan visualization data to this topic

        self.global_wp_viz_pub = rospy.Publisher(global_wp_viz_topic, MarkerArray, queue_size=10)
        self.sample_viz_pub = rospy.Publisher(sample_viz_topic, MarkerArray, queue_size = 10)
        self.plan_viz_pub = rospy.Publisher(plan_viz_topic, Marker, queue_size = 10)

    def occ_grid_callback(self, grid_msg):
        self.grid_resolution = grid_msg.info.resolution
        self.grid_width = grid_msg.info.width
        self.grid_length = grid_msg.info.height
        self.occ_grid = np.reshape(grid_msg.data, (self.grid_width, self.grid_length))
        new_waypoints = self.plan() #[i[1:3] for i in self.plan()]
        
        wp_msg = self.create_waypoint_message(new_waypoints)
        self.wp_pub.publish(wp_msg)

        plan_marker = self.create_plan_marker(new_waypoints)
        self.plan_viz_pub.publish(plan_marker)

    def odom_callback(self, odom_msg):
        """ Records the car's current position and orientation relative to the global frame, given odometry data """
        self.current_position = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y] 
        self.current_orientation = [odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
        self.transformation_matrix = self.calc_transformation_matrix(self.current_position, self.current_orientation)

    def calc_eucl_distance(self, array1, array2):
        return np.sqrt(np.power(array1[0] - array2[0], 2) + np.power(array1[1] - array2[1], 2))

    def calc_transformation_matrix(self, position, orientation):
        euler_angles = tf.transformations.euler_from_quaternion(orientation)
        heading = euler_angles[2]
        return np.array([[np.cos(heading), -np.sin(heading), position[0]], [np.sin(heading), np.cos(heading), position[1]], [0, 0, 1]])

    def set_optimal_waypoints(self, file_name):
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
    
    def create_waypoint_message(self, waypoints):
        wp_msg = Float64MultiArray()
        wp_msg.data = np.reshape(waypoints, (1,-1))[0]
        dim0 = MultiArrayDimension()
        dim0.size = np.shape(waypoints)[0]
        dim1 = MultiArrayDimension()
        dim1.size = np.shape(waypoints)[1]

        #layout = MultiArrayLayout()
        #layout.dim = list()
        wp_msg.layout.dim = [dim0, dim1]
        #wp_msg.layout.dim[1].size = np.shape(new_waypoints)[1]
        return wp_msg

    def find_nearest_waypoint(self, current_position):
        # Given the current XY position of the car, returns the index of the nearest waypoint
        ranges = []
        for index in range(len(self.opt_waypoints)):
            eucl_d = self.calc_eucl_distance(current_position, self.opt_waypoints[index]) #[:2]
            ranges.append(eucl_d)
        return ranges.index(min(ranges))
    
    def in_OccGrid(self, point):
        """ Checks if an (x,y) point, given in the LiDAR frame, is in the occupancy grid """
        if point[0] <= self.grid_resolution*self.grid_width and point[0] >= 0:
            if abs(point[1]) <= self.grid_resolution*self.grid_length/2:
                return True
        return False
    
    def get_OccGrid_indices(self, point):
        """ Returns the indices of where the point (LiDAR frame) is found in the Occupancy Grid """
        x_index = point[0]//self.grid_resolution
        y_index = (point[1]+(self.grid_length*self.grid_resolution)/2)//self.grid_resolution
        return (int(x_index), int(y_index))

    def get_goal_point(self):
        nearest_wp_idx = self.find_nearest_waypoint(self.current_position)
        best_wp_global = self.opt_waypoints[nearest_wp_idx]
        best_wp_lidar = np.dot(np.linalg.inv(self.transformation_matrix), [best_wp_global[0], best_wp_global[1], 1])[:2]
        counter = 1
        while True:
            new_idx = (nearest_wp_idx + counter) % len(self.opt_waypoints)
            new_wp_global = self.opt_waypoints[new_idx]
            new_wp_lidar = np.dot(np.linalg.inv(self.transformation_matrix), [new_wp_global[0], new_wp_global[1], 1])[:2]
            if not self.in_OccGrid(new_wp_lidar):
                return best_wp_lidar 
            
            waypoint_grid_indices = self.get_OccGrid_indices(new_wp_lidar)
            if self.occ_grid[waypoint_grid_indices[0]][waypoint_grid_indices[1]] == int(100):
                counter += 1
                continue
                
            best_idx = (nearest_wp_idx + counter) % len(self.opt_waypoints)
            best_wp_global = self.opt_waypoints[best_idx]
            best_wp_lidar = np.dot(np.linalg.inv(self.transformation_matrix), [best_wp_global[0], best_wp_global[1], 1])[:2]
            counter = counter + 1
        return None
    
    def collision_free_pos(self, pos):
        grid_point = self.get_OccGrid_indices(pos)
        if self.occ_grid[grid_point[1], grid_point[0]] == int(0):
            return True
        else:
            False

    def plan(self):
        sample_array = MarkerArray()
        goal_pos = self.get_goal_point()

        goal_added = False; num_iter = 0
        while not goal_added:
            num_iter += 1
            goal = False

            # Sampling step
            p = np.random.uniform() # goal biasing
            if p < self.goal_bias:
                pos = goal_pos
                goal = True
            else:
                pos = self.sample()

            #pos = self.sample()
            # Verify that the sample is in free space
            if not self.collision_free_pos(pos):
                continue

            # Get nearest vertex to the sample
            nearest_vert = self.tree.get_nearest_pos(pos)
            nearest_vert_idx = nearest_vert[0]

            """
            if self.ext_mode == 'E2':
                pos = self.extend(nearest_vert[1], pos)
                if not self.collision_free_pos(pos, self.occ_grid):
                    continue
                #print(pos)

            if np.linalg.norm(np.subtract(pos,self.curr_pos),2) > GOAL_DISTANCE:# and num_iter > 10:# and np.linalg.norm(pos,2) < MAX_SAMPLE_RADIUS:
                goal = True   
                #print("yp")
            """
             #if num_iter < 1000:
            sample_marker = self.create_sample_marker(pos, goal)
            sample_array.markers.append(sample_marker)
            self.create_waypoint_marker_array(sample_array)
            
            if num_iter > 100:
                goal_added = True

            # Partial extensions, if enabled
            #if self.ext_mode == 'E2':
            #    pos = self.extend(nearest_vert[1], pos) # config = x_new
                #if not env.config_validity_checker(config):
                #    continue
            
            # Check obstacle-collision for potential edge
            if self.edge_validity_checker(pos, nearest_vert[1]):
                pos_idx = self.tree.add_vertex(pos, nearest_vert)
                #cost = self.tree.compute_distance(config, nearest_vert[1])
                self.tree.add_edge(nearest_vert_idx, pos_idx)#, cost)
                if goal:# and self.ext_mode == 'E1':
                    goal_added = True
            else:
                goal_added = False

        # Record the plan
        plan = []
        plan.append(pos)
        child_idx = pos_idx
        parent_pos = nearest_vert[1]
        while self.tree.edges[child_idx]:
            plan.append(parent_pos)
            child_idx = self.tree.get_idx_for_pos(parent_pos)
            parent_idx = self.tree.edges[child_idx] # new parent
            parent_pos = self.tree.vertices[parent_idx].pos
        plan.append(parent_pos)
        plan = plan[::-1]
        return plan
    
    def edge_validity_checker(self, pos, nearest_vert):
        return True
    
    def sample(self):
        """ This method should randomly sample the free space, and returns a viable point """
        x = np.random.uniform(low=0, high=self.grid_resolution*self.grid_width)
        y = np.random.uniform(low=-self.grid_resolution*self.grid_length/2, high=self.grid_resolution*self.grid_length/2)
        return [x,y]
    
    def create_sample_marker(self, pos, goal=False):
        # Given the position of a sample, publishes the necessary Marker data to the 'wp_viz' topic for RViZ visualization
        sample_global_frame = np.dot(self.transformation_matrix, [pos[0], pos[1], 1])
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = 2 # sphere
        marker.action = 0 # add the marker
        marker.pose.position.x = sample_global_frame[0]
        marker.pose.position.y = sample_global_frame[1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        if goal:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
        return marker

    def create_waypoint_marker_array(self, markerArray, sample=True):
        id = 0
        for m in markerArray.markers:
            m.id = id
            id += 1
        if sample:
            self.sample_viz_pub.publish(markerArray)
        else:
            self.plan_viz_pub.publish(markerArray)
        return
    
    def create_plan_marker(self, plan):
        """ Given the position of a sample, publishes the necessary Marker data to the 'wp_viz' topic for RViZ visualization """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.action = 0 # add the marker
        marker.type = 5 # line_list

        for p in range(len(plan)-1):
            pt1 = Point()
            pt1_global = np.dot(self.transformation_matrix, [plan[p][0], plan[p][1], 1])
            pt1.x = pt1_global[0]
            pt1.y = pt1_global[1]
            marker.points.append(pt1)

            pt2 = Point()
            pt2_global = np.dot(self.transformation_matrix, [plan[p+1][0], plan[p+1][1], 1])
            pt2.x = pt2_global[0]
            pt2.y = pt2_global[1]
            #pt2.x = plan[p+1][0]
            #pt2.y = plan[p+1][1]
            marker.points.append(pt2)

        #
        #marker.pose.position.x = pos[0]
        #marker.pose.position.y = pos[1]
        #marker.pose.position.z = 0
        #marker.pose.orientation.x = 0.0
        #marker.pose.orientation.y = 0.0
        #marker.pose.orientation.z = 0.0
        #marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        #marker.scale.y = 0.1
        #marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0    
        return marker


def main():
    rospy.init_node('rrt_mp_node')
    mp = MotionPlanner()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Termination requested by user...")

if __name__ == '__main__':
    print("RRT Motion Planner Initialized...")
    main()