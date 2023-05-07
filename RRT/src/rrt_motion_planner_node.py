#!/usr/bin/env python
import rospy, time
import numpy as np

from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float64MultiArray, MultiArrayDimension #, MultiArrayLayout
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from rrt_tree import RRTTree
from rrt_auxiliary import *

# RRT parameters
GOAL_BIAS = rospy.get_param("/rrt_motion_planner_node/GOAL_BIAS")
EXT_MODE = 'E1'
# Choose map
LEVINE = rospy.get_param("/rrt_motion_planner_node/LEVINE")
SPIELBERG = rospy.get_param("/rrt_motion_planner_node/SPIELBERG")

class MotionPlanner(object):
    def __init__(self):
        ## Occupancy grid stuff
        self.occ_grid, self.grid_resolution, self.grid_width, self.grid_length = None, None, None, None

        ## Global waypoint stuff
        if LEVINE:
            opt_waypoints = [i[1:3] for i in set_optimal_waypoints(file_name='levine_raceline')] # in the global frame
        elif SPIELBERG:
            opt_waypoints = [i[:2] for i in set_optimal_waypoints(file_name='Spielberg_raceline')] # in the global frame
        self.opt_waypoints = [opt_waypoints[j] for j in range(len(opt_waypoints)) if j % 3 == 0]
        self.raceline_markers = create_raceline_marker_array(self.opt_waypoints)

        ## RRT Stuff
        self.goal_bias = GOAL_BIAS
        #self.goal_pos = None
        
        ## Data Topics
        wp_topic = rospy.get_param("/rrt_motion_planner_node/wp_topic") # sending waypoints to this topic
        occ_grid_topic = rospy.get_param("/rrt_motion_planner_node/occ_grid_topic") # reading the occupancy grid from this topic
        odom_topic = rospy.get_param("/rrt_motion_planner_node/odom_topic")

        self.wp_pub = rospy.Publisher(wp_topic, Float64MultiArray, queue_size=10)
        self.occ_grid_sub = rospy.Subscriber(occ_grid_topic, OccupancyGrid, self.occ_grid_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=10)
        
        ## Visualization Topics
        #global_wp_viz_topic = rospy.get_param("/rrt_motion_planner_node/global_wp_viz_topic") # sending waypoint visualization data to this topic
        raceline_viz_topic = rospy.get_param("/rrt_motion_planner_node/raceline_viz_topic") # sending raceline visualization data to this topic
        sample_viz_topic = rospy.get_param("/rrt_motion_planner_node/sample_viz_topic") # sending sample visualization data to this topic
        plan_viz_topic = rospy.get_param("/rrt_motion_planner_node/plan_viz_topic") # sending plan visualization data to this topic

        #self.global_wp_viz_pub = rospy.Publisher(global_wp_viz_topic, MarkerArray, queue_size=10)
        self.raceline_pub = rospy.Publisher(raceline_viz_topic, MarkerArray, queue_size = 10)
        self.sample_viz_pub = rospy.Publisher(sample_viz_topic, MarkerArray, queue_size = 10)
        self.plan_viz_pub = rospy.Publisher(plan_viz_topic, Marker, queue_size = 10)

    def occ_grid_callback(self, grid_msg):
        self.grid_resolution, self.grid_width, self.grid_length = grid_msg.info.resolution, grid_msg.info.width, grid_msg.info.height
        self.occ_grid = np.reshape(grid_msg.data, (self.grid_width, self.grid_length))
        
        #tic = time.time()

        self.tree = RRTTree()
        self.tree.add_vertex([0,0], parent=None)
        new_waypoints = self.plan()

        #toc = time.time() - tic
        #print(toc)
        #print(len(new_waypoints))
        
        if new_waypoints is not None:
            wp_msg = self.create_waypoint_message(new_waypoints)
            self.wp_pub.publish(wp_msg)

            plan_marker = self.create_plan_marker(new_waypoints)
            self.plan_viz_pub.publish(plan_marker)

    def odom_callback(self, odom_msg):
        """ Records the car's current position and orientation relative to the global frame, given odometry data """
        self.current_position = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y] 
        self.current_orientation = [odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
        self.transformation_matrix, _ = calc_transf_matrix(self.current_position, self.current_orientation)
        self.raceline_pub.publish(self.raceline_markers)
    
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
    
    def get_goal_point(self):
        nearest_wp_idx = (find_nearest_waypoint(self.current_position, self.opt_waypoints) + 1) % len(self.opt_waypoints)
        best_wp_global = self.opt_waypoints[nearest_wp_idx]
        #best_wp_lidar = np.dot(np.linalg.inv(self.transformation_matrix), [best_wp_global[0], best_wp_global[1], 1])[:2]
        best_wp_lidar = global_to_laser(self.transformation_matrix, best_wp_global)[:2]
        #if True:
        #    return best_wp_lidar
        counter = 1
        while counter < len(self.opt_waypoints):
            new_idx = (nearest_wp_idx + counter) % len(self.opt_waypoints)
            new_wp_global = self.opt_waypoints[new_idx]
            #new_wp_lidar = np.dot(np.linalg.inv(self.transformation_matrix), [new_wp_global[0], new_wp_global[1], 1])[:2]
            new_wp_lidar = global_to_laser(self.transformation_matrix, new_wp_global)[:2]
            if not in_OccGrid(new_wp_lidar, self.grid_resolution, self.grid_width, self.grid_length):
                return best_wp_lidar 
            
            waypoint_grid_indices = laser_to_grid_idx(new_wp_lidar, self.grid_resolution, self.grid_length)
            if self.occ_grid[waypoint_grid_indices[0]][waypoint_grid_indices[1]] == int(100):
                counter += 1
                continue
                
            best_idx = (nearest_wp_idx + counter) % len(self.opt_waypoints)
            best_wp_global = self.opt_waypoints[best_idx]
            #best_wp_lidar = np.dot(np.linalg.inv(self.transformation_matrix), [best_wp_global[0], best_wp_global[1], 1])[:2]
            best_wp_lidar = global_to_laser(self.transformation_matrix, best_wp_global)[:2]
            counter = counter + 1
        return None
    
    def collision_free_pos(self, pos):
        grid_point = laser_to_grid_idx(pos, self.grid_resolution, self.grid_length)
        if self.occ_grid[grid_point[0], grid_point[1]] == int(0):
            return True
        else:
            False

    def plan(self):
        sample_array = MarkerArray()
        goal_pos = self.get_goal_point()
        if goal_pos is None:
            print("Hey, I'm the problem - it's me!")
            return None
        
        # First, we see if the goal point is directly reachable from the root node (i.e. the local geometry is simple enough)
        pos = goal_pos
        nearest_vert = [0, self.tree.vertices[0].pos]
        if self.collision_check(pos, nearest_vert[1]):
            pos_idx = self.tree.add_vertex(pos, nearest_vert)
            self.tree.add_edge(nearest_vert[0], pos_idx)#, cost)
            goal_added = True
        else:
            goal_added = False
        
        num_iter = 0
        while not goal_added:
            if num_iter > 1000:
                return [self.tree.vertices[0].pos, goal_pos]

            num_iter += 1
            goal = False

            # Sampling step
            p = np.random.uniform() # goal biasing
            if p < self.goal_bias:
                pos = goal_pos
                goal = True
            else:
                if EXT_MODE == 'E1':
                    pos = self.sample()
                else:
                    pos = self.sample(free_only = False)

            # Verify that the sample is in free space
            #if EXT_MODE != 'E1' and not self.collision_free_pos(pos) and :
            #    continue

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
            #if num_iter < 1000:# or goal:
            sample_global_frame = laser_to_global(self.transformation_matrix, pos)
            sample_marker = create_point_marker(sample_global_frame[:2], goal)
            sample_array.markers.append(sample_marker)
            self.create_waypoint_marker_array(sample_array)
            
            #if num_iter > 100:
            #    goal_added = True

            # Partial extensions, if enabled
            #if self.ext_mode == 'E2':
            #    pos = self.extend(nearest_vert[1], pos) # config = x_new
                #if not env.config_validity_checker(config):
                #    continue
            
            # Check obstacle-collision for potential edge
            #step = self.grid_resolution
            #if check_edge_collision(pos, nearest_vert[1], step, self.occ_grid, self.grid_resolution, self.grid_width, self.grid_length, rrt=True):
            if self.collision_check(pos, nearest_vert[1]):
                pos_idx = self.tree.add_vertex(pos, nearest_vert)
                #cost = self.tree.compute_distance(config, nearest_vert[1])
                self.tree.add_edge(nearest_vert_idx, pos_idx)#, cost)
                if goal: # and self.ext_mode == 'E1':
                    goal_added = True
            else:
                goal_added = False
        #self.create_waypoint_marker_array(sample_array)

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
    
    def collision_check(self, sample_pos, nearest_neighbor_pos):
        pos_grid_idx = laser_to_grid_idx(sample_pos, self.grid_resolution, self.grid_length)
        nearest_grid_idx = laser_to_grid_idx(nearest_neighbor_pos, self.grid_resolution, self.grid_length)
        free_cells = traverse_grid(nearest_grid_idx, pos_grid_idx)
        for cell in free_cells:
            if (cell[0] * cell[1] < 0) or (cell[0] >= self.grid_width) or (cell[1] >= self.grid_length):
                continue
            if self.occ_grid[cell[0]][cell[1]] == int(100):
                return False
        return True
    
    def sample(self, free_only = True):
        """ This method should randomly sample the free space, and returns a viable point """
        if free_only:
            i, j = np.random.randint(self.grid_width), np.random.randint(self.grid_length)
            while self.occ_grid[i][j] != int(0):
                i, j = np.random.randint(self.grid_width), np.random.randint(self.grid_length)
            
            upper_bound_x = (j + 1)*self.grid_resolution
            lower_bound_x = (j)*self.grid_resolution
            upper_bound_y = (i + 1 - self.grid_length/2)*self.grid_resolution
            lower_bound_y = (i - self.grid_length/2)*self.grid_resolution 

            x = np.random.uniform(low=lower_bound_x, high=upper_bound_x)
            y = np.random.uniform(low=lower_bound_y, high=upper_bound_y)
        else:
            x = np.random.uniform(low=0, high=self.grid_resolution*self.grid_width)
            y = np.random.uniform(low=-self.grid_resolution*self.grid_length/2, high=self.grid_resolution*self.grid_length/2)
        return [x,y]

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
        """ Given a plan, creates the necessary Marker data for RViZ visualization """
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = "map", rospy.Time.now()
        marker.id, marker.action, marker.type = 0, 0, 5 # ID, adds the marker, line_list

        for p in range(len(plan)-1):
            pt1 = Point()
            pt1_global = laser_to_global(self.transformation_matrix, plan[p])
            #pt1_global = np.dot(self.transformation_matrix, [plan[p][0], plan[p][1], 1])
            pt1.x, pt1.y = pt1_global[0], pt1_global[1]
            marker.points.append(pt1)

            pt2 = Point()
            pt2_global = laser_to_global(self.transformation_matrix, plan[p+1])
            #pt2_global = np.dot(self.transformation_matrix, [plan[p+1][0], plan[p+1][1], 1])
            pt2.x, pt2.y = pt2_global[0], pt2_global[1]
            marker.points.append(pt2)

        marker.scale.x, marker.color.a = 0.1, 1.0
        marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0 
        marker.pose.orientation.x, marker.pose.orientation.y = 0.0, 0.0
        marker.pose.orientation.z, marker.pose.orientation.w = 0.0, 1.0
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