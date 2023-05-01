#!/usr/bin/env python
import rospy, os, csv
import numpy as np

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from visualization_msgs.msg import Marker, MarkerArray

class MotionPlanner(object):
    def __init__(self):
        self.occ_grid = None
        self.opt_waypoints = [i[1:3] for i in self.set_optimal_waypoints(file_name='levine_raceline')] # in the global frame
        self.opt_waypoints = [self.opt_waypoints[j] for j in range(len(self.opt_waypoints)) if j % 5 == 0]
        #print(self.opt_waypoints)

        wp_topic = '/rrt/waypoints'
        wp_viz_topic = '/rrt/wp_viz' # sending point visualization data
        occ_grid_topic = '/rrt/occ_grid'
        
        self.wp_pub = rospy.Publisher(wp_topic, Float64MultiArray, queue_size=10)
        self.wp_viz_pub = rospy.Publisher(wp_viz_topic, MarkerArray, queue_size=10)
        self.occ_grid_sub = rospy.Subscriber(occ_grid_topic, OccupancyGrid, self.occ_grid_callback, queue_size=10)

    def occ_grid_callback(self, grid_msg):
        self.occ_grid = grid_msg
        new_waypoints = self.plan() #[i[1:3] for i in self.plan()]
        wp_msg = Float64MultiArray()
        #wp_msg.data = np.reshape(new_waypoints, (1,-1))[0]
        #print(new_waypoints[:][1:3])
        #print(np.reshape(new_waypoints, (1,-1))[0])
        wp_msg.data = np.reshape(new_waypoints, (1,-1))[0]
        #print(list(np.shape(new_waypoints)))
        dim0 = MultiArrayDimension()
        #print(np.shape(new_waypoints)[1])
        dim0.size = np.shape(new_waypoints)[0]
        dim1 = MultiArrayDimension()
        dim1.size = np.shape(new_waypoints)[1]

        #layout = MultiArrayLayout()
        #layout.dim = list()
        wp_msg.layout.dim = [dim0, dim1]
        #wp_msg.layout.dim[1].size = np.shape(new_waypoints)[1]
        self.wp_pub.publish(wp_msg)

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

    def plan(self):
        markerArray = MarkerArray()
        for i in range(len(self.opt_waypoints)):
            wp_pos = self.opt_waypoints[i][:2]
            sample_marker = self.create_sample_marker(wp_pos)
            markerArray.markers.append(sample_marker)
        self.create_waypoint_marker_array(markerArray)
        return self.opt_waypoints # temporary
    
    def create_sample_marker(self, pos, goal=False):
        # Given the position of a sample, publishes the necessary Marker data to the 'wp_viz' topic for RViZ visualization
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = 2 # sphere
        marker.action = 0 # add the marker
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
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
        return marker

    def create_waypoint_marker_array(self, markerArray):
        id = 0
        for m in markerArray.markers:
            m.id = id
            id += 1
        self.wp_viz_pub.publish(markerArray)
        return

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