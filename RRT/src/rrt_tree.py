import operator
import numpy as np

class RRTTree(object):
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def add_vertex(self, pos, parent):
        """ Add a vertex to the tree """
        vid = len(self.vertices)
        self.vertices[vid] = Node(pos=pos, parent=parent)
        return vid

    def add_edge(self, start_id, end_id):#, edge_cost=0):
        """ Adds an edge to the tree """
        self.edges[end_id] = start_id
        #self.vertices[eid].set_cost(cost=self.vertices[sid].cost + edge_cost)

    def get_vertex_for_pos(self, pos):
        """ Search for the vertex with the given config and return it if exists """
        v_idx = self.get_idx_for_config(pos=pos)
        if v_idx is not None:
            return self.vertices[v_idx]
        return None

    def get_idx_for_pos(self, pos):
        """ Search for the vertex with the given config and return the index if exists """
        valid_idxs = [v_idx for v_idx, v in self.vertices.items() if (v.pos[0] == pos[0] and v.pos[1] == pos[1])]
        if len(valid_idxs) > 0:
            return valid_idxs[0]
        return None

    def get_nearest_pos(self, pos):
        """ Find the nearest vertex for the given config and returns its state index and configuration """
        # compute distances from all vertices
        dists = []
        for _, vertex in self.vertices.items():
            dists.append(self.compute_distance(pos, vertex.pos))

        # retrieve the id of the nearest vertex
        vid, _ = min(enumerate(dists), key=operator.itemgetter(1))

        return vid, self.vertices[vid].pos

    def compute_distance(self, pos1, pos2):
        return np.linalg.norm(np.subtract(pos1,pos2), 2)

class Node(object):
    def __init__(self, pos, parent=None):
        self.pos = pos # LiDAR frame
        self.parent = parent
        #self.cost = None # only used in RRT*