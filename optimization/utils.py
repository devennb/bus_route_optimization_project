
import pandas as pd
import numpy as np 
import osmnx as ox
from itertools import combinations

##### basic util functions for analysis, etc. #####

def get_osmnx_graph(): 
    '''
    Loads the saved osmnx graphml basemap
    '''

    return ox.load_graphml('inputs/nyc.graphml') 

def get_demand_parameters():
    '''
    Loads the individual demands per node, saved as a csv file
    '''

    return pd.read_csv('inputs/demands_w_airport_fac.csv', index_col=[0])

def get_cost_matrix(weight_type):
    '''
    Loads the cost matrices (distances/times), stored as a csv file
    Params: 
        **weight_type: 'distance' or 'time'
    '''

    if weight_type == 'distance':
        return pd.read_csv('inputs/distances_w_airport_fac.csv', index_col=[0])
    if weight_type == 'time':
        return pd.read_csv('inputs/times_w_airport_fac.csv', index_col=[0])
    else:
        raise Exception("Unacceptable Weight Type Input")

def search_nearby_nodes(node_id, D, S):
    '''
    Given a matrix of OD-pair distances, retrieve a list of "covered" nodes provided a centroid node and a radius-based distance (in meters)
    Params: 
        ** node_id: the centroid node ID
        ** D: the distance matrix object (pandas df, numpy array)
        ** S: coverage/radius value (numeric) wrto `node_id`
    '''

    s=(D.loc[int(node_id)] < S)
    if len(s) < 1:
        return []
    return s[s==True].index

def subsets_of_size_k(lst, k):
    '''
    Returns a list of all enumerated subsets of size k given a base-set `lst` (len(lst) > k)
    '''

    return list(combinations(lst, k))

def find_pairwise_cost(matrix, nodeA, nodeB):
    '''
    Searches for an OD-pair cost (given `nodeA` and `nodeB`) from some input cost matrix (`matrix`)
    '''

    return matrix.loc[int(nodeA), str(nodeB)]

def obtain_veh_route(G, origin_node, weight_type='travel_time', terminal_node=5969794486):
    '''
    Calculates an optimized vehicle route from an origin node to a destination, via osmnx `shortest_path`. 
    Params:
        **G: osmnx base graph 
        **origin_node: journey origin node id 
        **weight_type: journey optimized by `travel_time` or `distance`?
        **terminal_node: destination node (defaults to LGA Terminal C)
    '''

    return ox.shortest_path(
        G, 
        origin_node, 
        terminal_node, 
        weight=weight_type
    )