
from pulp import * 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from utils import *

class ParallelRouteOptimation: 
    '''
    Parallel Routing Optimization Class 

    Args: 
        ** alpha: weighting coefficient for objective prioritization (demand coverage vs cost minimization in-system). Must be a value between 0 to 1
        ** num_stops: number of pick-up locations enforced in the system design 
        ** coverage_radius: max coverage distance (meters) for each prospective pick-up location (demand node)
        ** geo_coverage_prop: min required proportion of covered nodes across the system (irresepctive of demands). Balances user maximization with geographical equity
    '''

    def __init__(
        self,
        alpha: float,
        num_stops: int, 
        coverage_radius: int, 
        geo_coverage_prop: float
    ):
        
        #define model 
        self.model = LpProblem("Parallel Routing Model", LpMaximize)

        #set terminal node to LGA Terminal C
        self.terminal_node_id = 5969794486

        #include the terminal node as part of the system design (not just the pick-up locations)
        self.ttl_stops = num_stops + 1

        #retrieve demands per node
        self.node_demands = get_demand_parameters()['d']
        self.ttl_nodes = len(self.node_demands)

        self.alpha = alpha 
        self.coverage_radius = coverage_radius
        self.geo_coverage_prop = geo_coverage_prop

        #retrieve costs
        self.link_distances = get_cost_matrix(weight_type='distance')
        self.link_travel_times = get_cost_matrix(weight_type='time')

    def solve_problem(self):
        '''
        Instantiates the optimization, constructs the formulation, and solves it
        '''

        #define X_i, which determines whether a node is covered...
        vars_covered_id = [f'X_{i}' for i in self.node_demands.index]
        self.vars_covered = LpVariable.dicts(
            name='assignments',
            indices=vars_covered_id,
            cat=LpInteger, 
            lowBound=0, 
            upBound=1, 
        ) 
        vars_covered_arr = np.array(list(self.vars_covered.values()))

        #define Y_i, determining facility assignment
        vars_fac_id = [f'Y_{i}' for i in self.node_demands.index]
        self.vars_fac = LpVariable.dicts(
            name='placements',
            indices=vars_fac_id,
            cat=LpInteger, 
            lowBound=0, 
            upBound=1, 
        ) 
        vars_fac_arr = np.array(list(self.vars_fac.values()))

        coverage_maximization = lpSum(self.alpha*(self.node_demands*vars_covered_arr))
        self.dest_costs = self.link_travel_times[str(self.terminal_node_id)].to_numpy()

        cost_minimization = lpSum((1-self.alpha)*(self.dest_costs*vars_fac_arr))
        self.model += (coverage_maximization - cost_minimization)

        #enforce number of stations
        self.model += (lpSum(self.vars_fac.values()) == self.ttl_stops)

        #balance w fair geographical coverage...
        self.model += ((lpSum(self.vars_covered.values())/self.ttl_nodes) >= self.geo_coverage_prop)

        #ensure a node is covered by at least one stop...
        for id, x_i in self.vars_covered.items():
            node_id = id.replace('X_', '')
            N_i = search_nearby_nodes(node_id, self.link_distances, self.coverage_radius) 
            coverage_i = 0
            for y_j in N_i: 
                node_id_j = f'Y_{y_j}'
                coverage_i += self.vars_fac[node_id_j]

            self.model += (x_i <= coverage_i)

        self.model.solve()
        
    def build_route_parameters(self):
        '''
        Calculates and prints (system-wide) summary statistics from the optimization results, including but not limited to: 
            ** averaged/min/max user system time
            ** demand covered (number of prospective riders) under the optimized route scheme
        '''

        assigned_stops = [
            int(i.replace('Y_', '')) for i,k in self.vars_fac.items() if k.value() == 1
        ]

        maximized_demand_coverage =  (np.array([i.value() for i in self.vars_covered.values()])*self.node_demands).sum() 
        objective_val = value(self.model.objective)
        costs_report = self.vars_fac.copy()
        for idx,v in enumerate(self.vars_fac.items()): 
            i,j = v
            costs_report[i] = j.value() * self.dest_costs[idx]
        
        self.results = {
            'Stops': assigned_stops, 
            'Objective': objective_val, 
            'Costs': costs_report, 
            'Total Covered Demand': maximized_demand_coverage
        }

        print('*** SUMMARY STATISTICS ***')
        costs = np.array(list(self.results['Costs'].values()))
        route_times = costs[costs > 0]
        demand = self.results['Total Covered Demand'] / self.node_demands.sum()

        print(f'Demand Covered: {demand}')
        print(f'Total System Time: {route_times.sum() / 60} min')
        print(f'Average User System Time: {route_times.mean() / 60} min')
        print(f'Upper Bound User Time in System: {route_times.max() / 60} min')
        print(f'Lower Bound User Time in System: {route_times.min() / 60} min')

        self.results['Summary Statistics'] = {
            'Demand Covered': demand,
            'Total System Time': route_times.sum() / 60,
            'Average User System Time': route_times.mean() / 60,
            'Upper Bound User Time in System': route_times.max() / 60,
            'Lower Bound User Time in System': route_times.min() / 60
        }
            
        return self.results
    
    def plot_route(self):
        '''
        Plot the route scheme via osmnx functionality after optimization
        '''

        G = get_osmnx_graph()
        assigned_nodes = self.results['Stops'][:-1]
        routes = [obtain_veh_route(G,i) for i in assigned_nodes]
        colors =  [np.random.rand(3) for _ in range(len(assigned_nodes))]

        center = (40.75047690227571, -73.94050480138289)
        radius = 7000
        bbox = ox.utils_geo.bbox_from_point(center, dist=radius)

        _ , ax = ox.plot_graph_routes(
            G, 
            show=False, 
            close=False, 
            routes=routes, 
            route_colors=colors,
            figsize=(15,15), 
            bbox=bbox, 
            node_alpha=0.1,
            edge_alpha=1.0, 
            edge_linewidth=2.0
        )
        nodes, _ = ox.graph_to_gdfs(G, nodes=True)
        assigned_nodes_df = nodes.loc[assigned_nodes]
        assigned_nodes_df['c_id'] = colors
        assigned_nodes_df['route'] = routes
        assigned_nodes_df.plot(
            ax=ax, 
            c=assigned_nodes_df["c_id"], 
            markersize=100
        )

        coverage_radius = self.coverage_radius / 100000
        for _, r in assigned_nodes_df.iterrows():
            ax.add_patch(
                plt.Circle((r['x'], r['y']), 
                    radius = coverage_radius, 
                    color=r['c_id'], 
                    alpha=0.2
            )
        )     
        plt.show()
