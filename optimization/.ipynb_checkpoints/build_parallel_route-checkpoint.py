
from pulp import * 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from utils import *

class ParallelRouteOptimation: 

    def __init__(
        self,
        alpha: float,
        num_stops: int, 
        coverage_radius: int, 
        geo_coverage_prop: float
    ):
        
        self.model = LpProblem("Parallel Routing Model", LpMaximize)
        self.terminal_node_id = 5969794486
        self.ttl_stops = num_stops + 1
        self.node_demands = get_demand_parameters()['d']
        self.ttl_nodes = len(self.node_demands)
        self.alpha = alpha 
        self.coverage_radius = coverage_radius
        self.geo_coverage_prop = geo_coverage_prop
        self.link_distances = get_cost_matrix(weight_type='distance')
        self.link_travel_times = get_cost_matrix(weight_type='time')

    def solve_problem(self):

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
        print(self.link_travel_times[str(self.terminal_node_id)])
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
        assigned_stops = [
            int(i.replace('Y_', '')) for i,k in self.vars_fac.items() if k.value() == 1
        ]

        maximized_demand_coverage =  (np.array([i.value() for i in self.vars_covered.values()])*self.node_demands).sum() 
        objective_val = value(self.model.objective)
        for idx,v in enumerate(self.vars_fac.items()): 
            i,j = v
            self.vars_fac[i] = j.value() * self.dest_costs[idx]
        
        self.results = {
            'Stops': assigned_stops, 
            'Objective': objective_val, 
            'Costs': self.vars_fac, 
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
            
        return self.results
    
    def plot_route(self):
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
        pass
