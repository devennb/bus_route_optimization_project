from pulp import * 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from utils import *
from build_parallel_route import ParallelRouteOptimation

class SequentialRouteOptimization(ParallelRouteOptimation):

    def __init__(
        self,
        alpha: float,
        num_stops: int, 
        coverage_radius: int, 
        geo_coverage_prop: float, 
        origin_node_id: int
    ):
        super().__init__(
            alpha = alpha, 
            num_stops = num_stops - 1, 
            coverage_radius = coverage_radius, 
            geo_coverage_prop = geo_coverage_prop
        )

        self.origin_node_id = origin_node_id
        self.model = LpProblem("Sequential Routing Model", LpMaximize)

    def solve_problem(self):

        #define X_i, which determines whether a node is covered...
        node_list = self.node_demands.index
        vars_covered_id = [f'{i}' for i in node_list]
        self.vars_covered = LpVariable.dicts(
            name='X',
            indices=vars_covered_id,
            cat=LpInteger, 
            lowBound=0, 
            upBound=1, 
        ) 
        vars_covered_arr = np.array(list(self.vars_covered.values()))

        #define Y_ij, determining route nodes to maximize coverage
        vars_fac_id = [f'{i}_{j}' for i in node_list for j in node_list]
        self.vars_fac = LpVariable.dicts(
            name='Y',
            indices=vars_fac_id,
            cat=LpInteger, ###change if needed
            lowBound=0, 
            upBound=1
        ) 
        vars_fac_arr = np.array(list(self.vars_fac.values()))

        #subtour elim proxy vars
        tour_order_vars = LpVariable.dicts(
            name='T', indices=list(node_list), cat=LpContinuous, lowBound=0
        )

        coverage_maximization = lpSum(self.alpha*(self.node_demands*vars_covered_arr))

        self.link_costs = np.array(list(map(lambda i: find_pairwise_cost(self.link_travel_times, *i.split('_')), self.vars_fac.keys())))
        cost_minimization = lpSum((1-self.alpha)*(self.link_costs*vars_fac_arr))
    
        self.model += (coverage_maximization - cost_minimization)

        #enforce number of stations
        self.model += (lpSum(self.vars_fac.values()) == self.ttl_stops)

        #balance w fair geographical coverage...
        self.model += ((lpSum(self.vars_covered.values())/self.ttl_nodes) >= self.geo_coverage_prop)

        #subtour/cycle elim. --- only consider "small paths" for runtime 
        M = 10000
        for i in node_list: 
            if i == self.origin_node_id:
                continue
            for j in node_list:
                if i != j: 
                    self.model += ((tour_order_vars[i] - tour_order_vars[j] + M*self.vars_fac[f'{i}_{j}']) <= M - 1)
                else: 
                    self.model += (self.vars_fac[f'{i}_{j}'] == 0)
        
        for i in node_list:
            s1 = sum([y_ij for idx, y_ij in self.vars_fac.items() if idx.split('_')[0] == str(i) and idx.split('_')[1] != str(i)])
            s2 = sum([y_ji for idx, y_ji in self.vars_fac.items() if idx.split('_')[1] == str(i) and idx.split('_')[0] != str(i)])
        
            if i == self.origin_node_id: 
                self.model += (s1 == (1 + s2))
            elif i == self.terminal_node_id: 
                self.model += (s1 == (-1 + s2))
            else:
                self.model += (s1 == s2)

            #ensure the MST is a linked list (no branches!!!)
            self.model += (s1 <= 1)

        #coverage 
        for id, x_k in self.vars_covered.items():
            S_k = search_nearby_nodes(id, self.link_distances, self.coverage_radius) 
            coverage_k = 0
            for j in S_k: 
            
                #only consider upstream nodes that are within the scope of the route we're constructing (from origin -> dest)
                N_j = [i for i in node_list 
                       if i != j 
                       and find_pairwise_cost(self.link_distances, i, self.terminal_node_id) <= find_pairwise_cost(self.link_distances, self.origin_node_id, self.terminal_node_id)
                ] 
                for i in N_j:
                    y_ij = self.vars_fac[f'{i}_{j}']
                    coverage_k += y_ij
            self.model += (x_k <= coverage_k) 

        self.model.solve()

    def build_route_parameters(self):

        objective_val = value(self.model.objective)
        coverage_optimized_route = [
            link for link, v in self.vars_fac.items() if v.value() > 0
        ]
        maximized_demand_coverage =  (np.array([i.value() for i in self.vars_covered.values()])*self.node_demands).sum()
        vars_fac_ret = self.vars_fac.copy()
        for idx,v in enumerate(self.vars_fac.items()): 
            i,j = v
            if j.value() == 0: 
                del vars_fac_ret[i]
            else:
                vars_fac_ret[i] = j.value() * self.link_costs[idx]

        vars_fac_df = pd.DataFrame(vars_fac_ret, index=['cost']).T.reset_index()
        vars_fac_df['index'] = vars_fac_df['index'].apply(lambda x: x.split('_')[0])
        vars_fac_df=vars_fac_df.sort_values(by=['index'], ascending=False)
        vars_fac_df['cost'] = vars_fac_df['cost'].cumsum() / 60

        self.results = {
            'Objective Value': objective_val, 
            'Coverage Optimized Route': coverage_optimized_route, 
            'Maximized Demand Coverage': maximized_demand_coverage, 
            'Minimized Costs/Times': vars_fac_df
        }

        print('*** SUMMARY STATISTICS ***')
    
        demand = self.results['Maximized Demand Coverage'] / self.node_demands.sum()
        print(f'Demand Covered: {demand}')

        print(f"Total System Time: {vars_fac_df['cost'].max()}")
        print(f"Average User System Time: {vars_fac_df['cost'].mean()}")
        print(f"Upper Bound User Time in System: {vars_fac_df['cost'].max()} min")
        print(f"Lower Bound User Time in System: {vars_fac_df['cost'].min()} min")

        return self.results
    
    def plot_route(self): 
        G = get_osmnx_graph()
        routes = [
            obtain_veh_route(
                G,
                origin_node=int(i.split('_')[0]), 
                terminal_node=int(i.split('_')[1])
            ) for i in self.results['Coverage Optimized Route']
        ]

        center = (40.75047690227571, -73.94050480138289)
        radius = 7000
        bbox = ox.utils_geo.bbox_from_point(center, dist=radius)

        _, ax = ox.plot_graph_routes(
            G, 
            show=False, 
            close=False, 
            routes=routes, 
            figsize=(15,15), 
            bbox=bbox, 
            node_alpha=0.1,
            edge_alpha=1.0, 
            edge_linewidth=2.0
        )

        assigned_nodes = list(self.results['Minimized Costs/Times']['index'].astype(int))
        nodes, _ = ox.graph_to_gdfs(G, nodes=True)
        print(assigned_nodes)
        assigned_nodes_df = nodes.loc[assigned_nodes]
        coverage_radius = self.coverage_radius / 100000
        for _, r in assigned_nodes_df.iterrows():
            ax.add_patch(
                plt.Circle((r['x'], r['y']), 
                    radius = coverage_radius, 
                    color='red', 
                    alpha=0.2
            )
        )     
        plt.show()
    
    



        
        