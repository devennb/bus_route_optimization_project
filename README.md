### Multi Objective Bus System Design and Optimization 

This repo contains all of the code/analytics associated with my term project (TR-GY 7013 Urban Transport/Logistics Systems at NYU Tandon). All deliverables including the final report (`Report.pdf`) are attached here. 

These workflows serve as a rough proof-of-concept for leveraging mathematical optimization techniques to propose bus routes, using dual-objective mixed-integer programs to maximize route-coverage/demand while minimizing cost/time in-system. Two route designs are proposed: a parallel model ("drop-off/pick-up" design) and a sequential model (more akin to a traditional bus route). The analysis here exclusively focuses on a bus system linking Manhattan (below 60th st) to LaGuardia Airport. However, in the future, this workflow could be reproducible for any set of planning constraints (any "proposed" route in NYC, beyond), conditional on the availability of ridership demand data. 

Core items include: 
* `data_playground`: directory containing analyses/scripts on GTFS bus-feed data (existing services, specifically the `Q70+` shuttle). `load_bus_location.py` processes the GTFS feed (via the MTA's BusTime API) into a usable format for analysis, recording performance-related attributes into a SQLite database. `benchmark_analysis.ipynb` is a jupyter notebook that visualizes the GTFS feed data, specifically looking at wait-times/delays as well as estimating passenger ridership and bus utilization. 
* `optimization`: all scripts/notebooks associated with building the alternative routes. Two scripts (`build_parallel_route.py` and `build_sequential_route.py`) provide functionality related to route construction, optimization, and visualization. `alternative_analysis.ipynb` invokes the optimization and performs an analysis on performance across the two route designs/models. `optimization_playground.ipynb` is a scratchpad notebook containing any exploratory data analysis prior to optimization. 
* `optimization/inputs`: all flatfiles containing the fixed parameters required to run the models. Specific to this workflow only. These files were generated via `optimization_playground.ipynb`. 
* `Report.pdf`: the final report/paper discussing the methodology, decisions around the work illustrated in this repo

Future work may include: 
* Building a discrete event simulation on the existing passenger journey from Manhattan to LGA Airport
* Incorporating a predictive model to estimate and forecast node-based demands and/or congestion effects (not currently taken into account)
* Building a wrapper around the current work (as a Python package, etc.), generalizing functionality for any route-construction scenario

