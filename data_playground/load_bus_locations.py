import numpy as np
import requests 
import pandas as pd
import sqlite3
import schedule
import os 
import time

mta_bustime_api_key = os.environ['BUSTIME_KEY']
get_movements = f'https://bustime.mta.info/api/siri/vehicle-monitoring.json?key={mta_bustime_api_key}'

def load_bus_locations(line, tbl_name, db_name):
    try:
        movements = requests.get(
            url = get_movements, 
            params = {
                'LineRef': line
            }
        ).json()
    except Exception as e:
        print(e)
        return 
    
    #find metadata in the response body
    veh_monitor_delivery = movements['Siri']['ServiceDelivery']['VehicleMonitoringDelivery'][0]

    #initialize list of cleaned responses
    data=[]

    ##### iterate over each "active" bus on the line...
    for bus in veh_monitor_delivery['VehicleActivity']:
        try:
            journey_metadata = bus['MonitoredVehicleJourney']
            record_time = bus['RecordedAtTime']
            status = journey_metadata['ProgressRate']
        except: 
            print('Invalid Bus Ping. Continuing...')
            continue

        #status should either be in-progress (normalProgress) or blocked (noProgress)
        if status in ['normalProgress','noProgress']:

            #get vehicle identifiers, block id (work shifts?), lat/lon, stop distances, misc ride features in the 'extension' key (including scheduled/ground-truth arrivals/departures)
            if 'MonitoredCall' not in journey_metadata.keys():
                print('No monitoring info. Continuing search')
                continue

            monitored_call = journey_metadata['MonitoredCall'].copy() 
            veh_location = journey_metadata['VehicleLocation']
            veh_id = journey_metadata['VehicleRef'] if 'VehicleRef' in journey_metadata.keys() else np.nan
            block_id = journey_metadata['BlockRef'] if 'BlockRef' in journey_metadata.keys() else np.nan
            ext = monitored_call['Extensions']
            distances = ext['Distances']
            veh_features =  ext['VehicleFeatures']

            #some responses have capacity info, others don't. if not, set to NaN's
            if 'Capacities' in ext.keys():
                capacities = ext['Capacities']
            else:
                capacities = {
                    'EstimatedPassengerCount': np.nan,
                    'EstimatedPassengerCapacity': np.nan
            }
            del monitored_call['Extensions']

            #append relevant data
            print(f'******* Recording Location Ping for Bus w/ Vehicle ID: {veh_id} [Status: {status}] *******')
            data.append({
                'Line': journey_metadata['LineRef'], 
                'RecordTime': record_time,
                'VehicleID': veh_id,
                'BlockID': block_id,
                'Status': status,
                'Destination': journey_metadata['DestinationName'], 
                **monitored_call,
                **veh_location,
                **capacities,
                **distances, 
            })
        else: 
            print(f'Unsupported status: {status}')

    #represent this as a Pandas dataframe    
    as_df = pd.DataFrame(data)

    ##### do some addtl feature engineering on the tbl, plus any necessary changes to data types (casting to datetime objects etc.)
    time_based_cols = as_df.columns[as_df.columns.str.contains('Time')]
    as_df[time_based_cols] = as_df[time_based_cols].apply(lambda x: pd.to_datetime(x))

    #calculate deltas btw expected/scheduled and projected arrivals
    #note to self, validate the definitions of "Aimed" and "Expected" in the documentation?
    #^expected is "predicted", aimed is the baseline (what's scheduled)
    as_df['ArrivalDelta'] = (as_df['AimedArrivalTime']-as_df['ExpectedArrivalTime']).dt.total_seconds() / 60
    as_df['DepartureDelta'] = (as_df['AimedDepartureTime']-as_df['ExpectedDepartureTime']).dt.total_seconds() / 60
    as_df['OverCapacity?'] = as_df['EstimatedPassengerCapacity'] < as_df['EstimatedPassengerCount']
    as_df['IsLate?'] = as_df['ArrivalDelta'] < 0 
    truncated = as_df[[
        'VehicleID', 
        'RecordTime',
        'Status',
        'ArrivalDelta', 
        'Longitude', 
        'Latitude', 
        'EstimatedPassengerCount', 
        'OverCapacity?', 
        'StopPointName', 
        'DistanceFromCall', 
        'CallDistanceAlongRoute'
    ]]

    #only parse out "active" buses only ("noProgress" buses are likely not taking passengers, perhaps bc of crew shifts, etc. --- might be interesting to look at further down the project)
    active_only = truncated.loc[truncated['Status'] == 'normalProgress']
    active_only['ttl_route_distance'] = active_only['CallDistanceAlongRoute'].max() 
    
    #add to sqlite db
    conn = sqlite3.connect(db_name,timeout=30)
    active_only.to_sql(
        tbl_name, conn, if_exists='append', index=False
    )
    conn.close()
    
    return active_only

if __name__ == '__main__':
    load_bus_locations(
        line = 'MTABC_Q70+',
        tbl_name = 'q70',
        db_name = 'bus_locations_benchmark2.db'
    )


    


    
