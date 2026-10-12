import os
import sys


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci


Sumo_config = [
    'sumo-gui',
    '-c', 'user.sumocfg',
    '--lateral-resolution', '0.5'
]


traci.start(Sumo_config)


vehicle_waiting = {}
total_waiting_time = 0

throughput = 0
step = 0

queue_per_step = []


junction_id = "J1"
incoming_edges = traci.junction.getIncomingEdges(junction_id)


while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    
    for v in vehicles:
        vehicle_waiting[v] = traci.vehicle.getAccumulatedWaitingTime(v)

    
    total_edge_queue = 0

    for edge in incoming_edges:
        lane_count = traci.edge.getLaneNumber(edge)

        for i in range(lane_count):
            lane_id = f"{edge}_{i}"
            total_edge_queue += traci.lane.getLastStepHaltingNumber(lane_id)

    # Store only when vehicles exist (avoid zero bias)
    if len(vehicles) > 0:
        queue_per_step.append(total_edge_queue)

    # -------- Throughput + Waiting Time --------
    arrived = traci.simulation.getArrivedIDList()

    for v in arrived:
        if v in vehicle_waiting:
            total_waiting_time += vehicle_waiting[v]

    throughput += len(arrived)

    step += 1


traci.close()



vehicle_count = throughput


avg_waiting_time = total_waiting_time / vehicle_count if vehicle_count > 0 else 0


avg_queue_length = sum(queue_per_step) / len(queue_per_step) if queue_per_step else 0
max_queue_length = max(queue_per_step) if queue_per_step else 0


step_length = 1  
throughput_rate = (throughput / (step * step_length)) * 60 if step > 0 else 0

# Results 
print("\n--- RESULTS ---")
print("Total Vehicles:", vehicle_count)
print("Average Waiting Time (s):", round(avg_waiting_time, 2))
print("Average Queue Length:", round(avg_queue_length, 2))
print("Max Queue Length:", max_queue_length)
print("Throughput Rate (veh/min):", round(throughput_rate, 2))