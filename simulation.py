import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
from tqdm import tqdm
from cars import Cars

def traffic_simulation(graph, n_cars, radius, speed_up, closeness_centrality, size = 100, scale = 0.3, dt = 0.05, n_frame = 300, animate = True):
    
    ''' speed_up = speed_up factor making simulation faster than reality
    
        size : refers to maximum value of animation image boundaries. Needs to be converted in meters using Bologna diameter

        scale : refers to arrows dimensions within the animation
       
        turn range : range with which the car sees a new node and turn its velocity. Take it around  max_speed*dt
       '''
    cars = Cars(n_cars)
    edges_passage = []
    edge_counts = []
    accident_positions = []
    travelling_times = []

    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    nodes['x'] = ((nodes['x'] - min(nodes['x']))/(max(nodes['x'])-min(nodes['x'])))*size
    nodes['y'] = ((nodes['y']-min(nodes['y']))/(max(nodes['y']) - min(nodes['y'])))*size
    
    cars.set_initial_condition(graph,size,radius,speed_up,dt,edges_passage,edge_counts,nodes,edges,closeness_centrality)
    print('\nInitialization done\n')

    n_cars = cars.get_n_cars()
    c_positions = np.zeros((n_frame, n_cars, 2))
    c_velocities = np.zeros((n_frame, n_cars, 2))
    c_accel = np.zeros((n_frame, n_cars, 2))

    #for loop computing time steps for animation
    for n in tqdm(range(n_frame)):
        c_positions[n] = cars.get_positions()
        c_velocities[n] = cars.get_velocities()
        c_accel[n] = cars.get_accel()
        
        cars.set_acceleration_zero()

        cars.update(graph,size,radius,speed_up,dt,n,nodes,edges,edges_passage,edge_counts,accident_positions,travelling_times,closeness_centrality)
        

    if animate:
        fig, ax = plt.subplots(figsize=(7,7))

        for N in range(len(graph.nodes)):
            circle = plt.Circle((nodes['x'][N] , nodes['y'][N]) , size/900, color = 'grey')
            ax.add_patch(circle)
        
        
        velocities_magnitudes = np.linalg.norm(c_velocities[0], axis=1)
        velocities_normalized = c_velocities[0] / np.vstack([velocities_magnitudes, velocities_magnitudes]).T 
        #velocities_normalized = c_velocities[0] * 0.5
        scat = ax.quiver(c_positions[0][:,0], 
                        c_positions[0][:,1],
                        velocities_normalized[:,0],
                        velocities_normalized[:,1], color = 'red')
        
        
        ax.set_xlim(0,size)
        ax.set_ylim(0,size)

        #update function useful for animation
        def update(frame):
            scat.set_offsets(c_positions[frame])

            velocities_magnitudes = np.linalg.norm(c_velocities[frame], axis=1)
            velocities_normalized = c_velocities[frame]/ np.vstack([velocities_magnitudes, velocities_magnitudes]).T 
            #velocities_normalized *= scale
            scat.set_UVC(velocities_normalized[:,0]*scale, 
                        velocities_normalized[:,1]*scale)

            return scat,

        ani = FuncAnimation(fig, update, frames=n_frame, blit=True)
        print("Simulation finished. Video processing ...\n")
        display(HTML(ani.to_jshtml()))
    
    cars_velocities = np.zeros((np.shape(c_velocities)[0],np.shape(c_velocities)[1]))
    max_velocity_per_frame = np.zeros(n_frame)
    mean_velocity_per_frame = np.zeros(n_frame)


    for i in range(np.shape(c_velocities)[0]):
        for n in range(np.shape(c_velocities)[1]):
            cars_velocities[i][n] = np.linalg.norm(c_velocities[i][n])*3.6*2*radius/(size*speed_up)
        mean_velocity_per_frame[i] = np.mean(cars_velocities[i])
        max_velocity_per_frame[i] = max(cars_velocities[i])

    mean_speed = np.mean(mean_velocity_per_frame)

    cars_accelerations = np.zeros((np.shape(c_accel)[0],np.shape(c_accel)[1]))
    max_accel_per_frame = np.zeros(n_frame)
    min_accel_per_frame = np.zeros(n_frame)
    mean_accel_per_frame = np.zeros(n_frame)

    for i in range(np.shape(c_accel)[0]):
        for n in range(np.shape(c_accel)[1]):
            cars_accelerations[i][n] = np.linalg.norm(c_accel[i][n])*3.6*2*radius/(size*speed_up)
        mean_accel_per_frame[i] = np.mean(cars_accelerations[i])
        max_accel_per_frame[i] = max(cars_accelerations[i])
        min_accel_per_frame[i] = min(cars_accelerations[i])

    mean_accel = np.mean(mean_accel_per_frame)
    
    mean_travelling_time = sum(travelling_times)/len(travelling_times)

    print(f"\nVelocity respect to real life is {speed_up} times faster \n")
    print(f"Number of accidents: {cars.number_of_accidents()}\n")
    print(f'Mean speed: {mean_speed:.2f} Km/h\n')
    print(f'Mean acceleration: {mean_accel:.2f}Km/h/s\n')
    print(f'Min accel: {min(min_accel_per_frame)}, Max accel: {max(max_accel_per_frame)}\n')
    print(f'Mean travelling time: {int(mean_travelling_time/60)}m {round((mean_travelling_time/60 - int(mean_travelling_time/60))*60,2)}s\n')
    #nodes, edges = ox.graph_to_gdfs(traffic_graph, nodes=True, edges=True)
    street_passage = [(x[0],x[1],y) for x,y in zip(edges_passage,edge_counts)]

    return cars_velocities , cars_accelerations, max_accel_per_frame, max_velocity_per_frame, mean_speed, mean_accel, travelling_times, mean_travelling_time, street_passage,accident_positions