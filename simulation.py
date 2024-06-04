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
import utils as ut


def traffic_simulation(graph, n_cars, radius, speed_up, closeness_centrality, size, scale = 0.3, dt = 0.05, n_frame = 300, animate = True):
    
    '''Function showing traffic simulation and statistics animations
    Args: 
               graph : networkx.MultiDiGraph
              n_cars : int = number of cars to simulate
              radius : int = radius of the city in meters, it must be the same used for initializing graph
            speed_up : int = speed up factor making simulation faster than reality 
closeness_centrality : list = list of closeness centrality values used fot predict initial and final node of a car path
                size : int = refers to maximum value of animation image boundaries. It will be converted in meters using Bologna diameter
               scale : float = refers to scale of arrows dimension within the animation
                  dt : float = time interval with which the simulation is computed
            n_frames : int = number of frames of the simulation
             animate : Bool = if True shows traffic animation, speeds animation and accelerations animation

        Return:
     streets:passage : 3d array = 3d array with [initial node, final node, number of times that edge has been traveled]
  accident_positions : 2d array = 2d array with [initial node,final node] denoting the edge where the accident occurs
       '''
    cars = Cars(n_cars)
    edges_passage = []
    edge_counts = []
    accident_positions = []
    travelling_times = []

    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    nodes['x'] = ((nodes['x'] - min(nodes['x']))/(max(nodes['x'])-min(nodes['x'])))*size
    nodes['y'] = ((nodes['y']-min(nodes['y']))/(max(nodes['y']) - min(nodes['y'])))*size
    
    print('\nInitialization: \n')
    cars.set_initial_condition(graph,size,radius,speed_up,dt,edges_passage,edge_counts,nodes,edges,closeness_centrality)
    print('Initialization done\n')

    n_cars = cars.get_n_cars()

    # three array that will be used for the animation
    c_positions = np.zeros((n_frame, n_cars, 2))
    c_velocities = np.zeros((n_frame, n_cars, 2))
    c_accel = np.zeros((n_frame, n_cars, 2))

    print('Simulation: \n')
    
    for n in tqdm(range(n_frame)):
        c_positions[n] = cars.get_positions()
        c_velocities[n] = cars.get_velocities()
        c_accel[n] = cars.get_accel()
        
        cars.set_acceleration_zero()

        # updating cars velocities and positions in every frame
        cars.update(graph,size,radius,speed_up,dt,n,nodes,edges,edges_passage,edge_counts,accident_positions,travelling_times,closeness_centrality)
        

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

    # computing mean accelerations, travel time and total time simulated
    mean_accel = np.mean(mean_accel_per_frame)
    
    mean_travelling_time = sum(travelling_times)/len(travelling_times)

    total_time = n_frame*dt*speed_up

    # printing statistics
    print('Simulation done: \n')
    print(f'\nSimulated time: {int(total_time/60)} m {(total_time/60 - int(total_time/60))*60:.2f}s')
    print(f"Velocity respect to real life is {speed_up} times faster \n")
    print(f"Number of accidents: {cars.number_of_accidents()}\n")
    print(f'Mean speed: {mean_speed:.2f} km/h \n')
    print(f'Mean acceleration: {mean_accel:.2f} Km/h/s\n')
    print(f'Mean travel time: {int(mean_travelling_time/60)} m {(mean_travelling_time/60 - int(mean_travelling_time/60))*60:.2f} s\n')
    
    #filling street_passage that will be used to show the most used streets
    street_passage = [(x[0],x[1],y) for x,y in zip(edges_passage,edge_counts)]

    # showing travel times histogram
    ut.show_travelling_times(travelling_times)

    # if animate is True an animation of the simulation is shown
    if animate:
        fig, ax = plt.subplots(figsize=(6,6))

        for N in graph.nodes:
            circle = plt.Circle((nodes['x'][N] , nodes['y'][N]) , size/900, color = 'grey')
            ax.add_patch(circle)
        
        
        velocities_magnitudes = np.linalg.norm(c_velocities[0], axis=1)
        velocities_normalized = c_velocities[0] / np.vstack([velocities_magnitudes, velocities_magnitudes]).T 
        
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

        # function computing the animation
        ani = FuncAnimation(fig, update, frames=n_frame, blit=True)
        print("Video processing . . .\n")
        display(HTML(ani.to_jshtml()))

        # showing speeds and accelerations animation
        print('Animating speeds . . .\n')
        ut.animate_speeds(cars_velocities,max_velocity_per_frame,n_cars,speed_up,dt)

        print('Animate accelerations . . .\n')
        ut.animate_accelerations(cars_accelerations,max_accel_per_frame,n_cars,speed_up,dt)

    return street_passage, accident_positions