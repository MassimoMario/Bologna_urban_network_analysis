import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm

class Cars():
    def __init__(self,n_cars):
        self.journey = np.zeros((n_cars,2))
        self.road = []
        self.positions = np.zeros((n_cars,2))
        self.velocities = np.zeros((n_cars,2))
        self.accel = np.zeros((n_cars,2))
        self.max_speed = np.zeros(n_cars)
        self.turn_range = np.zeros(n_cars)
        self.travelling_time = np.zeros((n_cars,2))
        self.position_along_path = np.zeros(n_cars,dtype=int)
        self.n_cars = n_cars
        self.speed_up = 1
        self.accidents = 0


    def get_positions(self):
        return self.positions
    
    def get_velocities(self):
        return self.velocities
    
    def get_accel(self):
        return self.accel
    
    def get_n_cars(self):
        return self.n_cars
    
    def number_of_accidents(self):
        return self.accidents
    
    def set_acceleration_zero(self):
        for i in range(self.n_cars):
            self.accel[i] = (0,0)
    
    def get_velocity_angle(self,car_i):
        vx = self.velocities[car_i][0]
        v = np.linalg.norm(self.velocities[car_i])
        arccos = np.arccos(vx/v)
        if arccos > 0:
            if self.velocities[car_i][1] > 0:
                angle = arccos*180/np.pi
            elif self.velocities[car_i][1] < 0:
                angle = -arccos*180/np.pi
            else:
                angle = 0
        
        if arccos < 0:
            if self.velocities[car_i][1] > 0:
                angle = arccos*180/np.pi
            elif self.velocities[car_i][1] < 0:
                angle = -arccos*180/np.pi
            else:
                angle = 180

        if arccos == 0:
            if self.velocities[car_i][1] > 0:
                angle = 90
            elif self.velocities[car_i][1] < 0:
                angle = 270

        return angle
             
    def get_angle(self,vector):
        x = vector[0]
        y = vector[1]
        v = np.linalg.norm(vector)
        arccos = np.arccos(x/v)
        if arccos > 0:
            if y > 0:
                angle = arccos*180/np.pi
            elif y < 0:
                angle = -arccos*180/np.pi
            else:
                angle = 0
        
        if arccos < 0:
            if y > 0:
                angle = arccos*180/np.pi
            elif y < 0:
                angle = -arccos*180/np.pi
            else:
                angle = 180

        if arccos == 0:
            if y > 0:
                angle = 90
            elif y < 0:
                angle = 270

        return angle

    def another_car_is_near(self,car_i):
        for n in range(self.n_cars):
            diff_position = self.positions[n] - self.positions[car_i]
            dist = np.linalg.norm(diff_position)
            v_i = np.linalg.norm(self.velocities[car_i])
            v_n = np.linalg.norm(self.velocities[n])
            
            if n!=car_i and dist <= 2*self.turn_range[car_i] and dist > 0:
                diff_angles = np.abs(self.get_velocity_angle(n) - self.get_velocity_angle(car_i))
                diff_velocities = v_i - v_n
                angle_positions = np.abs(self.get_angle(diff_position)-self.get_velocity_angle(car_i))

                if diff_angles <= 160 and diff_angles >= 20:
                    return True , dist
                
                if diff_angles <= 20 and diff_velocities > 0 and angle_positions <= 20:
                    return True , dist
                
                else:
                    return False , dist
            else: 
                return False , dist


    def accident_probability_distribution(self,v,size,radius):
        dv = v*3.6*2*radius/(size*self.speed_up)
        p = 0.01/(0.01+np.exp(-dv*2/100))

        return p/50
    

    def accident(self,car_i,size,radius):
        for n in range(self.n_cars):
            dist = np.linalg.norm(self.positions[car_i] - self.positions[n])
            v_i = np.linalg.norm(self.velocities[car_i])
            a_i = np.linalg.norm(self.accel[car_i])
            diff_velocities = np.linalg.norm(self.velocities[car_i] - self.velocities[n])

            if n!=car_i and dist <= 10*self.turn_range[car_i]: # and a_i >= 4*size*self.speed_up/(2*radius) A THRESHOLD CAMBIALO
                diff_angles = np.abs(self.get_velocity_angle(n) - self.get_velocity_angle(car_i))
                if diff_angles <= 180 and diff_angles >= 20:
                    r = np.random.uniform(0,1)
                    if r <= self.accident_probability_distribution(diff_velocities,size,radius):
                        return True
                    else:
                        return False
                else: 
                    return False
            else: 
                return False

    def start_another_journey(self,car_i,graph,size,radius,edge_passage,edge_counts,nodes,edges,dt,closeness_centrality):
        # Convert degree centrality dictionary to an array
        nodes_list = list(closeness_centrality.keys())
        degree_values = np.array(list(closeness_centrality.values()))
        inverse_degree_values = np.array([1/(closeness_centrality[node]+1) for node in nodes_list])

        # Normalize degree centrality values to create a probability distribution
        probability_distribution = degree_values / np.sum(degree_values)
        inverse_probability_distribution = inverse_degree_values/np.sum(inverse_degree_values)

        start = np.random.choice(nodes_list, p=inverse_probability_distribution)
            
        while start not in graph.nodes:
            start = np.random.choice(nodes_list, p=inverse_probability_distribution)

        # Convert degree centrality dictionary to an array
        nodes_list = list(closeness_centrality.keys())
        degree_values = np.array(list(closeness_centrality.values()))

        # Normalize degree centrality values to create a probability distribution
        probability_distribution = degree_values / np.sum(degree_values)

        target = np.random.choice(nodes_list, p=probability_distribution)

        while target not in graph.nodes or target == start:
            target = np.random.choice(nodes_list, p=probability_distribution)
    

        road = ox.shortest_path(graph,start,target,weight='travelling_time')
        

        while road is None:
            start = np.random.choice(nodes_list, p=inverse_probability_distribution)
            while start not in graph.nodes:
                start = np.random.choice(nodes_list, p=inverse_probability_distribution)
            target = np.random.choice(nodes_list, p=probability_distribution)
            while target not in graph.nodes or target == start:
                target = np.random.choice(nodes_list, p=probability_distribution)
            road = ox.shortest_path(graph,start,target,weight='travelling_time')
            
            
        for l in range(len(road)-1):
            if (road[l],road[l+1]) in edge_passage:
                index = edge_passage.index((road[l],road[l+1]))
                edge_counts[index] += 1
                continue
            edge_passage.append((road[l],road[l+1]))
            edge_counts.append(1)
        
        #graph = ox.graph_from_gdfs(nodes,edges) #CONTROLLA E CORREGGI QUESTA PARTE MOLTO IMPORTANTE
            
        self.road[car_i] = road
        self.journey[car_i] = (start , target)

        node_direction = np.array([nodes['x'][road[1]] - nodes['x'][road[0]] , nodes['y'][road[1]] - nodes['y'][road[0]]])
        node_direction /= np.linalg.norm(node_direction)

        self.positions[car_i] = (nodes['x'][start] , nodes['y'][start])

        self.max_speed[car_i] = float(edges['maxspeed'][road[0]][road[1]][0])*self.speed_up*size/(3.6*2*radius)  #speed limit in m/s
        v = random.uniform(self.max_speed[car_i]/4 , self.max_speed[car_i])
        self.velocities[car_i] = v*node_direction

        self.turn_range[car_i] = v*dt
        

    def road_length(self,road,edges):
        if road is None:
            return 0
        edge_series = edges['length']
        length = 0
        for i in range(len(road)-1):
            length += edge_series[road[i]][road[i+1]][0]
            return length
        
    def set_initial_condition(self,graph,size,radius,speed_up,dt,edge_passage,edge_counts,nodes,edges,closeness_centrality):
        #self.max_speed = max_speed
        self.speed_up = speed_up

        for i in tqdm(range(self.n_cars)):
            # Convert degree centrality dictionary to an array
            nodes_list = list(closeness_centrality.keys())
            degree_values = np.array(list(closeness_centrality.values()))
            inverse_degree_values = np.array([1/(closeness_centrality[node]+1) for node in nodes_list])

            # Normalize degree centrality values to create a probability distribution
            probability_distribution = degree_values / np.sum(degree_values)
            inverse_probability_distribution = inverse_degree_values/np.sum(inverse_degree_values)

            start = np.random.choice(nodes_list, p=inverse_probability_distribution)
            
            while start not in graph.nodes:
                start = np.random.choice(nodes_list, p=inverse_probability_distribution)


            target = np.random.choice(nodes_list, p=probability_distribution)

            while target not in graph.nodes or target == start:
                target = np.random.choice(nodes_list, p=probability_distribution)
        

            road = ox.shortest_path(graph,start,target,weight='travelling_time')
            

            while road is None:
                start = np.random.choice(nodes_list, p=inverse_probability_distribution)
                while start not in graph.nodes:
                    start = np.random.choice(nodes_list, p=inverse_probability_distribution)
                target = np.random.choice(nodes_list, p=probability_distribution)
                while target not in graph.nodes or target == start:
                    target = np.random.choice(nodes_list, p=probability_distribution)
                road = ox.shortest_path(graph,start,target,weight='travelling_time')
                
            
            for l in range(len(road)-1):
                if (road[l],road[l+1]) in edge_passage:
                    index = edge_passage.index((road[l],road[l+1]))
                    edge_counts[index] += 1
                    continue
                edge_passage.append((road[l],road[l+1]))
                edge_counts.append(1)
                
            #graph = ox.graph_from_gdfs(nodes,edges) #CONTROLLA E CORREGGI QUESTA PARTE MOLTO IMPORTANTE
                
            self.road.append(road)
            self.journey[i] = (start , target)

            node_direction = np.array([nodes['x'][road[1]] - nodes['x'][road[0]] , nodes['y'][road[1]] - nodes['y'][road[0]]])
            node_direction /= np.linalg.norm(node_direction)

            self.positions[i] = (nodes['x'][start] , nodes['y'][start])

            self.max_speed[i] = float(edges['maxspeed'][road[0]][road[1]][0])*speed_up*size/(3.6*2*radius)  #speed limit in m/s
            v = random.uniform(self.max_speed[i]/4 , self.max_speed[i])
            self.velocities[i] = v*node_direction

            self.turn_range[i] = v*dt



    def update(self,graph,size,radius,speed_up,dt,n_frame,nodes,edges,edge_passage,edge_counts,accident_position,travelling_times,closeness_centrality):

        #nodes, _ = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        #nodes['x'] = ((nodes['x'] - min(nodes['x']))/(max(nodes['x'])-min(nodes['x'])))*size
        #nodes['y'] = ((nodes['y']-min(nodes['y']))/(max(nodes['y']) - min(nodes['y'])))*size

        for i in range(self.n_cars):
            road = self.road[i]
            v_abs = np.linalg.norm(self.velocities[i])
            L = len(road)
        
            #start another journey when target reached
            if np.linalg.norm(self.positions[i]- np.array([nodes['x'][road[L-1]] , nodes['y'][road[L-1]]])) <= self.turn_range[i]:
                self.start_another_journey(i,graph,size,radius,edge_passage,edge_counts,nodes,edges,dt,closeness_centrality)
                self.position_along_path[i] = 0
                self.travelling_time[i][0] = (n_frame * dt * speed_up) - self.travelling_time[i][1]
                self.travelling_time[i][1] = n_frame * dt * speed_up
                travelling_times.append(self.travelling_time[i][0])
                #self.velocities[i] = (0.0001,0.0001)
                #self.positions[i] = (-1,-1)
                continue
            

            #ACCELERAZIONI PER I NODI, GUARDA BENE E CAMBIA COSE 
            if np.linalg.norm(self.positions[i]-np.array([(nodes['x'][road[self.position_along_path[i]+1]]) , nodes['y'][road[self.position_along_path[i]+1]]])) <= 2*self.turn_range[i]:
                #self.accel[i] = (0,0)
                self.accel[i] = -self.velocities[i]/7
                #self.velocities[i] += self.accel[i]*dt
            elif np.linalg.norm(self.positions[i]-np.array([(nodes['x'][road[self.position_along_path[i]+1]]) , nodes['y'][road[self.position_along_path[i]+1]]])) >= 2*self.turn_range[i]:
                self.accel[i] = self.velocities[i]/5
                #self.velocities[i] += self.accel[i]*dt
                
            #another_car , dist = self.another_car_is_near(i)
            #DECELERAZIONI SE VEDE MACCHINE, ATTENZIONE A QUEL +=
            if self.another_car_is_near(i)[0]:
                self.accel[i] -= self.velocities[i]/5
                #self.accel[i] -= self.velocities[i]/self.another_car_is_near(i)[1]
                #self.velocities[i] += self.accel[i]*dt
            else:
                if v_abs <= 0.1*self.max_speed[i]:
                    #self.accel[i] = self.max_speed[i]*self.velocities[i]/(4*v_abs)
                    #self.velocities[i] += self.accel[i]*dt
                    self.accel[i] += (self.max_speed[i]-v_abs)*self.velocities[i]/v_abs
                    #self.accel[i] += self.max_speed[i]*self.velocities[i]/5

                    
            v = np.linalg.norm(self.velocities[i])
            self.turn_range[i] = v*dt

            #ACCIDENTS
            if self.accident(i,size,radius):
                self.accidents += 1   
                self.velocities[i] = (0.0,0.0)
                accident_position.append((road[self.position_along_path[i]],road[self.position_along_path[i]+1]))

            #change direction if an intermediate node is near
            if np.linalg.norm(self.positions[i]-np.array([(nodes['x'][road[self.position_along_path[i]+1]]) , nodes['y'][road[self.position_along_path[i]+1]]])) <= self.turn_range[i]:
                self.position_along_path[i] += 1
        
                v_abs = np.linalg.norm(self.velocities[i])
                self.max_speed[i] = float(edges['maxspeed'][road[self.position_along_path[i]]][road[self.position_along_path[i]+1]][0])*self.speed_up*size/(3.6*2*radius)

                node_direction = np.array([nodes['x'][road[self.position_along_path[i]+1]]- self.positions[i][0] , nodes['y'][road[self.position_along_path[i]+1]] - self.positions[i][1]])
                node_direction /= np.linalg.norm(node_direction)

                self.velocities[i] = v_abs*node_direction
                self.turn_range[i] = v_abs*dt
            
            #if exceed limit, put velocity back to speed limit
            v_abs = np.linalg.norm(self.velocities[i])
            if v_abs >= self.max_speed[i]:
                self.accel[i] -= (v_abs - self.max_speed[i])*self.velocities[i]/v_abs
                #self.velocities[i] += self.accel[i]*dt
                v = np.linalg.norm(self.velocities[i])
                self.turn_range[i] = v*dt

            '''if v_abs <= 0.05*self.max_speed[i]:
                #print(f'D: {self.accel[i]},{self.velocities[i]}')
                node_direction = np.array([nodes['x'][road[self.position_along_path[i]+1]]- self.positions[i][0] , nodes['y'][road[self.position_along_path[i]+1]] - self.positions[i][1]])
                node_direction /= np.linalg.norm(node_direction)
                self.accel[i] = self.max_speed[i]*node_direction'''

            #update positions
            self.velocities[i] += self.accel[i]*dt
            v = np.linalg.norm(self.velocities[i])
            self.turn_range[i] = v*dt
            self.positions[i] += self.velocities[i]*dt + self.accel[i]*dt*dt/2
            