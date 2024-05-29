import matplotlib.pyplot as plt
import networkx as nx
import warnings
from utils import set_road_network, clean_graph_data, città50, show_speed_limits, show_traffic_and_accident
from simulation import traffic_simulation

warnings.filterwarnings("ignore")
plt.rcParams['animation.embed_limit'] = 2**128


city = 'Bologna'
city_radius = 3900
n_cars = 200  
speed_up = 200
scale = 0.4
size = 2*city_radius
dt = 0.05
n_frame = 720
animate = True 
citta50 = False


G, city__radius = set_road_network(city,city_radius)

if city == 'Bologna' and citta50:
    città50(G,show = False)
    
clean_graph_data(G)
show_speed_limits(G)

closeness_centrality = nx.closeness_centrality(G)


street_passage, accident_positions = traffic_simulation(G,
                n_cars = n_cars,
                radius = city__radius,
                speed_up = speed_up,
                closeness_centrality=closeness_centrality,
                size = 2*city_radius,
                scale = scale,
                dt = dt,
                n_frame = n_frame, 
                animate=animate)


show_traffic_and_accident(G,street_passage,accident_positions)