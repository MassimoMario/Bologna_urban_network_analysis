import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import statistics as st

# Latitudes and longitudes of 'Zona 30' cities except Minneapolis (all the italians except Bologna are in discussion)
cities = {'Bologna' : (44.495555, 11.3428), 'Ferrara' : (44.835297, 11.619865), 'Modena' : (44.64582, 10.92572), 'Reggio Emilia' : (44.7, 10.633333), 
          'Firenze' : (43.771389, 11.254167), 'Torino' : (45.079167, 7.676111), 'Napoli' : (40.833333, 14.25), 'Milano' : (45.466944, 9.19), 
          'Roma' : (41.893056, 12.482778), 'Bergamo' : (45.695, 9.67), 'Verona' : (45.438611, 10.992778), 'Lecce' : (40.352011, 18.169139), 
          'Padova' : (45.406389, 11.877778), 'Olbia' : (40.916667, 9.5), 'Pesaro' : (43.91015, 12.9133), 'Lecco' : (45.85334, 9.39048),
          'Nantes' : (47.218056, -1.552778), 'La Plata' : (-34.92125, -57.954333), 'Bilbao' : (43.266667, -2.933334),'Zurich' : (47.374444, 8.541111), 
          'Valencia' : (39.483333, -0.366667), 'Barcelona' : (41.3825, 2.176944), 'Madrid' : (40.415524, -3.707488), 'Toronto' : (43.716589, -79.340686), 
          'Bruxelles' : (50.846667, 4.351667), 'Lyon' : (45.766944, 4.834167), 'Graz' : (47.070833, 15.438611), 'Edinburgh' : (55.953333, -3.189167)}

# Street names for ZTL zone in 4 different cities
ztl = {'Bologna' : ['Via Francesco Rizzoli','Via Ugo Bassi',"Via dell'Indipendenza"],
       'Nantes' : ['Rue du Calvaire','Cours des Cinquante Otages','Cours Olivier de Clisson'],
       'Zurich' : ['Langstrasse'],
       'Edinburgh' : ['Lothian Road','Cockburn Street','North Bridge',"St Mary's Street"]}

# --------------------------------------------------------------------------------------------- #


def clean_graph_data(graph):
    '''Function cleaning data of Osmnx edges data and setting travelling times (work fine for Bologna but it's not meant to be used on other cities)
       Needs to be runned before others when initializing the graph
       It's setted on Bologna, it's not meant to be runned on other cities
    Args:
          graph : networkx.MultiDiGraph 
          
    Returns: None
    '''
    
    for _,_,data in graph.edges(data=True):
        data['passage'] = 0
        data['accident'] = 0
        if 'maxspeed' not in data:
            data['maxspeed'] = 30.0
        if type(data['maxspeed']) == list:
            data['maxspeed'] = data['maxspeed'][0]
        if data['maxspeed'] == 'signals' or data['maxspeed'] == 'IT:urban':
            data['maxspeed'] = 30.0
        if 'name' in data and type(data['name']) == list:
            data['name'] = data['name'][0]

        data['maxspeed'] = float(data['maxspeed'])
        data['travelling_time'] = data['length']*3.6 / float(data['maxspeed'])

    for _,data in graph.nodes(data=True):
        data['accident'] = 0


# --------------------------------------------------------------------------------------------- #


def set_road_network(city:str, city_radius = 3900):
    ''' Function initializing the network
    Args: 
              city : str = name of the city
       city_radius : int = radius of the city in meters

    Returns:
       graph : networkx.MultiDiGraph
 city_radius : int
    '''
    # Create urban network from latitude and longitude
    graph = ox.graph_from_point(cities[city], dist=city_radius, dist_type='bbox', network_type='drive_service', simplify = True)
    graph = ox.project_graph(graph)
    osmids = list(graph.nodes)
    graph = nx.relabel.convert_node_labels_to_integers(graph)

    # give each node its original osmid as attribute since we relabeled them
    osmid_values = {k: v for k, v in zip(graph.nodes, osmids)}
    nx.set_node_attributes(graph, osmid_values, "osmid")

    if city == 'Bologna':
        #inizializing some useful graph attributes
        clean_graph_data(graph)

        tangenziale = ['Tangenziale di Bologna', 'Autostrada Bologna-Padova','Autostrada Adriatica']
        remove_edges(graph,tangenziale)

    #G.remove_node(9)
    #G.remove_node(11)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_facecolor('black')
    ox.plot.plot_graph(graph, ax, bgcolor='black',node_size=3, node_color='white', edge_linewidth=0.5,show=False)
    plt.axis('on')
    plt.title(city + ' road network')
    plt.show()

    return graph , city_radius


# --------------------------------------------------------------------------------------------- #



def centrality_analysis(city : str, city_radius = 3900):
    '''Function showing centrality analysis
    Args:
          city : str = name of the city
   city_radius : int = radius of the city in meters

    Returns: None
    '''

    # Create urban network from latitude and longitude
    graph = ox.graph_from_point(cities[city], dist=city_radius, dist_type='bbox', network_type='drive_service', simplify = True)
    graph = ox.project_graph(graph)
    osmids = list(graph.nodes)
    graph = nx.relabel.convert_node_labels_to_integers(graph)

    # give each node its original osmid as attribute since we relabeled them
    osmid_values = {k: v for k, v in zip(graph.nodes, osmids)}
    nx.set_node_attributes(graph, osmid_values, "osmid")

    fig, axs = plt.subplots(2, 2, figsize=(13,13))

    print('\n Computing Centralities analysis . . . \n')
    # compute degree centrality
    degree_centrality = nx.degree_centrality(graph)

    # compute edge betweness centrality
    edge_betweenness_centrality = nx.edge_betweenness_centrality(graph)

    # compute betweness centrality
    betweenness_centrality = nx.betweenness_centrality(graph)

    # compute closeness centrality
    closeness_centrality = nx.closeness_centrality(graph)

    #normalized degree centrality values
    dc_values = np.array([degree_centrality[node] for node in graph.nodes])
    norm_dc_values = (dc_values - min(dc_values)) / (max(dc_values) - min(dc_values))

    #normalized edge betweenness centrality values
    ebc_values = np.array([edge_betweenness_centrality[edge] for edge in graph.edges])
    norm_ebc_values = (ebc_values - min(ebc_values)) / (max(ebc_values) - min(ebc_values))

    #normalized betweenness centrality values
    bc_values = np.array([betweenness_centrality[node] for node in graph.nodes])
    norm_bc_values = (bc_values - min(bc_values)) / (max(bc_values) - min(bc_values))

    #normalized closeness centrality values
    c_values = np.array([closeness_centrality[node] for node in graph.nodes])
    norm_c_values = (c_values - min(c_values)) / (max(c_values) - min(c_values))
    
    cmap = 'viridis'
    values = [norm_dc_values , norm_ebc_values , norm_bc_values , norm_c_values]
    names = [city + ' degree centrality map',city + ' edge betweenness centrality map', city + ' betweenness centrality map',city + ' closeness centrality map']
    map_names = ['Degree centrality',r'$C^{EBC}$',r'$C^{BC}$',r'$C^C$']
    
    for i,values in enumerate(values):
        row = i // 2
        col = i % 2
        ax = axs[row, col]

        norm=plt.Normalize(vmin=values.min(), vmax=values.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        if i == 1:
            ox.plot_graph(
                    graph,
                    ax,
                    bgcolor='white',
                    node_size=0,
                    edge_color=plt.cm.viridis(values),  
                    edge_linewidth=1, 
                    show=False)
        else: 
            ox.plot_graph(graph, ax,
                        node_color=plt.cm.viridis(values), 
                        node_size=4, 
                        edge_linewidth=0.5, 
                        bgcolor = 'white', 
                        show=False)

        cb = fig.colorbar(sm, ax=ax, label = map_names[i])
        plt.axis('on')
        ax.set_title(names[i])

    plt.show()

    fig,ax = plt.subplots(figsize=(7,6))

    hist, bins = np.histogram(bc_values, bins=500)
    hist_normalized = hist / np.sum(hist)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    
    plt.plot(bin_centers, hist_normalized*100, marker='o',markersize = 1, linestyle='-',linewidth = 1)
    plt.grid()
    plt.xscale('log')
    plt.xlabel(r'$C^{EBC}$')
    plt.ylabel('Counts [%]')
    plt.title(city + ' edge betweenness centrality values distribution')

    def percent_formatter(x,pos):
        return '{:.0f}%'.format(x)

    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    
    plt.show()

    fig,ax = plt.subplots(figsize=(7,6))

    dc = sorted((d for n, d in graph.degree()), reverse=True)
    dc = np.unique(dc, return_counts=True)

    plt.plot(dc[0],dc[1]*100/sum(dc[1]))
    plt.title(city + ' degree centrality values distribution')
    plt.xlabel('Degree')
    plt.ylabel('Counts [%]')
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.grid()
    plt.show()

    fig,ax = plt.subplots(figsize=(7,6))

    hist, bins = np.histogram(c_values, bins=500)
    

    # Normalize histogram
    hist_normalized = hist / np.sum(hist)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot points
    plt.plot(bin_centers, hist_normalized*100, marker='o',linestyle='-')
    plt.grid()
    plt.xlabel(r'$C^{C}$')
    plt.ylabel('Counts [%]')
    plt.title(city + ' closeness centrality values distribution')
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.show()


# --------------------------------------------------------------------------------------------- #
    
    

def global_efficiency(graph):
    '''Function computing global efficiency of the entire graph
    Args:
          graph : networkx.MultiDiGraph

    Returns:
         global_efficiency : float
    '''

    efficiency = 0
    N = graph.number_of_nodes()

    for i in tqdm(graph.nodes):
        for j in graph.nodes:
            if i>j:
                if nx.has_path(graph,i,j):
                    d_ij = nx.shortest_path_length(graph, i, j, weight='length')
                    efficiency += 1 / d_ij
                    global_efficiency = efficiency/(N*(N-1))
    return global_efficiency


# --------------------------------------------------------------------------------------------- #


def vulnerability(city:str,city_radius = 1100):
    '''Function computing vulnerability factor, derived from global efficiency and information centrality
    Args:
          city : str = name of the city
   city_radius : int = radius of the city in meters

       Returns: None
    '''

    # Create urban network from latitude and longitude
    graph = ox.graph_from_point(cities[city], dist=city_radius, dist_type='bbox', network_type='drive_service', simplify = True)
    graph = ox.project_graph(graph)
    osmids = list(graph.nodes)
    graph = nx.relabel.convert_node_labels_to_integers(graph)

    # give each node its original osmid as attribute since we relabeled them
    osmid_values = {k: v for k, v in zip(graph.nodes, osmids)}
    nx.set_node_attributes(graph, osmid_values, "osmid")

    nx.set_edge_attributes(graph,'darkgray','color')
    
    edges_to_color = [(u, v,k) for u, v, k,data in graph.edges(keys=True,data=True) if 'name' in data and data['name'] in ztl[city]]
    for u,v,k in edges_to_color:
        graph.edges[u,v,k]['color'] = 'r'

    # Plot the graph with specific edge colors
    ec = np.array([graph.edges[edge]['color'] for edge in graph.edges])
    ox.plot_graph(graph, bgcolor='white',node_size=3, edge_color=ec, node_color='black', show=False)

    N1 = graph.number_of_nodes()

    print('Computing global efficiency . . .\n')
    c1 = global_efficiency(graph)

    remove_edges(graph,ztl[city])

    N2 = graph.number_of_nodes()
    print(f'\nComputing global efficiency after removing {N1-N2} nodes . . .\n')
    c2 = global_efficiency(graph)

    plt.axis('on')
    plt.title(city + ' ZTL',color='white')
    print(city + f' vulnerability factor: {(c1-c2)/c1:.4f}')
    plt.show()


# --------------------------------------------------------------------------------------------- #
    


def set_max_speed_all(graph,max_speed : int):
    '''Function setting the same speed limit to every street
    Args:
        graph : networkx.MultiDiGraph
    max_speed : int = max speed to set in every streets in km/h
     
    Returns: None
    '''
    for _,_,data in graph.edges(data=True):
        data['maxspeed'] = max_speed

        #set automatically the new travelling time
        data['travelling_time'] = data['length']*3.6 / float(data['maxspeed'])


# --------------------------------------------------------------------------------------------- #
        
        
def set_max_speed(graph,street_names, max_speed : int):
    '''Function setting speed limit in a given number of streets
    Args:
     street_names : list = array of strings with streets names
        max_speed : int = max speed to set in those streets in Km/h

    Returns: None
    '''
    for _,_,data in graph.edges(data=True):
        if 'name' in data:
            if data['name'] in street_names:
                data['maxspeed'] = max_speed

                #set automatically the new travelling time
                data['travelling_time'] = data['length']*3.6 / float(data['maxspeed'])


# --------------------------------------------------------------------------------------------- #
                
                

def remove_edges(graph,street_names):
    '''Function removing streets from the graph
    Args:
        graph : networkx.MultiDiGraph
 street_names : list = list of streets to be removed from the urban network
    
    Returns : None
    '''
    edges_to_be_removed = []
    for u,v,data in graph.edges(data=True):
        if 'name' in data and data['name'] in street_names:
            edges_to_be_removed.append([u,v])
    
    for u,v in edges_to_be_removed:
        if u in graph.nodes:
            graph.remove_node(u)
        if v in graph.nodes:
            graph.remove_node(v)


# --------------------------------------------------------------------------------------------- #
            
            
def show_traffic_and_accident(graph,street_passage,accident_positions):
    '''Function showing traffic and accident
    Args:
             graph : networkx.MultiDiGraph 
    street_passage : 3d array = 3d array with initial node, final node, number of times that edge has been traveled
accident_positions : 2d array = 2d array with [initial node,final node] denoting the edge where the accident occurs

    Returns: None
    '''
    nodes,edges = ox.graph_to_gdfs(graph,nodes=True,edges=True)

    for edge in street_passage:
        if (edge[0],edge[1]) in graph.edges():
            edges.loc[(edge[0],edge[1]),'passage'][0] += edge[2]

    
    for edge in accident_positions:
        if edge[1] in graph.nodes():
            nodes.loc[edge[1],'accident'] += 1

    graph = ox.graph_from_gdfs(nodes,edges)

    # Get edge passages
    edge_values = np.array([data['passage'] for u, v, key, data in graph.edges(keys=True, data=True)])
    norm_edge_values = (edge_values - min(edge_values))/(max(edge_values - min(edge_values)))

    # Define a colormap
    cmap = plt.cm.RdBu 

    # Normalize edge lengths
    norm = plt.Normalize(min(norm_edge_values), max(norm_edge_values))

    # Map edge lengths to colors
    colors = [cmap(norm(value)) for value in norm_edge_values]

    #Plot the network with colored edges
    fig , ax = ox.plot_graph(graph, edge_color=colors, node_size=0, bgcolor='black', edge_linewidth=0.5,show=False)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax,label='Edge Passage')
    plt.title('Roads more populated')
    plt.show()

    print(f'Number of accidents: {len(accident_positions)}')

    #Get edge passages
    edge_values = np.array([data['accident'] for u,data in graph.nodes(data=True)])
    norm_edge_values = (edge_values - min(edge_values))/(max(edge_values)-min(edge_values))
    norm_edge_values = edge_values/max(edge_values)
    
    # Define a colormap
    cmap = plt.cm.RdBu 

    # Normalize edge lengths
    norm = plt.Normalize(min(norm_edge_values), max(norm_edge_values))

    # Map edge lengths to colors
    colors = [cmap(norm(value)) for value in norm_edge_values]

    #Plot the network with colored edges
    fig , ax = ox.plot_graph(graph, node_color=plt.cm.RdBu(norm_edge_values), node_size=4, bgcolor='black', edge_color = 'grey',edge_linewidth=0.2,show=False)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax,label='Accidents')
    plt.title('Incidenti')
    plt.show()


# --------------------------------------------------------------------------------------------- #
    


def connectivity_analysis(city:str, city_radius = 4000, average_path_length=True):
    '''Function showing connectivity analysis
    Args:
                city : str = name of the city
         city_radius : int = radius of the city in meters
 average_path_length : bool = if True the function shows the computation of the average path length

    Returns: None
    '''
    # Create urban network from latitude and longitude
    graph = ox.graph_from_point(cities[city], dist=city_radius, dist_type='bbox', network_type='drive_service', simplify = True)
    graph = ox.project_graph(graph)
    osmids = list(graph.nodes)
    graph = nx.relabel.convert_node_labels_to_integers(graph)

    # give each node its original osmid as attribute since we relabeled them
    osmid_values = {k: v for k, v in zip(graph.nodes, osmids)}
    nx.set_node_attributes(graph, osmid_values, "osmid")

    #connectivity analysis
    n_edges = graph.number_of_edges()
    n_nodes = graph.number_of_nodes()

    #meshedness coefficient
    alpha = (n_edges - n_nodes + 1) / (2 * n_nodes - 5)

    #connectivity
    beta = n_edges/n_nodes

    # gammaindex is a measure of the relation between the real number of edges and the number of all possible edges in a network
    gamma = n_edges/(3*(n_nodes-2))

    # Average path length
    if average_path_length:
        apl = 0
        print('Computing average path length . . .\n\n')
        for i in tqdm(graph.nodes):
            for j in graph.nodes:
                if i>j:
                    if nx.has_path(graph,i,j):
                        #if nx.has_path(graph,i,j):
                        #apl += nx.shortest_path_length(graph, i, j, weight='length')/(n_nodes*(n_nodes-1))
                        apl += nx.shortest_path_length(graph, i, j, weight='length')
        apl /= n_nodes*(n_nodes-1)
    
    print('\n'+ city + ' urban network connectivity analysis:\n')
    print('# Nodes: ', n_nodes)
    print('# Edges: ', n_edges)
    print(f"Alpha coefficient: {alpha:.2f}")
    print(f"Beta connectivity: {beta:.2f}")
    print(f"Gamma index: {gamma:.2f}")
    if average_path_length:
        print(f"Average path length: {apl:.0f} m")


# --------------------------------------------------------------------------------------------- #
    


def degree_histogram(city:str, city_radius = 4000):
    '''Function showing degree histogram
    Args:
        city : str = name of the city
 city_radius : int = radius of the city in meters

    Returns: None
    '''
    # Create urban network from latitude and longitude
    graph = ox.graph_from_point(cities[city], dist=city_radius, dist_type='bbox', network_type='drive_service', simplify = True)
    graph = ox.project_graph(graph)
    osmids = list(graph.nodes)
    graph = nx.relabel.convert_node_labels_to_integers(graph)

    # give each node its original osmid as attribute since we relabeled them
    osmid_values = {k: v for k, v in zip(graph.nodes, osmids)}
    nx.set_node_attributes(graph, osmid_values, "osmid")

    degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
    dc = np.unique(degree_sequence, return_counts = True)

    for i in range(len(dc[1])):
        if i==0:
            continue
        dc[1][i] += dc[1][i-1]

    #fig = plt.figure(city + " degree histogram", figsize=(15, 15))
    fig,ax2 = plt.subplots(figsize=(7,6))
    plt.title(city + " degree histogram")
    #axgrid = fig.add_gridspec(5, 4)

    #ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True),label='Frequency')
    ax2.set_title(city + " urban network degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")
    ax = ax2.twinx()
    plt.plot(dc[0],dc[1]/np.max(dc[1]),label='cumulative',linewidth=1,color='red')
    def percent_formatter(x, pos):
        return '{:.0f}%'.format(x * 100)

    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.legend()
    ax2.legend(loc='upper left',bbox_to_anchor=(0, 0.945))



# --------------------------------------------------------------------------------------------- #
    
def show_speed_limits(graph):
    '''Function showing speed limits
    Args:
         graph : networkx.MultiDiGraph

      Returns: None
    '''
    edge_values = np.array([data['maxspeed'] for u, v, key, data in graph.edges(keys=True, data=True) if 'maxspeed' in data])
    norm_edge_values = (edge_values - min(edge_values))/(max(edge_values - min(edge_values)))

    # Define a colormap
    cmap = plt.cm.RdBu 

    # Normalize edge lengths
    norm = plt.Normalize(min(edge_values), max(edge_values))

    # Map edge lengths to colors
    colors = [cmap(norm(value)) for value in edge_values]

    #Plot the network with colored edges
    fig , ax = ox.plot_graph(graph, edge_color=colors, node_size=0, bgcolor='black', edge_linewidth=0.5,show=False)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax,label='$\\frac{km}{h}$')
    plt.title('Speed limits',color='white')
    plt.show()

# --------------------------------------------------------------------------------------------- #
    

def città50(graph,show = True):
    '''Function setting speed limit BEFORE zona30
       It's not meant to be used on other cities than Bologna
    Args:
         graph : networkx.MultiDiGraph
          show : bool

      Returns: None
    '''
    strade = []
    limiti = []
    i = 1
    with open('strade30.txt','r') as file:
        for line in file:
        #lines = [line.strip() for line in file.readlines()]
            #values = line.split()
            if i%2 != 0:
                lines = line.strip()
                strade.append(lines)
            else: 
                limiti.append(float(line))
            i += 1
    
    set_max_speed_all(graph,50)
    for _,_,data in graph.edges(data=True):
        if 'name' in data:
            if data['name'] in strade:
                index = strade.index(data['name'])
                data['maxspeed'] = limiti[index]

    set_max_speed(graph,['Via Francesco Zanardi','Via San Donato','Via Giacomo Matteotti','Via Castiglione','Via Leonetto Cipriani',
                 'Via Attilio Muggia','Via Iacopo Barozzi','Via Vallescura','Via Cino da Pistoia','Via Francesco Petrarca','Viale Antonio Aldini',
                 'Via Bambaglioli Graziolo','Viale Pietro Pietramellara','Viale Angelo Masini','Via Amedeo Parmeggiani'],50)
    
    set_max_speed(graph,['Via Gastone Rossi','Via Edmondo De Amicis','Via Paolo Fabbri','Via Oreste Regnoli','Via Massenzio Masia','Via Mario Musolesi','Via Gianni Palmieri','Via Giuseppe Bentivogli',
                 'Via Pietro Loreta','Via Athos Bellettini','Via Serena','Via Caduti della Via Fani','Via Ferruccio Garavaglia','Via Giuseppe Gioannetti','Via Virginia Marini',
                 'Via della Villa','Via Tommaso Salvini','Via Piana','Via Michelino','Via Oreste Trebbi','Via Corrado Masetti','Via Stefano Bottari','Via Giovanni Bertini',
                 'Via Enrico Ferri','Via Francesco Bartoli','Via Ferruccio Benini','Via Angelo Beolco','Via Enrico Capelli','Piazzetta Carlo Musi','Via Filippo e Angelo Cuccoli',
                 'Via Alfonso Torreggiani','Via Carlo Cignani','Via Ambrogio Magenta','Via Giuseppe Maria Mitelli','Via Pietro Faccini','Via del Mastelletta',
                 "Via de' Gandolfi",'Via Alfonso Lombardi','Via Bartolomeo Passarotti','Via Vittorio Alfieri','Via Giorgio Vasari','Via Bruno Arnaud','Via don Giovanni Fornasini',
                 'Via Alfredo Calzolari','Via Jacopo di Palo','Via Francesco Primaticcio', 'Via Delfino Insolera','Via John Cage','Piazza Lucio Dalla','Via Valerio Zurini',
                 'Via Cesare Masina','Via Giorgio Bassani','Via Carlo Togliani','Via Agostino Bignardi','Via della Selva Pescarola',"Via della Ca' Bianca",'Via Molino di Pescarola',
                 'Via Gino Rocchi','Via Cherubino Ghirardacci','Via Gaetano Monti','Via Gian Ludovico Bianconi','Via Giuseppe Albini','Via Giuseppe Ruggi','Via Umberto Monari',
                 'Via Aristide Busi','Via Carlo Francioni','Via Gaetano Giordani','Via Luigi Silvagni','Via di Villa Pardo','Via Antonio Francesco Ghiselli',
                 'Via dei Carrettieri','Via dal Lino','Via Brigate Partigiane','Via Volontari della Libertà','Via dello Sport','Via Estro Menabue','Via Porrettana',"Via Pietro de Coubertin",
                 'Via Eugenio Curiel','Via Irma Bandiera','Via Pietro Busacchi','Via Paolo Giovanni Martini','Via M. Bastia','Via Gino Onofri','Via Giovanni Cerbai',
                 'Via XXI Aprile 1945','Via Paride Pasquali','Via Francesco Orioli','Via Fernando De Rosa','Via Luigi Valeriani','Via Antonio Zoccoli',
                 'Via Antonio Zannoni','Via Rino Ruscello','Via Marino Dalmonte','Via Giovanni Battista Melloni','Via Edoardo Brizio',
                 'Via Carlo Zucchi','Via Domenico Bianchini','Via Rodolfo Audinot','Via Giuseppe Galletti','Via Giuseppe Pacchioni','Via Andrea Costa',
                 'Via Filippo Turati', 'Via Girolamo Giaccobbi','Via Francesco Roncati','Via Giovanni Spataro','Via Giambattista Martini','Via Stanislao Mattei','Via Pietro Busacchi',
                 'Piazza della Pace','Via del Partigiano','Via Duccio Galimberti','Via Rino Ruscello','Via Martino dal Monte','Via Girolamo Giacobbi','Via Luigi Breventani',
                 'Via Alessandro Guidotti','Via Carlo Rusconi','Via Luigi Tanari','Via dello Scalo','Via Innocenzo Malvasia','Via S.Pio V','Via della Ghisiliera','Via della Secchia',
                 'Via Floriano Ambrosini','Via Amedeo Parmeggiani','Via Montello','Via Gorizia','Via Vittorio Veneto','Via Bainsizza','Via Cimabue','Via Indro Montanelli',
                 "Via Ragazzi del '99",'Via Francesco Baracca','Via Oreste Vancini','Via Greta Garbo','Via Carlo Togliani','Via Domenico Svampa',
                 'Via Melozzo da Forlì','Via Giovanni Segantini','Via Berretta Rossa','Via Camonia','Via Amedeo Modigliani','Via Innocenzo da Imola',
                 'Via Pomponia','Via Valeria','Via Lemonia','Via Decumana','Via Tiziano Vecellio','Via della Ferriera','Via Giovanni Fattori','Via Telemaco Signorini',
                 'Via Armando Spadini','Via del Giglio','Via del Cardo','Via Giorgione','Via Gino Cervi','Via Demetrio Martinelli','Via Pinturicchio',
                 'Via del Giacinto','Via Elio Bernardi','Via di Saliceto','Via Piero Gobetti'],30)
    if show:
        # Get edge maxspeed
        edge_values = np.array([data['maxspeed'] for u, v, key, data in graph.edges(keys=True, data=True)])
        norm_edge_values = (edge_values - min(edge_values))/(max(edge_values - min(edge_values)))

        # Define a colormap
        cmap = plt.cm.RdBu 

        # Normalize edge lengths
        norm = plt.Normalize(min(edge_values), max(edge_values))

        # Map edge lengths to colors
        colors = [cmap(norm(value)) for value in edge_values]

        #Plot the network with colored edges
        fig , ax = ox.plot_graph(graph, edge_color=colors, node_size=0, bgcolor='black', edge_linewidth=0.5,show=False)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax,label='$\\frac{km}{h}$')
        plt.title('Città50 speed limits',color='white')
        plt.show() 
    

# --------------------------------------------------------------------------------------------- #
    

def animate_speeds(cars_velocities,max_velocity_per_frame,n_cars,speed_up,dt):
    '''Function showing animated speeds
       It's runned inside the simulation
    Args:
         cars_velocities : 2d array with shape (#frames,#cars) = contains the norm of the velocity of every car per every frame
  max_velocity_per_frame : float 
                  n_cars : int
                speed_up : int = speed up factor from the simulation
                      dt : int = dt interval from the simulation
            
      Return: None
    '''
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()

    # Define the initial histogram
    n_bins = int(max(max_velocity_per_frame))
    #n_bins = int(max(cars_velocities[0]))
    hist, _ = np.histogram(cars_velocities[0], bins=n_bins)
    bars = ax.bar(range(n_bins), hist)

    # Update function for each frame
    def update(frame):
        ax.cla()  # Clear previous plot
        #n_bins = int(max(cars_velocities[frame]))
        hist, _ = np.histogram(cars_velocities[frame], bins=n_bins)
        ax.bar(range(n_bins), hist)
        ax.set_title('Speed')
        ax.set_xlabel('$\\frac{Km}{h}$')
        ax.set_ylabel('Frequency')

        mean = np.mean(cars_velocities[frame])
        mode = st.mode(cars_velocities[frame])
        plt.text(0.95, 0.95, f"#Cars: {n_cars}\n Mean: {mean:.2f}Km/h\n Mode: {mode:.2f}Km/h\n Time: {int(speed_up*frame*dt/60)}m {(speed_up*frame*dt/60 - int(speed_up*frame*dt/60))*60:.1f}s\n", 
            horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.shape(cars_velocities)[0])
    display(HTML(ani.to_jshtml()))

    plt.show()


# --------------------------------------------------------------------------------------------- #
    
    
def animate_accelerations(cars_accelerations,max_accel_per_frame,n_cars,speed_up,dt):
    '''Function showing animated accelerations
       It's runned inside the simulation
    Args:
         cars_accelerations : 2d array with shape (#frames,#cars) = contains the norm of the acceleration of every car per every frame
        max_accel_per_frame : float 
                  n_cars : int
                speed_up : int = speed up factor from the simulation
                      dt : int = dt interval from the simulation
            
      Return: None
    '''
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()

    # Define the initial histogram
    n_bins = int(max(max_accel_per_frame))
    #n_bins = int(max(cars_velocities[0]))
    hist, _ = np.histogram(cars_accelerations[0], bins=n_bins)
    bars = ax.bar(range(n_bins), hist)

    # Update function for each frame
    def update(frame):
        ax.cla()  # Clear previous plot
        #n_bins = int(max(cars_velocities[frame]))
        hist, _ = np.histogram(cars_accelerations[frame], bins=n_bins)
        ax.bar(range(n_bins), hist)
        ax.set_title('Accelerations')
        ax.set_xlabel('$\\frac{Km}{h \\cdot s}$')
        ax.set_ylabel('Frequency')

        mean = np.mean(cars_accelerations[frame])
        mode = st.mode(cars_accelerations[frame])
        plt.text(0.95, 0.95, f"#Cars: {n_cars}\n Mean: {mean:.2f}Km/h\n Mode: {mode:.2f}Km/h\n Time: {int(speed_up*frame*dt/60)}m {(speed_up*frame*dt/60 - int(speed_up*frame*dt/60))*60:.1f}s\n", 
            horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.shape(cars_accelerations)[0])
    display(HTML(ani.to_jshtml()))

    plt.show()


# --------------------------------------------------------------------------------------------- #
    
    
def show_travelling_times(travelling_times):
    '''Function showing travelling times
       It's runned inside the simulation
    Args:
        travelling_times : list = contains all the travelling times computed within the simulation

      Returns: None
    '''
    travelling_times = np.array(travelling_times)
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()

    # Define the initial histogram
    n_bins = int(max(travelling_times/60))
    hist, _ = np.histogram(travelling_times/60, bins=n_bins)
    bars = ax.bar(range(n_bins), hist)

    ax.set_title('Travelling times')
    ax.set_xlabel('Minutes')
    ax.set_ylabel('Frequency')

    mean_travelling_time = sum(travelling_times)/len(travelling_times)
    mode = st.mode(travelling_times)

    plt.text(0.95, 0.95, f"Mean: {int(mean_travelling_time/60)}m {(mean_travelling_time/60 - int(mean_travelling_time/60))*60:.2f}s\n Mode: {int(mode/60)}m {(mode/60 - int(mode/60))*60:.2f}s\n", 
            horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

    plt.show()