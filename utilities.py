import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import statistics as st

# Function cleaning data of Osmnx edges data and setting travelling times (work fine for Bologna but it's not meant to be used on other cities)
# Needs to be runned before others when initializing the graph
# It's setted on Bologna, it's not meant to be runned on other cities
def clean_graph_data(graph):
    '''function cleaning data in graph.edges and setting travelling time

        needs to be runned as first function after the urban network is initialized

        it's setted on Bologna, it's not meant to be runned on other cities '''
    
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

###
        

# Function showing centrality analysis
def centrality_analysis(graph,city : str):
    fig, axs = plt.subplots(2, 2, figsize=(13,13))

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
    map_names = ['Degree centrality','Edge Betweenness centrality','Betweenness centrality','Closeness centrality']
    
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
                        node_size=3, 
                        edge_linewidth=0.5, 
                        bgcolor = 'white', 
                        show=False)

        cb = fig.colorbar(sm, ax=ax, label = map_names[i])
        plt.axis('on')
        ax.set_title(names[i])

    plt.show()

###
    
# Function computing global efficiency of the entire graph
def global_efficiency(graph):
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    efficiency = 0
    efficiency_euclid = 0
    N = len(graph.nodes)
    for i in range(N):
        for j in range(N):
            if i!=j:
                if graph.has_edge(i,j):
                    d_ij = nx.shortest_path_length(graph, i, j, weight='length')
                    efficiency += 1 / d_ij
                    #euclid_distance = (2*np.pi*6371000*ox.distance.euclidean(nodes['lat'][i],nodes['lon'][i],nodes['lat'][j],nodes['lon'][j]))/360
                    #efficiency_euclid += 1 / euclid_distance
                    global_efficiency = efficiency/(N*(N-1))
    return global_efficiency

###

# Function setting the same speed limit to every street
def set_max_speed_all(graph,max_speed : int):
    '''max_speed : in Km/h'''
    for _,_,data in graph.edges(data=True):
        data['maxspeed'] = max_speed

        #set automatically the new travelling time
        data['travelling_time'] = data['length']*3.6 / float(data['maxspeed'])

###
        
# Function setting speed limit in a given number of streets
def set_max_speed(graph,street_names, max_speed : int):
    '''street_names : array of strings with streets names
        max_speed : in Km/h
    '''
    for _,_,data in graph.edges(data=True):
        if 'name' in data:
            if data['name'] in street_names:
                data['maxspeed'] = max_speed
                #set automatically the new travelling time
                data['travelling_time'] = data['length']*3.6 / float(data['maxspeed'])

###
                
# Function removing streets from the graph
def remove_edges(graph,street_names):
    edges_to_be_removed = []
    for u,v,data in graph.edges(data=True):
        if 'name' in data and data['name'] in street_names:
            edges_to_be_removed.append([u,v])
    
    for u,v in edges_to_be_removed:
        if u in graph.nodes:
            graph.remove_node(u)
        if v in graph.nodes:
            graph.remove_node(v)

###
            
# Function showing traffic and accident
def show_traffic_and_accident(graph,street_passage,accident_positions):
    nodes,edges = ox.graph_to_gdfs(graph,nodes=True,edges=True)

    for edge in street_passage:
        edges.loc[(edge[0],edge[1]),'passage'][0] += edge[2]
        #edges.loc[(edge[0],edge[1]),'passage'][0] += 1

    for edge in accident_positions:
        edges.loc[edge,'accident'][0] += 1

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

    # Get edge passages
    edge_values = [data['accident'] for u, v, key, data in graph.edges(keys=True, data=True)]

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
    plt.colorbar(sm, ax=ax,label='Accidents')
    plt.title('Accident')
    plt.show()

###
    

# Function showing connectivity analysis
def connectivity_analysis(graph,city:str):
    #connectivity analysis
    n_edges = graph.number_of_edges()
    n_nodes = graph.number_of_nodes()

    #meshedness coefficient
    alpha = (n_edges - n_nodes + 1) / (2 * n_nodes - 5)

    #connectivity
    beta = n_edges/n_nodes

    # gammaindex is a measure of the relation between the real number of edges and the number of all possible edges in a network
    gamma = n_edges/(3*(n_nodes-2))

    #characteristic path length
    #l_geo = nx.average_shortest_path_length(G)

    #print results
    print(city + ' urban network connectivity analysis:\n')
    print('# Edges: ', n_edges)
    print('# Nodes: ', n_nodes)
    print("meshedness coefficient:", alpha)
    print("beta connectivity:", beta)
    print("gamma index", gamma)
    #print("characteristic path length:", l_geo)

###
    

# Function showing degree histogram
def degree_histogram(graph,city:str):
    degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)

    fig = plt.figure("Degree of a random graph", figsize=(15, 15))

    axgrid = fig.add_gridspec(5, 4)



    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title(city + " urban network degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

###

# Function setting speed limit BEFORE zona30
def città50(graph,show = True):
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
        plt.colorbar(sm, ax=ax,label='Edge Passage')
        plt.title('Roads more populated')
        plt.show() 

###
        
# Function showing animated speeds
def animate_speeds(cars_velocities,max_velocity_per_frame,n_cars,speed_up,dt):
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
        ax.set_xlabel('Km/h')
        ax.set_ylabel('Frequency')

        mean = np.mean(cars_velocities[frame])
        mode = st.mode(cars_velocities[frame])
        plt.text(0.95, 0.95, f"#Cars: {n_cars}\n Mean: {mean:.2f}Km/h\n Mode: {mode:.2f}Km/h\n Time: {int(speed_up*frame*dt/60)}m {(speed_up*frame*dt/60 - int(speed_up*frame*dt/60))*60:.1f}s\n", 
            horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.shape(cars_velocities)[0])
    display(HTML(ani.to_jshtml()))

    plt.show()

###
    
# Function showing animated accelerations
def animate_accelerations(cars_accelerations,max_accel_per_frame,n_cars,speed_up,dt):
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
        ax.set_xlabel('$\\frac{Km}{h \cdot s}$')
        ax.set_ylabel('Frequency')

        mean = np.mean(cars_accelerations[frame])
        mode = st.mode(cars_accelerations[frame])
        plt.text(0.95, 0.95, f"#Cars: {n_cars}\n Mean: {mean:.2f}Km/h\n Mode: {mode:.2f}Km/h\n Time: {int(speed_up*frame*dt/60)}m {(speed_up*frame*dt/60 - int(speed_up*frame*dt/60))*60:.1f}s\n", 
            horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.shape(cars_accelerations)[0])
    display(HTML(ani.to_jshtml()))

    plt.show()