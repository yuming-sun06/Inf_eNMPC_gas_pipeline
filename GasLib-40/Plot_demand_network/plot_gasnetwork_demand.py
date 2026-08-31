"""Plot the GasLib-40 network schematic and cyclic demand profile."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pyomo.environ as pyo
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import import_data_from_excel

# Use Matplotlib mathtext without requiring an external LaTeX installation.
plt.rcParams["text.usetex"] = False

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
INFINITE_CASE_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "Infinite")
)
NETWORK_FILE = os.path.join(
    INFINITE_CASE_DIR, "Input_data", "networkData.xlsx")
INPUT_FILE = os.path.join(
    INFINITE_CASE_DIR, "Input_data", "inputData.xlsx")

NUM_TIME_PERIODS = 1
TIME_LENGTH      = 13    # 0-12 hrs, including both endpoints
T_MAX            = 12.0


#==============================================================================
""" network_graph """
#==============================================================================
#==============================================================================
""" CREATE GRAPH """

TRANSPARENT = True
FONTSIZE = 10
AXESOFF = True
LINEWIDTH = 4.0

plt.rcParams["font.size"] = FONTSIZE
plt.rcParams["text.usetex"] = False
def set_elem_attr(G, elem, attr, isedge = True):
    """ Set attributes to an elemet of a graph """
    for a in attr.keys():
        if isedge:
            G.edges[elem][a] = attr[a] 
        else:
            G.nodes[elem][a] = attr[a]  
    return G

def graph_construction(networkData):
    
    """ 
    Build (and plot) graph from network dictionary.
    
    """
    
    #Declaration of the network graph
    G = nx.Graph()
        
    Arcs = networkData["Arcs"]
    Valves = networkData["Valves"]
    Pipes = networkData["Pipes"]
    Nodes = networkData["Nodes"]    
    Stations = networkData["Stations"]    

    # create dictionary of nodes elements
    n_elems = {}
    for elem, node in enumerate(sorted(Nodes.keys())):
        n_elems[node]=elem
            
    # build graph
    for node in Nodes.keys():
        elem = n_elems[node]
        G.add_node(elem, name=node)
        G = set_elem_attr(G, elem, Nodes[node], isedge = False)   
    
    for arc in Arcs.keys():       
        row = Arcs[arc]
        n_in = n_elems[row["nodeIN"]]
        n_out = n_elems[row["nodeOUT"]]
        if arc in Pipes.keys():
            category = "pipe"
        elif arc in Valves.keys():
            category = "valve"
        elif arc in Stations.keys():
            category = "compressor station"
        G.add_edge(n_in, n_out, array_origin = n_in, category=category)
        
        # set attributes
        elem = (n_in, n_out)
        G.edges[elem]["name"] = arc
        G = set_elem_attr(G, elem, Arcs[arc])

    return G


#==============================================================================
""" PLOT GRAPH """

def offset_labels(G, pos, offset_x = 0.1,offset_y=0):
    pos_labels = {}
    labels = {}
    labels_to_plot = {}
    keys = pos.keys()
    for key in keys:
        x, y = pos[key]
        pos_labels[key] = (x+offset_x, y+offset_y)
        labels[key] = G.nodes[key]['name']
        if G.nodes[key]['name'].startswith('source'):
            labels_to_plot[key] = 'S' + G.nodes[key]['name'].split('_', 1)[1]
        if G.nodes[key]['name'].startswith('sink'):
            labels_to_plot[key] = 'D' + G.nodes[key]['name'].split('_', 1)[1]
        if G.nodes[key]['name'].startswith('innode'):
            labels_to_plot[key] = 'N' + G.nodes[key]['name'].split('_', 1)[1]
         
    return pos_labels, labels, labels_to_plot

def graph_plot(G, node_labels=False, edge_labels = False):
    
    ## set positions
    ref_node =list(G.nodes)[0]
    pos = {n : [G.nodes[n]["long"], G.nodes[n]["lat"]] for n in G.nodes}

    ## set figure
    fig = plt.figure()        

    ## draw
    # edge_colors
    edge_colors = []
    widths = []
    for a in G.edges:
        # set color
        color = 'tab:green'
        if 'category' in G.edges[a].keys():
            if G.edges[a]["category"] == "pipe": 
                color = "black"
                width = 0.5
            elif "valve" in G.edges[a]["category"]: 
                color = "pink"
                width = 4
            elif G.edges[a]["category"] == "compressor station": 
                color = "red"  
                width = 4
            else:
                color = "black"
                width = 0.5                  
        edge_colors.append(color)
        widths.append(width)
    
    # node colors
    node_colors = ["tab:blue" for n in G.nodes]
    node_sizes = [5 for n in G.nodes]    
    nx.draw_networkx(G, pos=pos, 
            edge_color = edge_colors, 
            node_color = node_colors,
            linewidths = 4,
            with_labels = False,
            node_size = node_sizes)

    # nodes labels
    if node_labels:
        pos_labels, labels, labels_to_plot = offset_labels(G, pos, offset_x = 0, offset_y = 0.05)
        nx.draw_networkx_labels(G, pos=pos_labels, labels=labels_to_plot)

    return 

def plot_graph_with_layout(G, node_labels=False, edge_labels = False):
    ## draw
    # edge_colors
    show_edge_labels = edge_labels
    edge_colors = []
    widths = []
    edge_label_text = {}
    for a in G.edges:
        # set color
        color = 'tab:green'
        if 'category' in G.edges[a].keys():
            if G.edges[a]["category"] == "pipe": 
                color = "black"
                width = 0.5
                edge_label_text[a] = ''
            elif "valve" in G.edges[a]["category"]: 
                color = "pink"
                width = 4
                edge_label_text[a] = ''
            elif G.edges[a]["category"] == "compressor station": 
                color = "red"  
                width = 4
                edge_label_text[a] = ''
            else:
                color = "black"
                width = 0.5  
                edge_label_text[a] = ''
        edge_colors.append(color)
        widths.append(width)

    compressor_labels = {
        (3, 39): 'C4',
        (1, 18): 'C3',
        (5, 25): 'C1',
        (30, 7): 'C6',
        (38, 6): 'C5',
        (0, 10): 'C2',
    }
    for edge, label in compressor_labels.items():
        if G.has_edge(*edge):
            edge_label_text[edge] = label
    
    
    # node colors
    node_colors = ["tab:blue" for n in G.nodes]
    node_sizes = [5 for n in G.nodes]
    layout = nx.kamada_kawai_layout(G)
    
    pos_labels, labels, labels_to_plot = offset_labels(G, pos=layout)
    node_colors, node_types = colorcode_nodes(labels)
    
    plt.figure()
    nx.draw_networkx(G, pos=layout, 
            edge_color = edge_colors, 
            linewidths = 4,
            with_labels = False,
            node_size = node_sizes,
            node_color=[node_colors[node_types[node]] for node in G.nodes()])
    
    # Adjust label positions by adding the offset to the x-coordinate
    label_positions = get_label_positions(layout, labels)

    if node_labels:
        nx.draw_networkx_labels(G, pos=label_positions, labels={node: f'{label}\n' for node, label in labels_to_plot.items()},
                            font_size=7, font_color='black')
    
    if show_edge_labels:
        edge_label_positions = {node: (x + 0.05, y-0.03) for node, (x, y) in layout.items()}

        custom_label_positions = {
            3: (-0.07, 0.15),    # C4
            1: (-0.07, 0.15),    # C3
            5: (0.06, 0.1),      # C1
            0: (-0.17, 0.05),    # C2
            38: (-0.18, 0.05),   # C5
            30: (-0.08, 0.17),   # C6
        }
        for node, (dx, dy) in custom_label_positions.items():
            if node in layout:
                edge_label_positions[node] = (layout[node][0] + dx, layout[node][1] + dy)

        nx.draw_networkx_edge_labels(G, pos = edge_label_positions, edge_labels=edge_label_text, font_color='red', font_size=7, rotate=False)
   
    node_legend_items = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=7, label=r'$\mathrm{Sources (S)}$'),
                         Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=7, label=r'$\mathrm{Sinks (D)}$'),
                         Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=7, label=r'$\mathrm{Connecting \: nodes (N)}$'),
                         Line2D([0], [0], color='r', markersize=7, label=r'$\mathrm{Compressor \: lines (C)}$'),
                         Line2D([0], [0], color='k', markersize=7, label=r'$\mathrm{Pipes}$')]
    
    # Add legend to the plot
    plt.legend(handles=node_legend_items, loc='lower left')
    plt.axis("off")

    plt.savefig("gaslib40_schematic.pdf")
    plt.savefig("gaslib40_schematic.png")
    return 

def get_label_positions(layout, labels):
    label_positions = {}
    for node, (x,y) in layout.items():
        #Default label position
        label_positions[node] = (x + 0.05, y-0.03)
        node_name = labels[node]
        
        #Change label position if its not good
        if node_name.startswith('source'):
            label_positions[node] = (x - 0.05, y-0.06)
        if node_name.startswith('sink') or node_name =='innode_2' or node_name == 'innode_6':
            label_positions[node] = (x + 0.03, y+0.03)
        if node_name == 'sink_2' or node_name == 'sink_3':
            label_positions[node] = (x - 0.04, y+0.02)
        if node_name == 'innode_5':
            label_positions[node] = (x - 0.03, y+0.03)
        if node_name == 'innode_4' or node_name == 'innode_3':
            label_positions[node] = (x + 0.01, y-0.1)
        if node_name == 'sink_11' or node_name == 'innode_1' or node_name == 'innode_7' or node_name =='sink_15' or node_name == 'sink_4':
            label_positions[node] = (x + 0.07, y - 0.03)
        if node_name =='sink_10' or node_name =='innode_8' or node_name == 'sink_26' or node_name == 'sink_6':
            label_positions[node] = (x - 0.01, y - 0.1)
        if node_name=='sink_23' or node_name =='sink_29':
            label_positions[node] = (x - 0.08, y-0.05)
        if node_name =='sink_25':
            label_positions[node] = (x - 0.08, y)
        if node_name == 'sink_19':
            label_positions[node] = (x, y - 0.1)
        if node_name == 'sink_20':
            label_positions[node] = (x, y + 0.05)
    return label_positions

def colorcode_nodes(labels):
    # Define node colors based on node types
    node_colors = {'source': 'blue', 'sink': 'green', 'bypass': 'yellow', 'other':'lightpink'}

    # Extract node types from labels and assign colors accordingly
    node_types = {}
    for node, label in labels.items():
        if label.startswith('sink') or label.startswith('exit'):
            node_types[node] = 'sink'
        elif label.startswith('source') or label.startswith('entry'):
            node_types[node] = 'source'
        elif label.startswith('innode'):
            node_types[node] = 'bypass'
        else:
            node_types[node] = 'other'
    return node_colors, node_types

#==============================================================================
""" plot_dynamic_profiles """
#==============================================================================
def plot_compressor_beta(m):
    plt.figure()
    compressor_beta = {}
    for c in m.Stations:
        beta = []
        for t in m.Times:
            beta.append(pyo.value(m.compressor_beta[c, t]))
        compressor_beta[c] = beta
        plt.plot(beta, label = c)
    plt.xlabel("Time (hrs)")
    plt.ylabel("Compressor beta (Pout/Pin)")
    plt.legend()
    
def plot_compressor_power(m):
    plt.figure()
    compressor_power = {}
    for c in m.Stations:
        power = []
        for t in m.Times:
            power.append(pyo.value(m.compressor_P[c, t]))
        compressor_power[c] = power
        plt.plot(power, label = c)
    plt.xlabel("Time (hrs)")
    plt.ylabel("Compressor Power (kWh)")
    plt.legend()


def plot_demand_profile(inputData, out_file="plot_demand.pdf"):
    wcons = inputData["wCons"]

    ss_demand = None
    sink_name = None
    for key, vol_dict in sorted(wcons.items()):
        if key.startswith("sink") and 0 in vol_dict:
            ss_demand = float(list(vol_dict[0].values())[0])
            sink_name = key
            break

    if ss_demand is None:
        print("No sink demand data found.")
        return

    # Match Infinite/utils.py::dynamic_demand_calculation and the
    # 12-hour Cycle_length in Infinite/Gaussian_quadrature.py.
    shape = np.sin(
        np.linspace(0, 2*NUM_TIME_PERIODS*np.pi, TIME_LENGTH)
    )
    dyn = ss_demand + ss_demand*shape/20
    t = np.linspace(0, T_MAX, TIME_LENGTH)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, dyn, 'b-', linewidth=2.0)
    ax.set_xlabel("Time (hrs)", fontsize=16)
    ax.set_ylabel("Demand (kg/s)", fontsize=16)
    # ax.set_title("Cyclic Demand", fontsize=18)
    ax.set_xlim(0, T_MAX)
    margin = (dyn.max() - dyn.min()) * 0.1
    ax.set_ylim(dyn.min() - margin, dyn.max() + margin)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, out_file)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    png_path = out_path.replace('.pdf', '.png')
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Saved {out_path}")
    print(f"Saved {png_path}")
    print(f"  Sink: {sink_name}, ss={ss_demand:.4f} kg/s, "
          f"range=[{dyn.min():.4f}, {dyn.max():.4f}]")


def main():
    os.chdir(BASE_DIR)
    networkData, inputData = import_data_from_excel(NETWORK_FILE, INPUT_FILE)
    G = graph_construction(networkData)

    # The label positions are specific to the GasLib-40 node indices.
    plot_graph_with_layout(G, node_labels=True, edge_labels=False)
    print(f"Saved {os.path.join(BASE_DIR, 'gaslib40_schematic.pdf')}")
    print(f"Saved {os.path.join(BASE_DIR, 'gaslib40_schematic.png')}")

    plot_demand_profile(inputData, "plot_demand.pdf")
    plt.show()


if __name__ == "__main__":
    main()
