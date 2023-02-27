import numpy as np
import networkx as nx  # ask if ok to use this
import matplotlib.pyplot as plt
import itertools  # ask if ok to use this

# open question: should I include state transistion with states where I could not end up?

gamma = 0.9  # discount factor

R = np.array(
    (
        [0, -3, -1, 0, 0, 0, 0, 0],  # start: home
        [0, 0, 0, -2, -4, 0, 0, 0],  # Auld Triangle
        [0, 0, 0, -3, -5, 0, 0, 0],  # Lötlampe
        [0, 0, 0, 0, 0, -4, -5, 0],  # Globetrotter
        [0, 0, 0, 0, 0, -5, -6, 0],  # Black Sheep
        [0, 0, 0, 0, 0, 0, 0, -6],   # Limericks 
        [0, 0, 0, 0, 0, 0, 0, -7],   # Fat Louis
        [0, 0, 0, 0, 0, 0, 0, 0],    # end: home
    )
)

STATES = {
    0: "Start: Home",
    1: "Auld Triangle",
    2: "Lötlampe",
    3: "Globetrotter",
    4: "Black Sheep",
    5: "Limericks",
    6: "Fat Louis",
    7: "End: Home",
}


def draw_graph(P):
    """Draw a multilayered graph from a numpy array."""
    # Create graph and prepare layers
    subset_sizes = [1, 2, 2, 2, 1]
    G = multilayered_graph(*subset_sizes, P=P)
    G = nx.relabel_nodes(G, STATES)

    # Draw graph
    pos = nx.multipartite_layout(G, subset_key="layer")
    pos_labels = nudge(pos, 0, 0.1)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False)
    nx.draw_networkx_labels(G, pos=pos_labels)
    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        edge_labels=nx.get_edge_attributes(G, "weight"),
        label_pos=0.35,
        rotate=False,
        verticalalignment="baseline",
    )
    plt.axis("equal")
    plt.show()


def multilayered_graph(*subset_sizes, P):
    """Create a multilayered graph from a numpy array."""
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.from_numpy_array(P, create_using=nx.DiGraph)
    for (i, layer) in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for layer1, layer2 in nx.utils.pairwise(layers):
        G.add_edges_from(itertools.product(layer1, layer2))
    return G


def nudge(pos, x_shift, y_shift):
    # https://stackoverflow.com/questions/14547388/networkx-in-python-draw-node-attributes-as-labels-outside-the-node
    """Shift the position of label by a given amount in x and y direction."""
    return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}


# state transition matrix for always going up
P_up = np.array(
    (
        [0, 1, 0, 0, 0, 0, 0, 0],  # start: home -> Auld triangle
        [0, 0, 0, 1, 0, 0, 0, 0],  # Auld Triangle -> Globetrotter
        [0, 0, 0, 1, 0, 0, 0, 0],  # Lötlampe -> Globetrotter
        [0, 0, 0, 0, 0, 1, 0, 0],  # Globetrotter -> Limericks
        [0, 0, 0, 0, 0, 1, 0, 0],  # Black Sheep -> Limericks
        [0, 0, 0, 0, 0, 0, 0, 1],  # Limericks -> end: home
        [0, 0, 0, 0, 0, 0, 0, 1],  # Fat Louis -> end: home
        [0, 0, 0, 0, 0, 0, 0, 1],  # end: home -> end: home
    )
)

# draw_graph(P_up)

# state transition matrix for always going down
P_down = np.array(
    (
        [0, 0, 1, 0, 0, 0, 0, 0],  # start: home -> Lötlampe
        [0, 0, 0, 0, 1, 0, 0, 0],  # Auld Triangle -> Black Sheep
        [0, 0, 0, 0, 1, 0, 0, 0],  # Lötlampe -> Black Sheep
        [0, 0, 0, 0, 0, 0, 1, 0],  # Globetrotter -> Fat Louis
        [0, 0, 0, 0, 0, 0, 1, 0],  # Black Sheep -> Fat Louis
        [0, 0, 0, 0, 0, 0, 0, 1],  # Limericks -> end: home
        [0, 0, 0, 0, 0, 0, 0, 1],  # Fat Louis -> end: home
        [0, 0, 0, 0, 0, 0, 0, 1],  # end: home -> end: home
    )
)

# draw_graph(P_down)

# For all first three questions: after computing state values always print them out!

# Question 1:
# compute the state value by matrix inversion

### your code here ###
# compute state transitions for the 50/50 policy
print("\n#### compute state transitions for the 50/50 policy------------------------")
P = 0.5 * P_up + 0.5 * P_down
draw_graph(P)
print(P)


# compute expected rewards for the 50/50 policy
print("\n#### compute expected rewards for the 50/50 policy-------------------------")

draw_graph(R)

# compute expected rewards for the 50/50 policy
r5050 = np.sum(P * R, axis=1)
print(r5050)

# compute state values
print("\n#### compute state values ---------------------------------------")
I = np.identity(P.shape[0])
v = np.matmul(np.linalg.inv(I - gamma*P), r5050)
print(v)

######################

# Question 2:
# compute the state values of the 50/50 policy by Richardson iteration.
# Let the iteration run so long as the nomr of the state values vector does not change by more than 0.001

### your code here ###
...
######################

# Question 3:
# compute the optimal state values by dynamic programming
# Determine the number of iterations as for the Richardson iteration

### your code here ###
...
######################
