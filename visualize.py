import pydot
import matplotlib.pyplot as plt
import constants
# from IPython.display import Image, display


# def view_pydot(pdot):
#    plt = Image(pdot.create_png())
#    display(plt)


def visualize(filename, sequence_graph, data_maps_human):
    data, types, parents, edges = sequence_graph
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')
    if len(data) > 0:
        nodes = [pydot.Node(i, label="'"+data_maps_human[types[i]][data[i]] + "'", style="filled", fillcolor="green") for i in range(len(data))]
        for node in nodes:
            graph.add_node(node)

        # add invisible edges for alignment
        last_node = nodes[0]
        for node in nodes[1:]:
            graph.add_edge(pydot.Edge(last_node, node, weight=100, style='invis'))
            last_node = node

        for i in range(len(data)):
            graph.add_edge(pydot.Edge(nodes[i],
                                      nodes[i + parents[i]],
                                      label=data_maps_human[constants.EDGE_EMBEDDING][edges[i]]))

    # print(graph.to_string())

    graph.write_png(filename)
    # view_pydot(graph)


def unfold_and_plot(data, width):
    t = data.squeeze().data
    print(len(t))
    #unfolded = t.unfold(0,net.edge_count, net.edge_count).numpy()
    unfolded = t.numpy().reshape((len(t)/width, width))
    print(unfolded)
    plt.imshow(unfolded, aspect='auto', interpolation='none')
