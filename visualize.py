import pydot
# from IPython.display import Image, display


# def view_pydot(pdot):
#    plt = Image(pdot.create_png())
#    display(plt)


def visualize(filename, sequence_graph, data_mapping, edge_vocab):
    data, types, parents, edges = sequence_graph
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')
    if len(data) > 0:
        nodes = [pydot.Node(i, label="'"+data_mapping[types[i]][data[i]] + "'", style="filled", fillcolor="green") for i in range(len(data))]
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
                                      label=edge_vocab[edges[i]]))

    # print(graph.to_string())

    graph.write_png(filename)
    # view_pydot(graph)
