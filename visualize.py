import pydot
# from IPython.display import Image, display


# def view_pydot(pdot):
#    plt = Image(pdot.create_png())
#    display(plt)


def visualize(filename, seq_data, sliced_types, seq_heads, seq_edges, data_mapping, edge_vocab):
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')
    if len(seq_data) > 0:
        nodes = [pydot.Node(i, label="'"+data_mapping[sliced_types[i]][seq_data[i]]+"'", style="filled", fillcolor="green") for i in range(len(seq_data))]
        for node in nodes:
            graph.add_node(node)

        # add invisible edges for alignment
        last_node = nodes[0]
        for node in nodes[1:]:
            graph.add_edge(pydot.Edge(last_node, node, weight=100, style='invis'))
            last_node = node

        for i in range(len(seq_data)):
            graph.add_edge(pydot.Edge(nodes[i], nodes[i + seq_heads[i]], label=edge_vocab[seq_edges[i]]))

    # print(graph.to_string())

    graph.write_png(filename)
    # view_pydot(graph)
