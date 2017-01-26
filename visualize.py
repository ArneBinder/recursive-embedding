import pydot
#from IPython.display import Image, display


#def view_pydot(pdot):
#    plt = Image(pdot.create_png())
#    display(plt)


def visualize(filename, seq_data, seq_heads, seq_edges, vocab):
    graph = pydot.Dot(graph_type='digraph')

    nodes = [pydot.Node(vocab[data].orth_, style="filled", fillcolor="green") for data in seq_data]
    for node in nodes:
        graph.add_node(node)

    for i in range(len(seq_data)):
        graph.add_edge(pydot.Edge(nodes[i], nodes[seq_heads[i]], label=str(seq_edges[i])))

    graph.write_png(filename)
    # view_pydot(graph)
