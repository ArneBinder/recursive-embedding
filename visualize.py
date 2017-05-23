import os

import pydot
import matplotlib.pyplot as plt
import constants
import sys
from PIL import Image
# from IPython.display import Image, display


# def view_pydot(pdot):
#    plt = Image(pdot.create_png())
#    display(plt)
import preprocessing


def visualize_dep(filename, sequence_graph, data_maps_rev, vocab):
    data, types, parents, edges = sequence_graph
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')
    if len(data) > 0:
        nodes = []
        for i in range(len(data)):
            if data[i] == constants.NOT_IN_WORD_DICT:
                l = constants.NOT_IN_WORD_DICT_
            else:
                v_id = data_maps_rev[types[i]][data[i]]
                l = vocab[v_id].orth_
            nodes.append(pydot.Node(i, label="'" + l + "'", style="filled", fillcolor="green"))

        for node in nodes:
            graph.add_node(node)

        # add invisible edges for alignment
        last_node = nodes[0]
        for node in nodes[1:]:
            graph.add_edge(pydot.Edge(last_node, node, weight=100, style='invis'))
            last_node = node

        for i in range(len(data)):
            if edges[i] == constants.INTER_TREE:
                label = constants.INTER_TREE_
            else:
                label = vocab[data_maps_rev[constants.EDGE_EMBEDDING][edges[i]]].orth_
            graph.add_edge(pydot.Edge(nodes[i],
                                      nodes[i + parents[i]],
                                      dir='back',
                                      label=label))

    # print(graph.to_string())

    graph.write_png(filename)
    # view_pydot(graph)


def visualize(filename, sequence_graph, data_maps_rev, vocab, vocab_neg):
    data, parents = sequence_graph
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')
    if len(data) > 0:
        nodes = []
        for i in range(len(data)):
            # print('data[i]: ' + str(data[i]))
            v_id = data_maps_rev[data[i]]
            if v_id < 0:
                l = vocab_neg[v_id]
            else:
                try:
                    l = vocab[v_id].orth_
                except IndexError:
                    l = constants.vocab_manual[constants.UNKNOWN_EMBEDDING]
            nodes.append(pydot.Node(i, label="'" + l + "'", style="filled", fillcolor="green"))

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
                                      dir='back'))

    # print(graph.to_string())

    graph.write_png(filename)


def visualize_seq_node_seq(seq_tree_seq, data_maps_rev, vocab, vocab_neg, file_name='forest_temp.png'):
    for i, seq_tree in enumerate(seq_tree_seq['trees']):
        current_data, current_parents = preprocessing.sequence_node_to_arrays(seq_tree)
        visualize(file_name + '.' + str(i), (current_data, current_parents), data_maps_rev, vocab, vocab_neg)

    file_names = [file_name + '.' + str(i) for i in range(len(seq_tree_seq['trees']))]
    images = map(Image.open, file_names)
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    new_im.save(file_name)
    for fn in file_names:
        os.remove(file_names)


def unfold_and_plot(data, width):
    t = data.squeeze().data
    print(len(t))
    #unfolded = t.unfold(0,net.edge_count, net.edge_count).numpy()
    unfolded = t.numpy().reshape((len(t)/width, width))
    print(unfolded)
    plt.imshow(unfolded, aspect='auto', interpolation='none')


def getFromVocs(d_pos, d_neg, e):
    if e < 0:
        return d_neg[e]
    return d_pos[e].orth_
