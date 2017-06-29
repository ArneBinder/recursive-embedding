import copy
import os
import re

import pydot
import matplotlib.pyplot as plt
import constants
from PIL import Image
import svgutils.transform as sg


import preprocessing

TEMP_FN = 'temp_forest.svg'

if os.path.isfile(TEMP_FN):
    os.remove(TEMP_FN)


# DEPRECATED
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


def visualize(filename, sequence_graph, types):

    data, parents = sequence_graph
    # copy, because we modify parent
    parents = copy.copy(parents)
    for i, p in enumerate(parents):
        if i + p < 0 or i + p >= len(parents):
            parents[i] = 0

    graph = pydot.Dot(graph_type='digraph', rankdir='LR', bgcolor='transparent')
    if len(data) > 0:
        nodes = []
        for i, d in enumerate(data):
            if d < len(types):
                l = types[d] #data_to_word(d, data_maps_rev, vocab, vocab_neg)
            else:
                l = types[constants.UNKNOWN_EMBEDDING]
            nodes.append(pydot.Node(i, label="'" + l + "'", style="filled", fillcolor="limegreen"))

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
    graph.write_svg(filename)


def visualize_seq_node_list(seq_tree_list, types, file_name=TEMP_FN):
    for i, seq_tree in enumerate(seq_tree_list):
        current_data, current_parents = preprocessing.sequence_node_to_arrays(seq_tree)
        visualize(file_name + '.' + str(i), (current_data, current_parents), types)
    concat_visualizations_svg(file_name, len(seq_tree_list))


def visualize_list(sequence_graph_list, types, file_name=TEMP_FN):
    for i, seq_graph in enumerate(sequence_graph_list):
        current_data, current_parents = seq_graph
        visualize(file_name + '.' + str(i), (current_data, current_parents), types)
    concat_visualizations_svg(file_name, len(sequence_graph_list))


# deprecated
def concat_visualizations(file_name, count):
    file_names = [file_name + '.' + str(i) for i in range(count)]
    images = map(Image.open, file_names)
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (max_width, total_height), color='white')

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    new_im.save(file_name)
    for fn in file_names:
        os.remove(fn)


def concat_visualizations_svg(file_name, count):
    file_names = [file_name + '.' + str(i) for i in range(count)]
    # plots = [fig.getroot() for fig in map(sg.fromfile, file_names)]
    images = map(sg.fromfile, file_names)
    widths, heights = zip(*(i.get_size() for i in images))

    rx = re.compile(r"[-+]?\d*\.\d+|\d+", re.VERBOSE)
    widths = [float(rx.search(w).group()) for w in widths]
    heights = [float(rx.search(h).group()) for h in heights]

    total_height = 0
    plots = []
    for i, image in enumerate(images):
        plot = image.getroot()
        plot.moveto(0, total_height)
        plots.append(plot)
        total_height += heights[i]
    max_width = max(widths)

    fig = sg.SVGFigure(max_width, total_height)
    fig.append(plots)
    fig.save(file_name)

    for fn in file_names:
        os.remove(fn)

# deprecated
def unfold_and_plot(data, width):
    t = data.squeeze().data
    print(len(t))
    #unfolded = t.unfold(0,net.edge_count, net.edge_count).numpy()
    unfolded = t.numpy().reshape((len(t)/width, width))
    print(unfolded)
    plt.imshow(unfolded, aspect='auto', interpolation='none')


# deprecated
def getFromVocs(d_pos, d_neg, e):
    if e < 0:
        return d_neg[e]
    return d_pos[e].orth_


# deprecated
def data_to_word(d_, rev_m_, vocab_, vocab_man_):
    v_id = rev_m_[d_]
    if v_id < 0:
        t = vocab_man_[v_id]
    else:
        try:
            t = vocab_[v_id].orth_
        except IndexError:
            t = constants.vocab_manual[constants.UNKNOWN_EMBEDDING]
    return t