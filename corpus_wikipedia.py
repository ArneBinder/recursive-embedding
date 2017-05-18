from __future__ import print_function
import csv
import os
from sys import maxsize

import pickle
import tensorflow as tf
import numpy as np

import spacy
import preprocessing
import sequence_node_sequence_pb2
import tools
import random
from multiprocessing import Pool

tf.flags.DEFINE_string(
    'corpus_data_input_train', '/home/arne/devel/ML/data/corpora/WIKIPEDIA/documents_utf8_filtered_20pageviews.csv', # '/home/arne/devel/ML/data/corpora/SICK/sick_train/SICK_train.txt',
    'The path to the SICK train data file.')
tf.flags.DEFINE_string(
    'corpus_data_input_test', '/home/arne/devel/ML/data/corpora/SICK/sick_test_annotated/SICK_test_annotated.txt',
    'The path to the SICK test data file.')
tf.flags.DEFINE_string(
    'corpus_data_output_dir', '/media/arne/WIN/Users/Arne/tf/data/corpora/wikipedia',#'data/corpora/wikipedia',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'corpus_data_output_fn', 'WIKIPEDIA',
    'Base filename of the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'dict_filename', 'data/nlp/spacy/dict',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_integer(
    'max_articles', 10000,
    'How many articles to read.')
tf.flags.DEFINE_integer(
    'max_depth', 10,
    'The maximal depth of the sequence trees.')
tf.flags.DEFINE_integer(
    'sample_count', 14,
    'Amount of samples per tree. This excludes the correct tree.')
tf.flags.DEFINE_string(
    'sentence_processor', 'process_sentence3',
    'How long to make the expression embedding vectors.')
tf.flags.DEFINE_string(
    'tree_mode',
    None,
    #'aggregate',
    #'sequence',
    'How to structure the tree. '
     + '"sequence" -> parents point to next token, '
     + '"aggregate" -> parents point to an added, artificial token (TERMINATOR) in the end of the token sequence,'
     + 'None -> use parsed dependency tree')

FLAGS = tf.flags.FLAGS


def articles_from_csv_reader(filename, max_articles=100):
    csv.field_size_limit(maxsize)
    print('parse', max_articles, 'articles...')
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['article-id', 'content'])
        i = 0
        for row in reader:
            if i >= max_articles:
                break
            if (i * 100) % max_articles == 0:
                # sys.stdout.write("progress: %d%%   \r" % (i * 100 / max_rows))
                # sys.stdout.flush()
                print('read article:', row['article-id'], '... ', i * 100 / max_articles, '%')
            i += 1
            content = row['content'].decode('utf-8')
            # cut the title (is separated by two spaces from main content)
            yield content.split('  ', 1)[1]


@tools.fn_timer
def convert_wikipedia(in_filename, out_filename, sentence_processor, parser, mapping, max_articles=10000, max_depth=10, sample_count=15, tree_mode=None):
    if os.path.isfile(out_filename+'.data'):
        print('load data and parents from files: '+out_filename + ' ...')
        seq_data = np.load(out_filename+'.data')
        seq_parents = np.load(out_filename+'.parents')
    else:
        if parser is None:
            print('load spacy ...')
            parser = spacy.load('en')
            parser.pipeline = [parser.tagger, parser.entity, parser.parser]
        if mapping is None:
            vecs, mapping = preprocessing.create_or_read_dict(FLAGS.dict_filename, parser.vocab)
        print('parse articles ...')
        seq_data, seq_parents = preprocessing.read_data(articles_from_csv_reader,
                                                        sentence_processor, parser, mapping,
                                                        args={
                                                            'filename': in_filename,
                                                            'max_articles': max_articles},
                                                        tree_mode=tree_mode)
        print('dump data and parents ...')
        seq_data.dump(out_filename + '.data')
        seq_parents.dump(out_filename + '.parents')

    print('data points: '+str(len(seq_data)))

    print('calc children and roots ...')
    children, roots = preprocessing.children_and_roots(seq_parents)

    if os.path.isfile(out_filename + '.depth'):
        print('load depth from: ' + out_filename + '.depth ...')
        depth = np.load(out_filename + '.depth')
    else:
        print('calc depths ...')
        depth = -np.ones(len(seq_data), dtype=np.int)
        for root in roots:
            preprocessing.calc_depth(children, seq_parents, depth, root)
        print('dump depth to: ' + out_filename + '.depth ...')
        depth.dump(out_filename + '.depth')

    have_children = np.array(children.keys())

    print('calc indices ...')
    idx_tuples = np.zeros(shape=(0, 3), dtype=int)
    #print('current_depth: '+str(current_depth))
    for i, idx in enumerate(have_children):
        if (i * 100) % len(have_children) == 0:
            print('generate tuples ... ', i * 100 / len(have_children), '%')
        for current_depth in range(1, min(max_depth, depth[idx])):
            for (child, child_steps_to_root) in preprocessing.get_all_children_rec(idx, children, current_depth):
                idx_tuples = np.append(idx_tuples, [[idx, child, child_steps_to_root]])

    #p = Pool(4)
    #idx_tuples = np.array(p.map(f, have_children))
    #print(p.map(f, have_children))
    print('shuffle indices ...')
    np.random.shuffle(idx_tuples)
    print('dump indices ...')
    idx_tuples.dump(out_filename + '.indices')
    return

    #print('sort depths ...')
    #print(np.sort(depth)[-100:])
    #for i, d in enumerate(depth):
    #    if d >= 500:
    #        print(str(i) + ': '+str(d))
    print('calc depth maps ...')
    depth_maps = {}
    for i, d in enumerate(depth):
        try:
            depth_maps[d].append(i)
        except KeyError:
            depth_maps[d] = [i]
    #print(len(depth_maps))
    # get maximum depth of parsed data
    real_max_depth = max(depth_maps.keys())
    print('real_max_depth: '+str(real_max_depth))
    print('max_depth: ' + str(max_depth))

    depth_counts_summed = {}
    depth_counts_sum = 0
    for i in reversed(sorted(depth_maps.keys())):
        depth_counts_sum += len(depth_maps[i])
        depth_counts_summed[i] = depth_counts_sum
        #print(str(i)+': '+str(len(depth_maps[i])))

    def get_idx_from_depth_maps(r):
        for k in depth_maps.keys():
            l = len(depth_maps[k])
            if r < l:
                return depth_maps[k][r]
            else:
                r -= l

    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    # create for every depth a separate dataset
    for current_depth in range(1, max_depth+1):
        current_fn = out_filename+'.trees.'+ str(current_depth)
        # process only, if not already done
        if not os.path.isfile(current_fn + '.indices'):
            # walk all data points with depth >= current_depth
            # init indices with -1
            currend_indices = -np.ones([depth_counts_summed[current_depth]], dtype=int)
            pos = 0
            for min_depth in range(current_depth, real_max_depth+1):
                if min_depth in depth_maps:
                    print('current_depth: '+str(current_depth) + ', min_depth: '+str(min_depth) + ', count: '+str(len(depth_maps[min_depth])))
                    np_temp = np.array(depth_maps[min_depth])
                    currend_indices[pos:pos+len(depth_maps[min_depth])] = np_temp
                    pos += len(depth_maps[min_depth])
            print('shuffle indices ...')
            np.random.shuffle(currend_indices)
            print('dump indices to: '+ current_fn + '.indices ...')
            currend_indices.dump(current_fn + '.indices')
        else:
            print('load indices from: ' + current_fn + '.indices ...')
            currend_indices = np.load(current_fn + '.indices')


        # if not os.path.isfile(current_fn):
        #     print('write records to: ' + current_fn + ' ...')
        #     record_output = tf.python_io.TFRecordWriter(current_fn)
        #     for idx in currend_indices:
        #         # for every child in current tree (with root idx)
        #         for (child, child_steps_to_root) in preprocessing.get_all_children_rec(idx, children, current_depth):
        #             child_depth = current_depth - child_steps_to_root #depth[child]
        #             new_tree_seq = sequence_node_sequence_pb2.SequenceNodeSequence()
        #             # add original
        #             preprocessing.build_sequence_tree_with_candidate(seq_data, children, idx, child,
        #                                                              child, current_depth, child_depth,
        #                                                              new_tree_seq.trees.add())
        #             # create sample_count samples
        #             for _ in range(sample_count):
        #                 candidate_rand = random.randint(0, depth_counts_summed[child_depth]-1)
        #                 candidate_idx = get_idx_from_depth_maps(candidate_rand)
        #                 preprocessing.build_sequence_tree_with_candidate(seq_data, children, idx, child,
        #                                                                  candidate_idx, current_depth, child_depth,
        #                                                                  new_tree_seq.trees.add())
        #             record_output.write(new_tree_seq.SerializeToString())
        #     record_output.close()


if __name__ == '__main__':
    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
    out_dir = os.path.abspath(os.path.join(FLAGS.corpus_data_output_dir, sentence_processor.func_name))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, FLAGS.corpus_data_output_fn)
    if FLAGS.tree_mode is not None:
        out_path = out_path + '_' + FLAGS.tree_mode

    out_path = out_path + '_' + str(FLAGS.max_articles) + 'articles'

    nlp = None
    mapping = None
    print('handle train data ...')
    convert_wikipedia(FLAGS.corpus_data_input_train,
                      out_path,
                      sentence_processor,
                      nlp,
                      mapping,
                      max_articles=FLAGS.max_articles,
                      max_depth=FLAGS.max_depth,
                      sample_count=FLAGS.sample_count,
                      tree_mode=FLAGS.tree_mode)

    #print('parse train data ...')
    #convert_sick(FLAGS.corpus_data_input_train,
    #             out_path + '.train',
    #             sentence_processor,
    #             nlp,
    #             mapping,
    #             FLAGS.corpus_size,
    #             FLAGS.tree_mode)

    #print('parse test data ...')
    #convert_sick(FLAGS.corpus_data_input_test,
    #             out_path + '.test',
    #             sentence_processor,
    #             nlp,
    #             mapping,
    #             FLAGS.corpus_size,
    #             FLAGS.tree_mode)

    #print('len(mapping): ' + str(len(mapping)))

    #corpus.write_dict(out_path, mapping, nlp.vocab, constants.vocab_manual)