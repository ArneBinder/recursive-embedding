from __future__ import print_function
import csv
import os
from sys import maxsize

import pickle
import tensorflow as tf
import numpy as np

import spacy

import constants
import corpus
import preprocessing
import sequence_node_sequence_pb2
import tools
import random
from multiprocessing import Pool
import fnmatch
import ntpath
import re

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
    'article_batch_size', 250,
    'How many articles to process in one batch.')
tf.flags.DEFINE_integer(
    'max_depth', 10,
    'The maximal depth of the sequence trees.')
#tf.flags.DEFINE_integer(
#    'sample_count', 14,
#    'Amount of samples per tree. This excludes the correct tree.')
tf.flags.DEFINE_string(
    'sentence_processor', 'process_sentence3', #'process_sentence8',#'process_sentence3',
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


def articles_from_csv_reader(filename, max_articles=100, skip=0):
    csv.field_size_limit(maxsize)
    print('parse', max_articles, 'articles...')
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['article-id', 'content'])
        i = 0
        for row in reader:
            if skip > 0:
                skip -= 1
                continue
            if i >= max_articles:
                break
            if (i * 10) % max_articles == 0:
                # sys.stdout.write("progress: %d%%   \r" % (i * 100 / max_rows))
                # sys.stdout.flush()
                print('read article:', row['article-id'], '... ', i * 100 / max_articles, '%')
            i += 1
            content = row['content'].decode('utf-8')
            # cut the title (is separated by two spaces from main content)
            yield content.split('  ', 1)[1]


@tools.fn_timer
def convert_wikipedia(in_filename, out_filename, sentence_processor, parser, mapping, max_articles=10000, max_depth=10,
                      batch_size=100, tree_mode=None):
    # get child indices depth files:
    parent_dir = os.path.abspath(os.path.join(out_filename, os.pardir))

    if not os.path.isfile(out_filename+'.data') \
            or not os.path.isfile(out_filename + '.parent'):

        if parser is None:
            print('load spacy ...')
            parser = spacy.load('en')
            parser.pipeline = [parser.tagger, parser.entity, parser.parser]
        if mapping is None:
            # TODO: move to main method (to enable usage of mappings and vecs along several corpora)
            vecs, mapping = preprocessing.create_or_read_dict(FLAGS.dict_filename, parser.vocab)

        print('parse articles ...')
        for offset in range(0, max_articles, batch_size):
            if not os.path.isfile(out_filename + '.data.batch'+str(offset)):
                current_seq_data, current_seq_parents, current_idx_tuples, current_seq_depths = preprocessing.read_data_2(articles_from_csv_reader,
                                                                  sentence_processor, parser, mapping,
                                                                  args={
                                                                      'filename': in_filename,
                                                                      'max_articles': min(batch_size, max_articles)},
                                                                  max_depth=max_depth,
                                                                  batch_size=batch_size,
                                                                  tree_mode=tree_mode)
                print('dump data, parents, depths and child indices for offset='+str(offset) + ' ...')
                current_seq_data.dump(out_filename + '.data.batch'+str(offset))
                current_seq_parents.dump(out_filename + '.parent.batch'+str(offset))
                current_seq_depths.dump(out_filename + '.depth.batch' + str(offset))
                current_idx_tuples.dump(out_filename + '.children.batch'+str(offset))

        list_seq_data = []
        list_seq_parents = []
        list_seq_depths = []
        for offset in range(0, max_articles, batch_size):
            print('read data, parents and depths for offset='+str(offset) + ' ...')
            list_seq_data.append(np.load(out_filename + '.data.batch'+str(offset)))
            list_seq_parents.append(np.load(out_filename + '.parent.batch' + str(offset)))
            list_seq_depths.append(np.load(out_filename + '.depth.batch' + str(offset)))
        print('dump concatenated data, parents and depths ...')
        seq_data = np.concatenate(list_seq_data, axis=0)
        seq_data.dump(out_filename + '.data')
        seq_parents = np.concatenate(list_seq_parents, axis=0)
        seq_parents.dump(out_filename + '.parent')
        seq_depths = np.concatenate(list_seq_depths, axis=0)
        seq_depths.dump(out_filename + '.depth')

        # remove processed batch files
        print('remove data, parents and depths batch files ...')
        data_batch_files = fnmatch.filter(os.listdir(parent_dir),
                                           ntpath.basename(out_filename) + '.data.batch*')
        parent_batch_files = fnmatch.filter(os.listdir(parent_dir),
                                          ntpath.basename(out_filename) + '.parent.batch*')
        depth_batch_files = fnmatch.filter(os.listdir(parent_dir),
                                          ntpath.basename(out_filename) + '.depth.batch*')
        for fn in data_batch_files + parent_batch_files + depth_batch_files:
            os.remove(os.path.join(parent_dir, fn))
    else:
        print('load parents from file: '+out_filename + '.parent ...')
        #seq_data = np.load(out_filename+'.data')
        seq_parents = np.load(out_filename+'.parent')
        if os.path.isfile(out_filename + '.depth'):
            print('load depths from file: ' + out_filename + '.depth ...')
            seq_depths = np.load(out_filename+'.depth')
        else:
            print('calc children and roots ...')
            seq_children, seq_roots = preprocessing.children_and_roots(seq_parents)
            # calc depth for every position
            print('calc depths ...')
            seq_depths = -np.ones(len(seq_parents), dtype=np.int)
            for root in seq_roots:
                preprocessing.calc_depth(seq_children, seq_parents, seq_depths, root)
            print('dump depths to file: ' + out_filename + '.depth ...')
            seq_depths.dump(out_filename + '.depth')

    print('collect depth indices in depth_maps ...')
    #depth_maps_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.depth.*')
    depth_map = {}
    for current_depth in range(max_depth + 1):
        depth_map[current_depth] = []
    for idx, current_depth in enumerate(seq_depths):
        #if not os.path.isfile(out_filename+'.depth.'+str(current_depth)):
        try:
            depth_map[current_depth].append(idx)
        except KeyError:
            depth_map[current_depth] = [idx]

    depths_collected_files = fnmatch.filter(os.listdir(parent_dir),
                                      ntpath.basename(out_filename) + '.data*.collected')
    if len(depths_collected_files) < max_depth:
        depths_collected = np.array([], dtype=int)
        for current_depth in reversed(sorted(depth_map.keys())):
            depths_collected = np.append(depths_collected, depth_map[current_depth])
            if current_depth < max_depth:
                np.random.shuffle(depths_collected)
                depths_collected.dump(out_filename+'.depth' + str(current_depth) + '.collected')
                print('depth: ' + str(current_depth) + ', size: ' + str(
                    len(depth_map[current_depth])) + ', collected_size: ' + str(len(depths_collected)))

    children_depth_batch_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.children.depth*.batch*')
    children_depth_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.children.depth*')
    #children_depth_files = [fn for fn in children_depth_files if fn not in children_depth_batch_files]

    if len(children_depth_batch_files) < max_articles / batch_size and len(children_depth_files) < max_depth:
        for offset in range(0, max_articles, batch_size):
            current_depth_batch_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename)+'.children.depth*.batch'+str(offset))
            # skip, if already processed
            if len(current_depth_batch_files) < max_depth:
                print('read child indices for offset='+str(offset) + ' ...')
                current_idx_tuples = np.load(out_filename + '.children.batch' + str(offset))
                print(len(current_idx_tuples))
                print('get depths ...')
                children_depths = current_idx_tuples[:, 2]
                print('argsort ...')
                sorted_indices = np.argsort(children_depths)
                print('find depth changes ...')
                depth_changes = []
                for idx, sort_idx in enumerate(sorted_indices):
                    current_depth = children_depths[sort_idx]
                    if idx == len(sorted_indices)-1 or current_depth != children_depths[sorted_indices[idx+1]]:
                        print('new depth: ' + str(current_depth) + ' ends before index pos: ' + str(idx + 1))
                        depth_changes.append((idx+1, current_depth))
                prev_end = 0
                for (end, current_depth) in depth_changes:
                    size = end - prev_end
                    print('size: '+str(size))
                    current_indices = np.zeros(shape=(size, 2), dtype=int)
                    for idx in range(size):
                        current_indices[idx] = current_idx_tuples[sorted_indices[prev_end+idx]][:2]
                    print('dump children indices with depth difference (path length from root to child): '+str(current_depth) + ' ...')
                    current_indices.dump(out_filename + '.children.depth' + str(current_depth) + '.batch' + str(offset))
                    prev_end = end
                # remove processed batch file
                os.remove(out_filename + '.children.batch' + str(offset))

    print('load and concatenate child indices batches ...')
    for current_depth in range(1, max_depth + 1):
        if not os.path.isfile(out_filename + '.children.depth' + str(current_depth)):
            print('process batches for depth=' + str(current_depth) + ' ...')
            l = []
            current_children_batch_filenames = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.children.depth' + str(current_depth) + '.batch*')
            if len(current_children_batch_filenames) > 0:
                for child_index_fn in current_children_batch_filenames:
                    l.append(np.load(os.path.join(parent_dir, child_index_fn)))
                print('concatenate ...')
                concatenated = np.concatenate(l)
                print('size: '+str(len(concatenated)))
                #print('shuffle ...')
                #np.random.shuffle(concatenated)
                print('dump to: ' + out_filename + '.children.depth' + str(current_depth) + ' ...')
                concatenated.dump(out_filename + '.children.depth' + str(current_depth))
                # remove processed batch files
                for fn in current_children_batch_filenames:
                    os.remove(os.path.join(parent_dir, fn))

    #print('create shuffled child indices ...')
    ##children_depth_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.children.depth*')
    #collected_child_indices = np.zeros(shape=(2, 0), dtype=int)
    #for current_depth in range(1, max_depth + 1):
    #    current_depth_indices = np.pad(np.load(out_filename + '.children.depth' + str(current_depth)), ((0,0),(0,1)), 'constant', constant_values=((0, 0),(0,current_depth)))
    #    collected_child_indices = np.append(collected_child_indices, current_depth_indices)
    #    if not os.path.isfile(out_filename + '.children.depth' + str(current_depth)+'.collected'):
    #        np.random.shuffle(collected_child_indices)
    #        print(len(collected_child_indices))
    #        #TODO: re-add! (crashes, files to big?)
    #        collected_child_indices.dump(out_filename + '.children.depth' + str(current_depth)+'.collected')
    #        print('depth: ' + str(current_depth) + ', collected_size: '+str(len(collected_child_indices)))
    return


if __name__ == '__main__':
    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
    out_dir = os.path.abspath(os.path.join(FLAGS.corpus_data_output_dir, sentence_processor.func_name))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, FLAGS.corpus_data_output_fn)
    if FLAGS.tree_mode is not None:
        out_path = out_path + '_' + FLAGS.tree_mode

    out_path = out_path + '_articles' + str(FLAGS.max_articles)
    out_path = out_path + '_maxdepth' + str(FLAGS.max_depth)

    print('output base file name: '+out_path)

    nlp = None
    mapping = None
    #print('handle train data ...')
    convert_wikipedia(FLAGS.corpus_data_input_train,
                      out_path,
                      sentence_processor,
                      nlp,
                      mapping,
                      max_articles=FLAGS.max_articles,
                      max_depth=FLAGS.max_depth,
                      #sample_count=FLAGS.sample_count,
                      batch_size=FLAGS.article_batch_size)

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

    corpus.write_dict(out_path, mapping, nlp.vocab, constants.vocab_manual)