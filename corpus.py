import fnmatch
import logging
import ntpath
import os

import numpy as np
import spacy

import constants
import lexicon as lex
import preprocessing
import sequence_trees as sequ_trees


def load(filename):
    vecs, types = lex.load(filename)
    data, parents = sequ_trees.load(filename)
    return vecs, types, data, parents


def dump(filename, vecs=None, types=None, data=None, parents=None):
    lex.dump(filename, vecs, types)
    sequ_trees.dump(filename, data, parents)


def exist(filename):
    return sequ_trees.exist(filename) and lex.exist(filename)


#@mytools.fn_timer
def convert_texts(in_filename, out_filename, init_dict_filename, sentence_processor, parser, reader,  # mapping, vecs,
                  max_articles=10000, max_depth=10, batch_size=100, article_offset=0, count_threshold=2,
                  concat_mode='sequence', inner_concat_mode='tree'):
    parent_dir = os.path.abspath(os.path.join(out_filename, os.pardir))
    out_base_name = ntpath.basename(out_filename)
    if not os.path.isfile(out_filename + '.data') \
            or not os.path.isfile(out_filename + '.parent') \
            or not os.path.isfile(out_filename + '.vec') \
            or not os.path.isfile(out_filename + '.depth') \
            or not os.path.isfile(out_filename + '.count'):
        if not parser:
            logging.info('load spacy ...')
            parser = spacy.load('en')
            parser.pipeline = [parser.tagger, parser.entity, parser.parser]
        # get vecs and types and save it at out_filename
        if init_dict_filename:
            vecs, types = lex.create_or_read_dict(init_dict_filename)
            lex.dump(out_filename, vecs=vecs, types=types)
        else:
            lex.create_or_read_dict(out_filename, vocab=parser.vocab, dont_read=True)

        # parse
        parse_texts(out_filename=out_filename, in_filename=in_filename, reader=reader, parser=parser,
                    sentence_processor=sentence_processor, max_articles=max_articles,
                    batch_size=batch_size, concat_mode=concat_mode, inner_concat_mode=inner_concat_mode,
                    article_offset=article_offset, add_reader_args={})
        # merge batches
        merge_numpy_batch_files(out_base_name + '.parent', parent_dir)
        merge_numpy_batch_files(out_base_name + '.depth', parent_dir)
        seq_data = merge_numpy_batch_files(out_base_name + '.data', parent_dir)
    else:
        logging.debug('load data from file: ' + out_filename + '.data ...')
        seq_data = np.load(out_filename + '.data')

    logging.info('data size: ' + str(len(seq_data)))

    if not os.path.isfile(out_filename + '.converter') \
            or not os.path.isfile(out_filename + '.new_idx_unknown') \
            or not os.path.isfile(out_filename + '.lex_size'):
        if not parser:
            logging.info('load spacy ...')
            parser = spacy.load('en')
            parser.pipeline = [parser.tagger, parser.entity, parser.parser]
        vecs, types = lex.create_or_read_dict(out_filename, parser.vocab)
        # sort and filter vecs/mappings by counts
        converter, vecs, types, counts, new_idx_unknown = lex.sort_and_cut_and_fill_dict(seq_data, vecs, types,
                                                                                     count_threshold=count_threshold)
        # write out vecs, mapping and tsv containing strings
        lex.dump(out_filename, vecs=vecs, types=types, counts=counts)
        logging.debug('dump converter to: ' + out_filename + '.converter ...')
        converter.dump(out_filename + '.converter')
        logging.debug('dump new_idx_unknown to: ' + out_filename + '.new_idx_unknown ...')
        np.array(new_idx_unknown).dump(out_filename + '.new_idx_unknown')
        logging.debug('dump lex_size to: ' + out_filename + '.lex_size ...')
        np.array(len(types)).dump(out_filename + '.lex_size')

    if os.path.isfile(out_filename + '.converter') and os.path.isfile(out_filename + '.new_idx_unknown'):
        logging.debug('load converter from file: ' + out_filename + '.converter ...')
        converter = np.load(out_filename + '.converter')
        logging.debug('load new_idx_unknown from file: ' + out_filename + '.new_idx_unknown ...')
        new_idx_unknown = np.load(out_filename + '.new_idx_unknown')
        logging.debug('load lex_size from file: ' + out_filename + '.lex_size ...')
        lex_size = np.load(out_filename + '.lex_size')
        logging.debug('lex_size=' + str(lex_size))
        seq_data = sequ_trees.convert_data(seq_data, converter, lex_size, new_idx_unknown)
        logging.debug('dump data to: ' + out_filename + '.data ...')
        seq_data.dump(out_filename + '.data')
        logging.debug('delete converter, new_idx_unknown and lex_size ...')
        os.remove(out_filename + '.converter')
        os.remove(out_filename + '.new_idx_unknown')
        os.remove(out_filename + '.lex_size')

    logging.debug('load depths from file: ' + out_filename + '.depth ...')
    seq_depths = np.load(out_filename + '.depth')
    preprocessing.calc_depths_collected(out_filename, parent_dir, max_depth, seq_depths)

    logging.debug('load parents from file: ' + out_filename + '.parent ...')
    seq_parents = np.load(out_filename + '.parent')
    logging.debug('collect roots ...')
    roots = []
    for i, parent in enumerate(seq_parents):
        if parent == 0:
            roots.append(i)
    logging.debug('dump roots to: ' + out_filename + '.root ...')
    np.array(roots, dtype=np.int32).dump(out_filename + '.root')

    # preprocessing.rearrange_children_indices(out_filename, parent_dir, max_depth, max_articles, batch_size)
    # preprocessing.concat_children_indices(out_filename, parent_dir, max_depth)

    # logging.info('load and concatenate child indices batches ...')
    # for current_depth in range(1, max_depth + 1):
    #    if not os.path.isfile(out_filename + '.children.depth' + str(current_depth)):
    #        preprocessing.merge_numpy_batch_files(out_base_name + '.children.depth' + str(current_depth), parent_dir)

    # for re-usage
    return parser


def parse_texts(out_filename, in_filename, reader, parser, sentence_processor, max_articles, batch_size, concat_mode,
                inner_concat_mode, article_offset, add_reader_args={}):
    types = lex.read_types(out_filename)
    mapping = lex.mapping_from_list(types)
    logging.info('parse articles ...')
    for offset in range(0, max_articles, batch_size):
        # all or none: otherwise the mapping lacks entries!
        if not os.path.isfile(out_filename + '.data.batch' + str(offset)) \
                or not os.path.isfile(out_filename + '.parent.batch' + str(offset)) \
                or not os.path.isfile(out_filename + '.depth.batch' + str(offset)):
            logging.info('parse articles for offset=' + str(offset) + ' ...')
            current_reader_args = {
                'filename': in_filename,
                'max_articles': min(batch_size, max_articles),
                'skip': offset + article_offset
            }
            current_reader_args.update(add_reader_args)
            #current_seq_data, current_seq_parents, current_seq_depths = preprocessing.read_data(
            current_seq_data, current_seq_parents = preprocessing.read_data(
                reader=reader,
                sentence_processor=sentence_processor,
                parser=parser,
                data_maps=mapping,
                reader_args=current_reader_args,
                # max_depth=max_depth,
                batch_size=batch_size,
                concat_mode=concat_mode,
                inner_concat_mode=inner_concat_mode,
                #calc_depths=True,
                # child_idx_offset=child_idx_offset
            )
            lex.dump(out_filename, types=lex.revert_mapping_to_list(mapping))
            logging.info('dump data, parents and depths ...')
            current_seq_data.dump(out_filename + '.data.batch' + str(offset))
            current_seq_parents.dump(out_filename + '.parent.batch' + str(offset))
            current_seq_depths.dump(out_filename + '.depth.batch' + str(offset))


def parse_texts_clustered(filename, reader, reader_clusters, sentence_processor, parser, mapping, concat_mode,
                          inner_concat_mode):
    logging.info('convert texts scored ...')
    logging.debug('len(mapping)=' + str(len(mapping)))
    data, parents = preprocessing.read_data(reader=reader, sentence_processor=sentence_processor,
                                               parser=parser, reader_args={'filename': filename}, data_maps=mapping,
                                               batch_size=10000, concat_mode=concat_mode,
                                               inner_concat_mode=inner_concat_mode, expand_dict=True)
    logging.debug('len(mapping)=' + str(len(mapping)) + '(after parsing)')
    roots = [idx for idx, parent in enumerate(parents) if parent == 0]
    logging.debug('len(roots)=' + str(len(roots)))
    clusters = list(reader_clusters(filename))  # list(reader_scores(filename))
    logging.debug('len(clusters)=' + str(len(clusters)))
    assert len(clusters) == len(roots), 'len(roots):' + str(len(roots)) + ' != len(clusters):' + str(len(clusters))

    return data, parents, clusters, roots


def merge_numpy_batch_files(batch_file_name, parent_dir, expected_count=None, overwrite=False):
    logging.info('concatenate batches: ' + batch_file_name)
    # out_fn = batch_file_name.replace('.batch*', '', 1)
    if os.path.isfile(batch_file_name) and not overwrite:
        return np.load(batch_file_name)
    batch_file_names = fnmatch.filter(os.listdir(parent_dir), batch_file_name + '.batch*')
    batch_file_names = sorted(batch_file_names, key=lambda s: int(s[len(batch_file_name + '.batch'):]))
    if len(batch_file_names) == 0:
        return None
    if expected_count is not None and len(batch_file_names) != expected_count:
        return None
    l = []
    for fn in batch_file_names:
        l.append(np.load(os.path.join(parent_dir, fn)))
    concatenated = np.concatenate(l, axis=0)

    concatenated.dump(os.path.join(parent_dir, batch_file_name))
    for fn in batch_file_names:
        os.remove(os.path.join(parent_dir, fn))
    return concatenated


##### CURRENTLY UNUSED #################################################################################################


def calc_depths_collected(out_filename, parent_dir, max_depth, seq_depths):
    depths_collected_files = fnmatch.filter(os.listdir(parent_dir),
                                            ntpath.basename(out_filename) + '.depth*.collected')
    if len(depths_collected_files) < max_depth:
        logging.info('collect depth indices in depth_maps ...')
        # depth_maps_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.depth.*')
        depth_map = {}

        for idx, current_depth in enumerate(seq_depths):
            # if not os.path.isfile(out_filename+'.depth.'+str(current_depth)):
            try:
                depth_map[current_depth].append(idx)
            except KeyError:
                depth_map[current_depth] = [idx]

        # fill missing depths
        real_max_depth = max(depth_map.keys())
        for current_depth in range(real_max_depth + 1):
            if current_depth not in depth_map:
                depth_map[current_depth] = []

        depths_collected = np.array([], dtype=np.int16)
        for current_depth in reversed(sorted(depth_map.keys())):
            # if appending an empty list to an empty depths_collected, the dtype will change to float!
            if len(depth_map[current_depth]) > 0:
                depths_collected = np.append(depths_collected, depth_map[current_depth])
            if current_depth < max_depth:
                np.random.shuffle(depths_collected)
                depths_collected.dump(out_filename + '.depth' + str(current_depth) + '.collected')
                logging.info('depth: ' + str(current_depth) + ', size: ' + str(
                    len(depth_map[current_depth])) + ', collected_size: ' + str(len(depths_collected)))


def batch_file_count(total_count, batch_size):
    return total_count / batch_size + (total_count % batch_size > 0)


# unused
def rearrange_children_indices(out_filename, parent_dir, max_depth, max_articles, batch_size):
    # not yet used
    # child_idx_offset = 0
    ##
    children_depth_batch_files = fnmatch.filter(os.listdir(parent_dir),
                                                ntpath.basename(out_filename) + '.children.depth*.batch*')
    children_depth_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.children.depth*')
    if len(children_depth_batch_files) < batch_file_count(max_articles, batch_size) and len(
            children_depth_files) < max_depth:
        for offset in range(0, max_articles, batch_size):
            current_depth_batch_files = fnmatch.filter(os.listdir(parent_dir),
                                                       ntpath.basename(out_filename) + '.children.depth*.batch' + str(
                                                           offset))
            # skip, if already processed
            if len(current_depth_batch_files) < max_depth:
                logging.info('read child indices for offset=' + str(offset) + ' ...')
                current_idx_tuples = np.load(out_filename + '.children.batch' + str(offset))
                # add offset
                # current_idx_tuples += np.array([child_idx_offset, 0, 0])
                logging.info(len(current_idx_tuples))
                logging.info('get depths ...')
                children_depths = current_idx_tuples[:, 2]
                logging.info('argsort ...')
                sorted_indices = np.argsort(children_depths)
                logging.info('find depth changes ...')
                depth_changes = []
                for idx, sort_idx in enumerate(sorted_indices):
                    current_depth = children_depths[sort_idx]
                    if idx == len(sorted_indices) - 1 or current_depth != children_depths[sorted_indices[idx + 1]]:
                        logging.info('new depth: ' + str(current_depth) + ' ends before index pos: ' + str(idx + 1))
                        depth_changes.append((idx + 1, current_depth))
                prev_end = 0
                for (end, current_depth) in depth_changes:
                    size = end - prev_end
                    logging.info('size: ' + str(size))
                    current_indices = np.zeros(shape=(size, 2), dtype=int)
                    for idx in range(size):
                        current_indices[idx] = current_idx_tuples[sorted_indices[prev_end + idx]][:2]
                    logging.info('dump children indices with distance (path length from root to child): ' + str(
                        current_depth) + ' ...')
                    current_indices.dump(out_filename + '.children.depth' + str(current_depth) + '.batch' + str(offset))
                    prev_end = end
                # remove processed batch file
                os.remove(out_filename + '.children.batch' + str(offset))
                # not yet used
                # seq_data = np.load(out_filename + '.parent.batch' + str(offset))
                # child_idx_offset += len(seq_data)
                ##


# unused
def collected_shuffled_child_indices(out_filename, max_depth, dump=False):
    logging.info('create shuffled child indices ...')
    # children_depth_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.children.depth*')
    collected_child_indices = np.zeros(shape=(0, 3), dtype=np.int32)
    for current_depth in range(1, max_depth + 1):
        if not os.path.isfile(out_filename + '.children.depth' + str(current_depth) + '.collected'):
            # logging.info('load: ' + out_filename + '.children.depth' + str(current_depth))
            current_depth_indices = np.load(out_filename + '.children.depth' + str(current_depth))
            current_depth_indices = np.pad(current_depth_indices, ((0, 0), (0, 1)),
                                           'constant', constant_values=((0, 0), (0, current_depth)))
            collected_child_indices = np.append(collected_child_indices, current_depth_indices, axis=0)
            np.random.shuffle(collected_child_indices)
            if dump:
                # TODO: re-add! (crashes, files to big? --> cpickle size constraint! (2**32 -1))
                collected_child_indices.dump(out_filename + '.children.depth' + str(current_depth) + '.collected')
            logging.info('depth: ' + str(current_depth) + ', collected_size: ' + str(len(collected_child_indices)))
        else:
            collected_child_indices = np.load(out_filename + '.children.depth' + str(current_depth) + '.collected')
    return collected_child_indices


class Corpus(object):
    def __init__(self, filename=None, lexicon=None, sequence_trees=None, vecs=None, types=None, data=None, parents=None):
        if filename:
            self.load(filename)
        else:
            if lexicon:
                self._lexicon = lexicon
            else:
                self._lexicon = lex.Lexicon(vecs=vecs, types=types)

            if sequence_trees:
                self._sequence_trees = sequence_trees
            else:
                self._sequence_trees = sequ_trees.SequenceTrees(data=data, parents=parents)

    def load(self, filename):
        self._lexicon = lex.Lexicon(filename=filename)
        self._sequence_trees = sequ_trees.SequenceTrees(filename=filename)

    def dump(self, filename):
        self._lexicon.dump(filename)
        self._sequence_trees.dump(filename)

    @property
    def lexicon(self):
        return self._lexicon

    @property
    def sequence_trees(self):
        return self._sequence_trees

    @property
    def vecs(self):
        return self._lexicon.vecs

    @property
    def types(self):
        return self._lexicon.types

    @property
    def data(self):
        return self._sequence_trees.data

    @property
    def parents(self):
        return self._sequence_trees.parents
