import logging
import os
import spacy

import plac
import numpy as np

from constants import LOGGING_FORMAT, TYPE_CONTEXT, SEPARATOR, TYPE_PARAGRAPH, TYPE_RELATION, TYPE_NAMED_ENTITY, \
    TYPE_DEPENDENCY_RELATION, TYPE_ARTIFICIAL, OFFSET_ID, OFFSET_RELATION_ROOT, DTYPE_IDX, DTYPE_OFFSET, DTYPE_HASH, \
    TYPE_RELATION_TYPE, TYPE_RELATION_DIRECTION
from mytools import make_parent_dir
from corpus import process_records, merge_batches, create_index_files, DIR_BATCHES, save_class_ids
import preprocessing
from preprocessing import KEY_ANNOTATIONS
from sequence_trees import Forest

logger = logging.getLogger('corpus_semeval2010task8')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)

TYPE_SEMEVAL2010TASK8_ID = u'http://semeval2.fbk.eu/semeval2.php/task8'
TYPE_E1 = TYPE_NAMED_ENTITY + SEPARATOR + u'E1'
TYPE_E2 = TYPE_NAMED_ENTITY + SEPARATOR + u'E2'

KEY_TEXT = 'text'

DUMMY_RECORD_ORIG = '1	"The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."\nComponent-Whole(e2,e1)\nComment: Not a collection: there is structure here, organisation.'

DUMMY_RECORD = {
    u'http://semeval2.fbk.eu/semeval2.php/task8': u'1',
    u'RELATION': u'Component-Whole(e2,e1)',
    KEY_ANNOTATIONS: [(73, 86, [u'http://purl.org/olia/olia.owl#NamedEntity/e1'], [0]),
                    (98, 106, [u'http://purl.org/olia/olia.owl#NamedEntity/e2'], [0])],
    KEY_TEXT: u'The system as described above has its greatest application in an arrayed configuration of antenna elements.'
}


def reader(records, keys_text=(KEY_TEXT,), root_string=TYPE_SEMEVAL2010TASK8_ID,
           keys_meta=(TYPE_RELATION,), key_id=TYPE_SEMEVAL2010TASK8_ID,
           root_text_string=TYPE_CONTEXT):
    """

    :param records: dicts containing the textual data and optional meta data
    :param keys_text:
    :param root_string:
    :param keys_meta:
    :param key_id:
    :param root_text_string:
    :return:
    """
    count_finished = 0
    count_discarded = 0
    for record in records:
        try:
            record_data = []
            for i, key_text in enumerate(keys_text):
                prepend_data_strings = [root_string]
                prepend_parents = [0]
                if key_id is not None:
                    prepend_data_strings.append(key_id + SEPARATOR + record[key_id])# + SEPARATOR + str(i))
                    prepend_parents.append(-1)

                prepend_data_strings.append(root_text_string)
                prepend_parents.append(-len(prepend_parents))
                text_root_offset = len(prepend_parents) - 1

                for k_meta in keys_meta:
                    # add key
                    prepend_data_strings.append(k_meta)
                    prepend_parents.append(-len(prepend_parents))
                    # add value(s)
                    # ATTENTION: assumed to be string(s)!
                    v_meta = record[k_meta]
                    if not isinstance(v_meta, list):
                        v_meta = [v_meta]
                    # replace spaces by underscores
                    prepend_data_strings.extend([k_meta + SEPARATOR + v.replace(' ', '_') for v in v_meta])
                    prepend_parents.extend([-j - 1 for j in range(len(v_meta))])

                #prepend_data_strings.append(TYPE_REF_TUPLE)
                #prepend_parents.append(-len(prepend_parents))

                #prepend_data_strings.append(key_id + SEPARATOR + record[key_id] + SEPARATOR + str(1-i))
                #prepend_parents.append(-1)

                prepend_data_strings.append(TYPE_PARAGRAPH)
                prepend_parents.append(text_root_offset - len(prepend_parents))
                text_root_offset = len(prepend_parents) - 1

                prepend = (prepend_data_strings, prepend_parents)

                record_data.append((record[key_text], {'root_type': TYPE_PARAGRAPH, 'prepend_tree': prepend,
                                                       'parent_prepend_offset': text_root_offset,
                                                       KEY_ANNOTATIONS: record[KEY_ANNOTATIONS]}))


            # has to be done in the end because the whole record should be discarded at once if an exception is raised
            for d in record_data:
                yield d
            count_finished += 1
        except Warning as e:
            count_discarded += 1
            #logger.debug('failed to process record (%s): %s' % (e.message, str(record)))
            pass
        except Exception as e:
            count_discarded += 1
            logger.warning('failed to process record (%s): %s' % (e.message, str(record)))


@plac.annotations(
    out_base_name=('corpora output base file name', 'option', 'o', str),
    sentence_processor=('sentence processor', 'option', 'p', str),
)
def parse_dummy(out_base_name, sentence_processor=None):
    print(out_base_name)
    make_parent_dir(out_base_name)
    if sentence_processor is not None and sentence_processor.strip() != '':
        _sentence_processor = getattr(preprocessing, sentence_processor.strip())
    else:
        _sentence_processor = preprocessing.process_sentence1
    parser = spacy.load('en')
    process_records(records=[DUMMY_RECORD], out_base_name=out_base_name, record_reader=reader, parser=parser,
                    sentence_processor=_sentence_processor, concat_mode=None)


def extract_positions(text, tags):
    res = []
    for t in tags:
        res.append(text.find(t))
        text = text.replace(t, '', 1)
    return text, res


def read_file(file_name, annotations):
    with open(file_name) as f:
        lines = f.readlines()
    n = 0
    for i in range(0, len(lines) - 3, 4):
        id_w_text = lines[i].split('\t')
        text = id_w_text[1].strip()[1:-1]
        text, positions = extract_positions(text, ('<e1>', '</e1>', '<e2>', '</e2>'))

        record = {
            TYPE_SEMEVAL2010TASK8_ID: unicode(id_w_text[0]),
            KEY_TEXT: unicode(text),
            TYPE_RELATION: unicode(lines[i + 1].strip()),
            # omit comment field

            KEY_ANNOTATIONS: [(positions[0], positions[1], annotations[0], annotations[2]),
                              (positions[2], positions[3], annotations[1], annotations[2])],
        }
        yield record
        n += 1

    logger.info('read %i records from %s' % (n, file_name))

def iterate_relation_trees(forest):
    e1_hash = forest.lexicon.get_d(s=TYPE_E1, data_as_hashes=True)
    e2_hash = forest.lexicon.get_d(s=TYPE_E2, data_as_hashes=True)
    nlp_root_hash = forest.lexicon.get_d(s=TYPE_PARAGRAPH, data_as_hashes=True)
    e1_indices = np.argwhere(forest.data == e1_hash).flatten()
    e2_indices = np.argwhere(forest.data == e2_hash).flatten()
    nlp_root_indices = np.argwhere(forest.data == nlp_root_hash).flatten()

    assert len(e1_indices) == len(e2_indices) == len(nlp_root_indices), \
        'number of indices does not match: len(e1_indices)=%s, len(e2_indices)=%i, len(nlp_root_indices)=%i' \
        % (len(e1_indices), len(e2_indices), len(nlp_root_indices))
    for i, nlp_root in enumerate(nlp_root_indices):
        try:
            nlp_root_data = forest.data[nlp_root]
            nlp_root_parent = forest.parents[nlp_root]

            raise NotImplementedError('forest.get_path_indices does not work with graph structure anymore')
            #p1 = forest.get_path_indices(e1_indices[i], nlp_root)
            #p2 = forest.get_path_indices(e2_indices[i], nlp_root)
            j = 0
            for _ in range(min(len(p1), len(p2))):
                j += 1
                if p1[-j] != p2[-j]:
                    j -= 1
                    break

            root_common = p1[-j]
            yield nlp_root, nlp_root_data, nlp_root_parent, root_common

        except AssertionError as e:
            logger.error('failed to adjust tree with ID: %s'
                         % forest.lexicon.get_s(forest.data[forest.roots[i] + OFFSET_ID],
                                                data_as_hashes=forest.data_as_hashes))
            raise e


def move_relation_annotation_to_annotation_subtree(forest):

    new_data = []
    new_parents = []

    for i, (nlp_root, nlp_root_data, nlp_root_parent, root_common) in enumerate(iterate_relation_trees(forest)):
        relation_root = forest.roots[i] + OFFSET_RELATION_ROOT
        relation_data = forest.data[relation_root + 1]
        subtree = forest.get_slice(root=forest.roots[i], root_exclude=relation_root)
        tree_size = (forest.roots[i+1] if i+1 < len(forest.roots) else len(forest)) - forest.roots[i]
        root_common_offset = forest.roots[i] + tree_size - root_common
        # split into relation type and direction
        relation_str = forest.lexicon.get_s(relation_data, data_as_hashes=True)
        relation_str_split = relation_str[len(TYPE_RELATION + SEPARATOR):].split("(")
        relation_str_type = TYPE_RELATION_TYPE + SEPARATOR + relation_str_split[0]
        forest.lexicon.add(relation_str_type)
        relation_data_type = forest.lexicon.get_d(relation_str_type, data_as_hashes=True)
        data_append = [relation_data_type]
        parents_append = [-root_common_offset]
        # relation-type "other" has not direction
        if len(relation_str_split) > 1:
            relation_str_direction = TYPE_RELATION_DIRECTION + SEPARATOR + relation_str_split[1][:-1]
            forest.lexicon.add(relation_str_direction)
            relation_data_direction = forest.lexicon.get_d(relation_str_direction, data_as_hashes=True)
            data_append.append(relation_data_direction)
            parents_append.append(-1)
        new_data.extend([subtree.data, np.array(data_append, dtype=DTYPE_HASH)])
        new_parents.extend([subtree.parents, np.array(parents_append, dtype=DTYPE_OFFSET)])

    new_forest = Forest(data=np.concatenate(new_data), parents=np.concatenate(new_parents),
                        data_as_hashes=forest.data_as_hashes, lexicon=forest.lexicon,
                        lexicon_roots=forest.lexicon_roots)

    return new_forest


def extract_relation_subtree(forest):

    new_data = []
    new_parents = []

    for i, (nlp_root, nlp_root_data, nlp_root_parent, root_common) in enumerate(iterate_relation_trees(forest)):
        current_tree = forest.get_slice(root=forest.roots[i], root_exclude=nlp_root)
        subtree = forest.get_slice(root=root_common)

        subtree_root = subtree.roots[0]
        subtree.parents[subtree_root] = -(subtree_root + 1)

        new_data.extend([current_tree.data, [nlp_root_data], subtree.data])
        new_parents.extend([current_tree.parents, [nlp_root_parent], subtree.parents])

    new_forest = Forest(data=np.concatenate(new_data), parents=np.concatenate(new_parents),
                        data_as_hashes=forest.data_as_hashes, lexicon=forest.lexicon, lexicon_roots=forest.lexicon_roots)

    return new_forest


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
    sentence_processor=('sentence processor', 'option', 'p', str),
    n_threads=('number of threads for replacement operations', 'option', 't', int),
    parser_batch_size=('parser batch size', 'option', 'b', int),
    unused='not used parameters'
)
def parse(in_path, out_path, sentence_processor=None, n_threads=4, parser_batch_size=1000, *unused):
    if sentence_processor is not None and sentence_processor.strip() != '':
        _sentence_processor = getattr(preprocessing, sentence_processor.strip())
    else:
        _sentence_processor = preprocessing.process_sentence1

    if _sentence_processor == preprocessing.process_sentence1:
        annots = ([TYPE_E1], [TYPE_E2], [0])
    elif _sentence_processor == preprocessing.process_sentence3:
        annots = ([TYPE_E1, TYPE_DEPENDENCY_RELATION + SEPARATOR + TYPE_ARTIFICIAL],
                  [TYPE_E2, TYPE_DEPENDENCY_RELATION + SEPARATOR + TYPE_ARTIFICIAL],
                  [0, -1])
    elif _sentence_processor == preprocessing.process_sentence10:
        annots = ([TYPE_E1, TYPE_DEPENDENCY_RELATION + SEPARATOR + TYPE_ARTIFICIAL],
                  [TYPE_E2, TYPE_DEPENDENCY_RELATION + SEPARATOR + TYPE_ARTIFICIAL],
                  [0, -1])
    else:
        raise NotImplementedError('not implemented for sentence_processor=%s' % str(sentence_processor))

    file_names = ['SemEval2010_task8_training/TRAIN_FILE.TXT', 'SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT']
    parser = spacy.load('en')
    for fn in file_names:
        logger.info('create forest for %s ...' % fn)
        out_base_name = os.path.join(out_path, DIR_BATCHES, fn.split('/')[0])
        make_parent_dir(out_base_name)
        process_records(records=read_file(os.path.join(in_path, fn), annots), out_base_name=out_base_name,
                        record_reader=reader, parser=parser, sentence_processor=_sentence_processor, concat_mode=None,
                        n_threads=n_threads, batch_size=parser_batch_size)#, adjust_forest_func=move_relation_annotation_to_annotation_subtree)#adjust_forest_func=extract_relation_subtree)
        logger.info('done.')


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE', 'PARSE_DUMMY', 'TEST', 'MERGE', 'CREATE_INDICES', 'ALL']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PARSE_DUMMY':
        plac.call(parse_dummy, args)
    elif mode == 'TEST':
        res = list(plac.call(read_file, args))
    elif mode == 'PARSE':
        plac.call(parse, args)
    elif mode == 'MERGE':
        forest_merged, out_path_merged = plac.call(merge_batches, args)
        relation_ids, relation_strings = forest_merged.lexicon.get_ids_for_prefix(TYPE_RELATION_TYPE)
        save_class_ids(dir_path=out_path_merged, prefix_type=TYPE_RELATION_TYPE, classes_ids=relation_ids,
                       classes_strings=relation_strings)

        relation_ids, relation_strings = forest_merged.lexicon.get_ids_for_prefix(TYPE_RELATION_DIRECTION)
        save_class_ids(dir_path=out_path_merged, prefix_type=TYPE_RELATION_DIRECTION, classes_ids=relation_ids,
                       classes_strings=relation_strings)
    elif mode == 'CREATE_INDICES':
        plac.call(create_index_files, args)
    elif mode == 'ALL':
        plac.call(main, ('PARSE',) + args)
        plac.call(main, ('MERGE',) + args)
        plac.call(main, ('CREATE_INDICES', '--end-root', '2717', '--split-count', '1', '--suffix', 'test') + args)
        plac.call(main, ('CREATE_INDICES', '--start-root', '2717', '--split-count', '4', '--suffix', 'train') + args)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    plac.call(main)
