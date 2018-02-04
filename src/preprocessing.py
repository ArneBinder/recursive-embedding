from __future__ import print_function

import logging

import numpy as np

import constants
import mytools
import sequence_trees

#MARKER_DEP_EDGE = 'DEP/'


# unused
def merge_sentence_data(sen_data, sen_parents, sen_offsets, sen_a):
    root_offset = 0
    result_data = list()
    result_parents = list()
    l = len(sen_data)
    for i in range(l):
        # set root
        if sen_parents[i] == 0:
            root_offset = len(result_data)
        # add main data
        result_data.append(sen_data[i])
        # shift parent indices
        parent_idx = sen_parents[i] + i
        shift = mytools.get_default(sen_offsets, parent_idx - 1, 0) - mytools.get_default(sen_offsets, i - 1, 0)
        # add (shifted) main parent
        result_parents.append(sen_parents[i] + shift)
        # insert additional data
        a_data, a_parents = sen_a[i]
        if len(a_data) > 0:
            result_data.extend(a_data)
            result_parents.extend(a_parents)

    return result_data, result_parents, [root_offset]


def concat_roots(data, parents, root_offsets, root_parents=None, concat_mode='tree', new_root_id=None):
    if concat_mode is None:
        return data, parents, root_offsets

    if concat_mode == 'aggregate':
        assert new_root_id is not None, 'concat_mode=aggregate requires an aggregation root, but new_root_id is None.'
        root_offsets.append(len(data))
        parents.append(0)
        data.append(new_root_id)
        for i in range(len(root_offsets) - 1):
            parents[root_offsets[i]] = len(parents) - root_offsets[i] - 1
        return data, parents, [len(parents) - 1]
    if concat_mode == 'tree':
        assert root_parents and len(root_parents) == len(root_offsets), \
            'length of root_parents=%i does not match length of root_offsets=%i' % (len(root_parents), len(root_offsets))
        new_roots = []
        for i in range(len(root_parents)):
            if root_parents[i] != 0:
                parents[root_offsets[i]] = root_offsets[i + root_parents[i]] - root_offsets[i]
            else:
                new_roots.append(root_offsets[i])
        #return data, parents, new_roots
    # connect roots consecutively
    elif concat_mode == 'sequence':
        for i in range(len(root_offsets) - 1):
            parents[root_offsets[i]] = root_offsets[i + 1] - root_offsets[i]
        #return data, parents, [root_offsets[-1]]
        new_roots = [root_offsets[-1]]
    else:
        raise NameError('unknown concat_mode: ' + concat_mode)

    # concat_mode is 'tree' or 'sequence'
    if new_root_id is not None:
        parents.append(0)
        data.append(new_root_id)
        parents[new_roots[0]] = len(parents) - new_roots[0] - 1
        new_roots = [len(data)-1]
    return data, parents, new_roots


# embeddings for:
# word
def process_sentence1(sentence, parsed_data, data_maps, dict_unknown=None,
                      concat_mode=constants.default_inner_concat_mode):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):
        token = parsed_data[i]
        # add word embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING]
                                         + constants.SEPARATOR + token.orth_, dict_unknown))

        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # set as root
        sen_parents.append(0)

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)

    return sen_data, sen_parents, root_offsets


# embeddings for:
# word, word embedding
def process_sentence2(sentence, parsed_data, data_maps, dict_unknown=None,
                      concat_mode=constants.default_inner_concat_mode):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):
        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING]
                                         + constants.SEPARATOR + token.orth_, dict_unknown))
        sen_parents.append(0)
        # add word embedding embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING], dict_unknown))
        sen_parents.append(-1)

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# DEPRECATED
# embeddings for:
# word, edge
def process_sentence3_dep(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):
        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(mytools.getOrAdd(data_maps, token.orth_, dict_unknown))
        sen_parents.append(0)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(data_maps, token.dep_, dict_unknown))
        sen_parents.append(-1)

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# word, edge (marked)
def process_sentence3(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):
        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING]
                                         + constants.SEPARATOR + token.orth_, dict_unknown))
        sen_parents.append(0)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.DEPENDENCY_EMBEDDING]
                                         + constants.SEPARATOR + token.dep_, dict_unknown))
        sen_parents.append(-1)

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# word, word embedding, edge, edge embedding
def process_sentence4(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):
        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING]
                                         + constants.SEPARATOR + token.orth_, dict_unknown))
        sen_parents.append(0)
        # add word embedding embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING], dict_unknown))
        sen_parents.append(-1)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.DEPENDENCY_EMBEDDING]
                                         + constants.SEPARATOR + token.dep_, dict_unknown))
        sen_parents.append(-2)
        # add edge type embedding embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.DEPENDENCY_EMBEDDING], dict_unknown))
        sen_parents.append(-1)

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, edges, entity type (if !=0)
def process_sentence5(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING]
                                         + constants.SEPARATOR + token.orth_, dict_unknown))
        sen_parents.append(0)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.DEPENDENCY_EMBEDDING]
                                         + constants.SEPARATOR + token.dep_, dict_unknown))
        sen_parents.append(-1)

        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.ENTITY_EMBEDDING]
                                             + constants.SEPARATOR + token.ent_type_, dict_unknown))
            sen_parents.append(-2)

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, word type, edges, edge type, entity type (if !=0), entity type type
def process_sentence6(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING]
                                         + constants.SEPARATOR + token.orth_, dict_unknown))
        sen_parents.append(0)

        # add word type type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING], dict_unknown))
        sen_parents.append(-1)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.DEPENDENCY_EMBEDDING]
                                         + constants.SEPARATOR + token.dep_, dict_unknown))
        sen_parents.append(-2)
        # add edge type type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.DEPENDENCY_EMBEDDING], dict_unknown))
        sen_parents.append(-1)

        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.ENTITY_EMBEDDING]
                                             + constants.SEPARATOR + token.ent_type_, dict_unknown))
            sen_parents.append(-2)
            sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.ENTITY_EMBEDDING], dict_unknown))
            sen_parents.append(-1)

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, edges, entity type (if !=0),
# lemma (if different), pos-tag
def process_sentence7(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offset = len(sen_parents)
        root_offsets.append(root_offset)
        # add word embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING]
                                         + constants.SEPARATOR + token.orth_, dict_unknown))
        sen_parents.append(0)

        # add edge type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.DEPENDENCY_EMBEDDING]
                                         + constants.SEPARATOR + token.dep_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add pos-tag type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.POS_EMBEDDING]
                                         + constants.SEPARATOR + token.tag_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))

        # add entity type embedding
        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.ENTITY_EMBEDDING]
                                             + constants.SEPARATOR + token.ent_type_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))
        # add lemma type embedding
        if token.lemma != token.orth:
            sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEMMA_EMBEDDING]
                                             + constants.SEPARATOR + token.lemma_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, word type, edges, edge type, entity type (if !=0), entity type type,
# lemma (if different), lemma type, pos-tag, pos-tag type
def process_sentence8(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offset = len(sen_parents)
        root_offsets.append(root_offset)
        # add word embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING]
                                         + constants.SEPARATOR + token.orth_, dict_unknown))
        sen_parents.append(0)

        # add word type type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING], dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.DEPENDENCY_EMBEDDING]
                                         + constants.SEPARATOR + token.dep_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add edge type type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.DEPENDENCY_EMBEDDING], dict_unknown))
        sen_parents.append(-1)
        # add pos-tag type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEMMA_EMBEDDING]
                                         + constants.SEPARATOR + token.tag_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add pos-tag type type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.POS_EMBEDDING], dict_unknown))
        sen_parents.append(-1)

        # add entity type embedding
        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.ENTITY_EMBEDDING]
                                             + constants.SEPARATOR + token.ent_type_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))
            # add entity type type embedding
            sen_data.append(
                mytools.getOrAdd(data_maps, constants.vocab_manual[constants.ENTITY_EMBEDDING], dict_unknown))
            sen_parents.append(-1)
        # add lemma type embedding
        if token.lemma != token.orth:
            sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEMMA_EMBEDDING]
                                             + constants.SEPARATOR + token.lemma_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))
            # add lemma type type embedding
            sen_data.append(
                mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEMMA_EMBEDDING], dict_unknown))
            sen_parents.append(-1)

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# word, edge (marked)
def process_sentence10(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):
        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.DEPENDENCY_EMBEDDING]
                                         + constants.SEPARATOR + token.dep_, dict_unknown))
        sen_parents.append(0)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING]
                                         + constants.SEPARATOR + token.orth_, dict_unknown))
        sen_parents.append(-1)

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)

    return sen_data, sen_parents, root_offsets


# embeddings for:
# lemma, pos (filtered!)
def process_sentence9(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    assert concat_mode != 'tree', "concat_mode='tree' is not allowed. Use 'aggregate', 'sequence' or None."

    postag_whitelist = [u'VERB', u'ADJ', u'NOUN']

    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i

        if token.pos_ in postag_whitelist:
            root_parents.append(parent_offset)
            # save root offset
            root_offsets.append(len(sen_parents))
            # add word embedding
            #sen_data.append(mytools.getOrAdd(data_maps, token.lemma_, dict_unknown))
            sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.LEXEME_EMBEDDING]
                                             + constants.SEPARATOR + token.lemma_, dict_unknown))
            sen_parents.append(0)
            # add edge type embedding
            # sen_data.append(mytools.getOrAdd(data_maps, token.pos_, dict_unknown))
            # sen_parents.append(-1)

    #if concat_mode == 'aggregate':
    #    root_offsets.append(len(sen_data))
    #    sen_parents.append(0)
    #    sen_data.append(mytools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    new_root_id = mytools.getOrAdd(data_maps, constants.vocab_manual[constants.SENTENCE_EMBEDDING], dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


def dummy_str_reader():
    yield u'I like RTRC!'


def identity_reader(content):
    yield content


def read_data(reader, sentence_processor, parser, data_maps, reader_args={}, batch_size=1000,
              concat_mode=constants.default_concat_mode, inner_concat_mode=constants.default_inner_concat_mode,
              expand_dict=True, reader_roots=None, reader_roots_args={}, reader_annotations=None):
    # ids (dictionary) of the data points in the dictionary
    seq_data = list()
    # offsets of the parents
    seq_parents = list()

    # init as list containing an empty dummy array with dtype=int16 to allow numpy concatenation even if empty
    #depth_list = [np.ndarray(shape=(0,), dtype=np.int16)]

    assert concat_mode in constants.concat_modes, 'unknown concat_mode="' + concat_mode + '". Please use one of: ' + ', '.join(
        [str(s) for s in constants.concat_modes])
    assert inner_concat_mode in constants.concat_modes, 'unknown inner_concat_mode="' + inner_concat_mode + '". Please use one of: ' + ', '.join(
        [str(s) for s in constants.concat_modes])

    if expand_dict:
        unknown_default = None
    else:
        unknown_default = constants.vocab_manual[constants.UNKNOWN_EMBEDDING]

    if reader_roots is None:
        if 'root_labels' in reader_roots_args:
            root_labels = reader_roots_args['root_labels']
        else:
            root_labels = constants.vocab_manual[constants.AGGREGATOR_EMBEDDING]
        _reader_root = iter(lambda: root_labels, -1)
    else:
        _reader_root = reader_roots(**reader_roots_args)

    if reader_annotations is not None:
        _reader_annotations = reader_annotations()
    else:
        _reader_annotations = iter(lambda: None, -1)

    logging.debug('start read_data ...')
    sen_count = 0
    for parsed_data in parser.pipe(reader(**reader_args), n_threads=4, batch_size=batch_size):
        # prev_root = None
        temp_roots = []
        #start_idx = len(seq_data)
        annotations = _reader_annotations.next()
        for sentence in parsed_data.sents:
            processed_sen = sentence_processor(sentence, parsed_data, data_maps, unknown_default, inner_concat_mode)
            # skip not processed sentences (see process_sentence)
            if processed_sen is None:
                continue

            sen_data, sen_parents, root_offsets = processed_sen

            sen_roots = [offset + len(seq_data) for offset in root_offsets]

            seq_parents += sen_parents
            seq_data += sen_data
            temp_roots += sen_roots

            sen_count += 1

        # add source node(s), if concat_mode is not None:
        #if concat_mode is not None:
        #    root_data_list = _reader_root.next()
        #    if type(root_data_list) != list:
        #        root_data_list = [root_data_list]
        #    root_temp_roots = []
        #    for i, root_data in enumerate(root_data_list):
        #        root_temp_roots.append(len(seq_parents))
        #        seq_parents.append(0)
        #        seq_data.append(mytools.getOrAdd(data_maps, root_data, unknown_default))
        #    seq_parents, new_roots = concat_roots(seq_parents, root_temp_roots, concat_mode='aggregate')
        #    temp_roots.extend(new_roots)

        new_root_id = mytools.getOrAdd(data_maps, _reader_root.next(), unknown_default)
        seq_data, seq_parents, _ = concat_roots(data=seq_data, parents=seq_parents, root_offsets=temp_roots,
                                                concat_mode=concat_mode,  new_root_id=new_root_id)

    logging.debug('sentences read: ' + str(sen_count))
    data = np.array(seq_data)
    parents = np.array(seq_parents)

    return data, parents
