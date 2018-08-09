from __future__ import print_function

import logging

import numpy as np

import mytools
from constants import DTYPE_HASH, default_concat_mode, default_inner_concat_mode, concat_modes,vocab_manual, \
    TYPE_DEPENDENCY_RELATION, SEPARATOR, TYPE_SENTENCE, TYPE_POS_TAG, \
    TYPE_LEMMA, UNKNOWN_EMBEDDING, AGGREGATOR_EMBEDDING, TYPE_LEXEME, TYPE_NAMED_ENTITY

PREFIX_LEX = TYPE_LEXEME + SEPARATOR
PREFIX_DEP = TYPE_DEPENDENCY_RELATION + SEPARATOR
PREFIX_ENT = TYPE_NAMED_ENTITY + SEPARATOR
PREFIX_POS = TYPE_POS_TAG + SEPARATOR
PREFIX_LEM = TYPE_LEMMA + SEPARATOR


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


def get_annotations_for_token(annotations, token):
    # add annotations after respective token (if its parent isn't a match, too)
    j = 0
    data = []
    parents = []
    for _ in range(len(annotations)):
        annot = annotations[j]
        if ((token.idx <= annot[0] <= token.idx + len(token)) or (token.idx <= annot[1] <= token.idx + len(token))) \
                and (token.head == token or not ((token.head.idx <= annot[0] <= token.head.idx + len(token.head))
                                                 or (token.head.idx <= annot[1] <= token.head.idx + len(token.head)))):
            data.extend(annot[2])
            annot[3][0] = -len(parents) - 1
            parents.extend(annot[3])
            del annotations[j]
    return data, parents


#def as_lexeme(s):
#    return PREFIX_LEX + s


def without_prefix(s, prefix=PREFIX_LEX):
    if s.startswith(prefix):
        return s[len(prefix):]
    else:
        return None


# embeddings for:
# word
def process_sentence1(sentence, parsed_data, strings, dict_unknown=None,
                      concat_mode=default_inner_concat_mode, annotations=[], **kwargs):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):
        token = parsed_data[i]
        # add word embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.orth_, dict_unknown))

        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # set as root
        sen_parents.append(0)

        # check and append annotations, eventually
        annot_data, annot_parents = get_annotations_for_token(annotations=annotations, token=token)
        sen_data.extend([mytools.getOrAdd(strings, s, dict_unknown) for s in annot_data])
        sen_parents.extend(annot_parents)

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)

    return sen_data, sen_parents, root_offsets


# embeddings for:
# word, word embedding
def process_sentence2(sentence, parsed_data, strings, dict_unknown=None,
                      concat_mode=default_inner_concat_mode, **kwargs):
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
        sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.orth_, dict_unknown))
        sen_parents.append(0)
        # add word embedding embedding
        sen_data.append(mytools.getOrAdd(strings, TYPE_LEXEME, dict_unknown))
        sen_parents.append(-1)

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# DEPRECATED
# embeddings for:
# word, edge
def process_sentence3_dep(sentence, parsed_data, strings, dict_unknown=None, concat_mode=None, **kwargs):
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
        sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.orth_, dict_unknown))
        sen_parents.append(0)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_DEP + token.dep_, dict_unknown))
        sen_parents.append(-1)

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# word, edge (marked)
def process_sentence3(sentence, parsed_data, strings, dict_unknown=None, concat_mode=None, **kwargs):
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
        sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.orth_, dict_unknown))
        sen_parents.append(0)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_DEP + token.dep_, dict_unknown))
        sen_parents.append(-1)

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# word, word embedding, edge, edge embedding
def process_sentence4(sentence, parsed_data, strings, dict_unknown=None, concat_mode=None, **kwargs):
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
        sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.orth_, dict_unknown))
        sen_parents.append(0)
        # add word embedding embedding
        sen_data.append(mytools.getOrAdd(strings, TYPE_LEXEME, dict_unknown))
        sen_parents.append(-1)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_DEP + token.dep_, dict_unknown))
        sen_parents.append(-2)
        # add edge type embedding embedding
        sen_data.append(mytools.getOrAdd(strings, TYPE_DEPENDENCY_RELATION, dict_unknown))
        sen_parents.append(-1)

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, edges, entity type (if !=0)
def process_sentence5(sentence, parsed_data, strings, dict_unknown=None, concat_mode=None, **kwargs):
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
        sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.orth_, dict_unknown))
        sen_parents.append(0)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_DEP + token.dep_, dict_unknown))
        sen_parents.append(-1)

        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(mytools.getOrAdd(strings, PREFIX_ENT + token.ent_type_, dict_unknown))
            sen_parents.append(-2)

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, word type, edges, edge type, entity type (if !=0), entity type type
def process_sentence6(sentence, parsed_data, strings, dict_unknown=None, concat_mode=None, **kwargs):
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
        sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.orth_, dict_unknown))
        sen_parents.append(0)

        # add word type type embedding
        sen_data.append(mytools.getOrAdd(strings, TYPE_LEXEME, dict_unknown))
        sen_parents.append(-1)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_DEP + token.dep_, dict_unknown))
        sen_parents.append(-2)
        # add edge type type embedding
        sen_data.append(mytools.getOrAdd(strings, TYPE_DEPENDENCY_RELATION, dict_unknown))
        sen_parents.append(-1)

        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(mytools.getOrAdd(strings, PREFIX_ENT + token.ent_type_, dict_unknown))
            sen_parents.append(-2)
            sen_data.append(mytools.getOrAdd(strings, TYPE_NAMED_ENTITY, dict_unknown))
            sen_parents.append(-1)

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, edges, entity type (if !=0),
# lemma (if different), pos-tag
def process_sentence7(sentence, parsed_data, strings, dict_unknown=None, concat_mode=None, **kwargs):
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
        sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.orth_, dict_unknown))
        sen_parents.append(0)

        # add edge type embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_DEP + token.dep_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add pos-tag type embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_POS + token.tag_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))

        # add entity type embedding
        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(mytools.getOrAdd(strings, PREFIX_ENT + token.ent_type_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))
        # add lemma type embedding
        if token.lemma != token.orth:
            sen_data.append(mytools.getOrAdd(strings, PREFIX_LEM + token.lemma_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, word type, edges, edge type, entity type (if !=0), entity type type,
# lemma (if different), lemma type, pos-tag, pos-tag type
def process_sentence8(sentence, parsed_data, strings, dict_unknown=None, concat_mode=None, **kwargs):
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
        sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.orth_, dict_unknown))
        sen_parents.append(0)

        # add word type type embedding
        sen_data.append(mytools.getOrAdd(strings, TYPE_LEXEME, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_DEP + token.dep_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add edge type type embedding
        sen_data.append(mytools.getOrAdd(strings, vocab_manual[DEPENDENCY_EMBEDDING], dict_unknown))
        sen_parents.append(-1)
        # add pos-tag type embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_POS + token.tag_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add pos-tag type type embedding
        sen_data.append(mytools.getOrAdd(strings, TYPE_POS_TAG, dict_unknown))
        sen_parents.append(-1)

        # add entity type embedding
        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(mytools.getOrAdd(strings, PREFIX_ENT + token.ent_type_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))
            # add entity type type embedding
            sen_data.append(
                mytools.getOrAdd(strings, vocab_manual[ENTITY_EMBEDDING], dict_unknown))
            sen_parents.append(-1)
        # add lemma type embedding
        if token.lemma != token.orth:
            sen_data.append(mytools.getOrAdd(strings, PREFIX_LEM + token.lemma_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))
            # add lemma type type embedding
            sen_data.append(
                mytools.getOrAdd(strings, vocab_manual[LEMMA_EMBEDDING], dict_unknown))
            sen_parents.append(-1)

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


# embeddings for:
# word, edge (marked)
def process_sentence10(sentence, parsed_data, strings, dict_unknown=None, concat_mode=None, **kwargs):
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
        sen_data.append(mytools.getOrAdd(strings, PREFIX_DEP + token.dep_, dict_unknown))
        sen_parents.append(0)
        # add edge type embedding
        sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.orth_, dict_unknown))
        sen_parents.append(-1)

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)

    return sen_data, sen_parents, root_offsets


# embeddings for:
# lemma, pos (filtered!)
def process_sentence9(sentence, parsed_data, strings, dict_unknown=None, concat_mode=None, **kwargs):
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
            sen_data.append(mytools.getOrAdd(strings, PREFIX_LEX + token.lemma_, dict_unknown))
            sen_parents.append(0)
            # add edge type embedding
            # sen_data.append(mytools.getOrAdd(data_maps, token.pos_, dict_unknown))
            # sen_parents.append(-1)

    new_root_id = mytools.getOrAdd(strings, TYPE_SENTENCE, dict_unknown)
    sen_data, sen_parents, root_offsets = concat_roots(sen_data, sen_parents, root_offsets, root_parents, concat_mode,
                                                       new_root_id=new_root_id)
    return sen_data, sen_parents, root_offsets


def dummy_str_reader():
    yield u'I like RTRC!'


def identity_reader(content):
    yield content


def read_data(reader, sentence_processor, parser, strings, reader_args={}, batch_size=1000,
              concat_mode=default_concat_mode, inner_concat_mode=default_inner_concat_mode,
              expand_dict=True, reader_roots=None, reader_roots_args={}, as_tuples=False, n_threads=4):
    """

    :param reader:
    :param sentence_processor:
    :param parser:
    :param data_maps:
    :param reader_args:
    :param batch_size:
    :param concat_mode:
    :param inner_concat_mode:
    :param expand_dict:
    :param reader_roots:
    :param reader_roots_args:
    :param reader_annotations: a generator yielding tuples of the format (char_begin, char_end, data_array, parents_array),
                               where the first entry of parents_array does not matter (for convenience, the arrays have
                               the same size)
    :return:
    """
    # ids (dictionary) of the data points in the dictionary
    seq_data = list()
    # offsets of the parents
    seq_parents = list()

    # init as list containing an empty dummy array with dtype=int16 to allow numpy concatenation even if empty
    #depth_list = [np.ndarray(shape=(0,), dtype=np.int16)]

    assert concat_mode in concat_modes, 'unknown concat_mode="%s". Please use one of: %s' \
                                                  % (concat_mode, ', '.join([str(s) for s in concat_modes]))
    assert inner_concat_mode in concat_modes, 'unknown inner_concat_mode="%s". Please use one of: %s' \
                                                        % (inner_concat_mode, ', '.join([str(s) for s in concat_modes]))

    if expand_dict:
        idx_unknown = None
    else:
        assert vocab_manual[UNKNOWN_EMBEDDING] in strings, '"%s" is not in StringStore.' \
                                                                               % UNKNOWN_EMBEDDING
        idx_unknown = strings[vocab_manual[UNKNOWN_EMBEDDING]]

    if reader_roots is None:
        if 'root_label' in reader_roots_args:
            root_label = reader_roots_args['root_label']
        else:
            root_label = vocab_manual[AGGREGATOR_EMBEDDING]
        _reader_root = iter(lambda: root_label, -1)
    else:
        _reader_root = reader_roots(**reader_roots_args)

    logging.debug('start read_data ...')
    sen_count = 0
    annotations = None
    last_prepend_offset = None
    parent_root_pos = None
    for parsed_data in parser.pipe(reader(**reader_args), n_threads=n_threads, batch_size=batch_size, as_tuples=as_tuples):
        if parsed_data is None:
            continue

        if as_tuples:
            parsed_data, context = parsed_data
            prepend = context.get('prepend_tree', None)
            if prepend is not None:
                last_prepend_offset = len(seq_data)
                seq_data.extend([mytools.getOrAdd(strings, s) for s in prepend[0]])
                seq_parents.extend(prepend[1])
            annotations = context.get('annotations', None)
            root_type = context.get('root_type', _reader_root.next())
            parent_prepend_offset = context.get('parent_prepend_offset', None)
            parent_root_pos = last_prepend_offset + parent_prepend_offset
        else:
            root_type = _reader_root.next()
        temp_roots = []
        for sentence in parsed_data.sents:
            sent_start = sentence.start_char
            sent_end = sentence.end_char
            sent_annots = []
            if annotations is not None:
                for annot in annotations:
                    if (sent_start <= annot[0] <= sent_end) or (sent_start <= annot[1] <= sent_end):
                        sent_annots.append(annot)
            processed_sen = sentence_processor(sentence, parsed_data, strings, idx_unknown, inner_concat_mode,
                                               annotations=sent_annots)
            # skip not processed sentences (see process_sentence)
            if processed_sen is None:
                continue

            sen_data, sen_parents, root_offsets = processed_sen

            sen_roots = [offset + len(seq_data) for offset in root_offsets]

            seq_parents += sen_parents
            seq_data += sen_data
            temp_roots += sen_roots

            sen_count += 1

        new_root_id = mytools.getOrAdd(strings, root_type, idx_unknown)
        seq_data, seq_parents, new_roots = concat_roots(data=seq_data, parents=seq_parents, root_offsets=temp_roots,
                                                        concat_mode=concat_mode,  new_root_id=new_root_id)
        # add links to prepended data/parents
        if parent_root_pos is not None:
            for new_root in new_roots:
                seq_parents[new_root] = parent_root_pos - new_root

    logging.debug('sentences read: ' + str(sen_count))
    data = np.array(seq_data, dtype=DTYPE_HASH)
    parents = np.array(seq_parents)

    return data, parents
