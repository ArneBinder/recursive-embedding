import logging
import os

import tensorflow as tf
import numpy as np

from constants import LOGGING_FORMAT, DTYPE_IDX
from config import Config, FLAGS_FN
from lexicon import Lexicon

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.INFO)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))


def get_parameter_count_from_shapes(shapes, shapes_neg=(), selector_prefix='', selector_suffix='',
                                    selector_suffixes_not=()):
    def valid_name(name):
        return name.endswith(selector_suffix) and name.startswith(selector_prefix) \
               and not any([name.endswith(suf_not) for suf_not in selector_suffixes_not])

    filtered_shapes = {k: shapes[k] for k in shapes if valid_name(k) and k not in shapes_neg}
    count = 0
    for tensor_name in filtered_shapes:
        if len(shapes[tensor_name]) > 0:
            count += reduce((lambda x, y: x * y), shapes[tensor_name])
    return count, filtered_shapes


def log_shapes_info(reader, tree_embedder_prefix='TreeEmbedding/', optimizer_suffixes=('/Adam', '/Adam_1')):
    saved_shapes = reader.get_variable_to_shape_map()

    shapes_rev = {k: saved_shapes[k] for k in saved_shapes if '_reverse_' in k}
    p_count, shapes_train_rev = get_parameter_count_from_shapes(shapes_rev,
                                                                selector_suffix=optimizer_suffixes[0])
    logger.debug('(trainable) reverse parameter count: %i' % p_count)
    logger.debug(shapes_train_rev)
    saved_shapes_wo_rev = {k: saved_shapes[k] for k in saved_shapes if k not in shapes_rev}
    # logger.debug(saved_shapes)
    p_count, shapes_te_trainable = get_parameter_count_from_shapes(saved_shapes_wo_rev,
                                                                   selector_prefix=tree_embedder_prefix,
                                                                   selector_suffix=optimizer_suffixes[0])
    logger.debug('(trainable) tree embedder parameter count: %i' % p_count)
    logger.debug(shapes_te_trainable)
    p_count, shapes_te_total = get_parameter_count_from_shapes(saved_shapes_wo_rev,
                                                               shapes_neg=['/'.join(k.split('/')[:-1]) for k in
                                                                           shapes_te_trainable],
                                                               selector_prefix=tree_embedder_prefix,
                                                               selector_suffixes_not=optimizer_suffixes)
    logger.debug('(not trainable) tree embedder parameter count: %i' % p_count)
    logger.debug(shapes_te_total)
    p_count, shapes_nte_trainable = get_parameter_count_from_shapes(saved_shapes_wo_rev,
                                                                    shapes_neg=shapes_te_trainable.keys(),
                                                                    selector_suffix=optimizer_suffixes[0])
    logger.debug('(trainable) remaining parameter count: %i' % p_count)
    logger.debug(shapes_nte_trainable)
    p_count, shapes_nte_total = get_parameter_count_from_shapes(saved_shapes_wo_rev,
                                                                shapes_neg=['/'.join(k.split('/')[:-1]) for k in
                                                                            shapes_te_trainable] + shapes_te_total.keys() + [
                                                                               '/'.join(k.split('/')[:-1]) for k in
                                                                               shapes_nte_trainable],
                                                                selector_suffixes_not=optimizer_suffixes)
    logger.debug('(not trainable) remaining parameter count: %i' % p_count)
    logger.debug(shapes_nte_total)


def get_lexicon(logdir, train_data_path=None, logdir_pretrained=None, #logdir_continue=None,
                dont_dump=False,
                no_fixed_vecs=False, all_vecs_fixed=False, var_vecs_zero=False, var_vecs_random=False,
                additional_vecs_path=None, vecs_pretrained=None):
    checkpoint_fn = tf.train.latest_checkpoint(logdir)
    #if logdir_continue:
    #    raise NotImplementedError('usage of logdir_continue not implemented')
    #    assert checkpoint_fn is not None, 'could not read checkpoint from logdir: %s' % logdir
    #old_checkpoint_fn = None
    #fine_tune = False
    prev_config = None
    if checkpoint_fn is not None:
        if not checkpoint_fn.startswith(logdir):
            raise ValueError('entry in checkpoint file ("%s") is not located in logdir=%s' % (checkpoint_fn, logdir))
        prev_config = Config(logdir=logdir)
        logger.info('read lex_size from model ...')
        reader = tf.train.NewCheckpointReader(checkpoint_fn)
        log_shapes_info(reader)

        lexicon = Lexicon(filename=os.path.join(logdir, 'model'), checkpoint_reader=reader, add_vocab_manual=True,
                          load_ids_fixed=True)
    else:
        assert train_data_path is not None, 'no checkpoint found and no train_data_path given'
        lexicon = Lexicon(filename=train_data_path, load_ids_fixed=(not no_fixed_vecs), add_vocab_manual=True)

        if logdir_pretrained:
            prev_config = Config(logdir=logdir_pretrained)
            no_fixed_vecs = prev_config.no_fixed_vecs
            additional_vecs_path = prev_config.additional_vecs
            #fine_tune = True

        if lexicon.has_vecs:
            #if not no_fixed_vecs and not all_vecs_fixed:
            #    lexicon.set_to_zero(indices=lexicon.ids_fixed, indices_as_blacklist=True)

            if additional_vecs_path:
                logger.info('add embedding vecs from: %s' % additional_vecs_path)
                # ATTENTION: add_lex should contain only lower case entries, because self_to_lowercase=True
                add_lex = Lexicon(filename=additional_vecs_path)
                ids_added = lexicon.add_vecs_from_other(add_lex, self_to_lowercase=True)
                #ids_added_not = [i for i in range(len(lexicon)) if i not in ids_added]
                # remove ids_added_not from lexicon.ids_fixed
                mask_added = np.zeros(len(lexicon), dtype=bool)
                mask_added[ids_added] = True
                mask_fixed = np.zeros(len(lexicon), dtype=bool)
                mask_fixed[lexicon.ids_fixed] = True
                #lexicon._ids_fixed = np.array([_id for _id in lexicon._ids_fixed if _id not in ids_added_not], dtype=lexicon.ids_fixed.dtype)
                lexicon._ids_fixed = (mask_added & mask_fixed).nonzero()[0]

            #ROOT_idx = lexicon.get_d(vocab_manual[ROOT_EMBEDDING], data_as_hashes=False)
            #IDENTITY_idx = lexicon.get_d(vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=False)
            if logdir_pretrained or vecs_pretrained:
                p = logdir_pretrained or vecs_pretrained
                logger.info('load lexicon from pre-trained model: %s' % p)
                # Check, if flags file is available (because of docker-compose file, logdir_pretrained could be just
                # train path prefix and is therefore not None, but does not point to a valid train dir).
                if os.path.exists(os.path.join(p, FLAGS_FN)):
                    #old_config = Config(logdir=logdir_pretrained)
                    checkpoint_fn = tf.train.latest_checkpoint(p)
                    assert checkpoint_fn is not None, 'No checkpoint file found in logdir_pretrained: ' + p
                    reader_old = tf.train.NewCheckpointReader(checkpoint_fn)
                    log_shapes_info(reader_old)
                    lexicon_old = Lexicon(filename=os.path.join(p, 'model'))
                    lexicon_old.init_vecs(checkpoint_reader=reader_old)
                    logger.debug('merge old lexicon into new one...')
                    lexicon.merge(lexicon_old, add_entries=True, replace_vecs=True)
                else:
                    logger.warning('logdir_pretrained is not None (%s), but no flags file found. Do not try to load '
                                   'from logdir_pretrained.' % logdir_pretrained)

            if all_vecs_fixed:
                # zero (UNKNOWN) has to remain trainable because of double assignment bug (see TreeEmbedding.embed and
                # Lexicon.transform_idx)
                lexicon.init_ids_fixed(ids_fixed=np.arange(len(lexicon) - 1, dtype=DTYPE_IDX) + 1)

            assert not (var_vecs_zero and var_vecs_random), 'use either var_vecs_zero OR (exclusive) var_vecs_random'
            if var_vecs_zero:
                lexicon.set_to_zero(indices=lexicon.ids_fixed, indices_as_blacklist=True)
            elif var_vecs_random:
                lexicon.set_to_random(indices=lexicon.ids_fixed, indices_as_blacklist=True)

            if not dont_dump:
                logger.debug('dump lexicon to: %s ...' % os.path.join(logdir, 'model'))
                lexicon.dump(filename=os.path.join(logdir, 'model'), strings_only=True)
                assert lexicon.is_filled, 'lexicon: not all vecs for all types are set (len(types): %i, len(vecs): %i)' % \
                                          (len(lexicon), len(lexicon.vecs))
        else:
            logger.warning('NO VECS AVAILABLE FOR LEXICON')



    logger.info('lexicon size: %i' % len(lexicon))
    #logger.debug('IDENTITY_idx: %i' % IDENTITY_idx)
    #logger.debug('ROOT_idx: %i' % ROOT_idx)
    return lexicon, checkpoint_fn, prev_config#, fine_tune