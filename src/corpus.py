
from lexicon import Lexicon

import preprocessing


def process_records(reader, parser, batch_size=1000, n_threads=4):
    """

    :param reader:
    :return: forest object in hash version
    """
    lexicon = Lexicon()

    forest = lexicon.read_data(reader=reader, sentence_processor=preprocessing.process_sentence1,
                               parser=parser, batch_size=batch_size, concat_mode='sequence',
                               inner_concat_mode='tree', expand_dict=True, as_tuples=True,
                               return_hashes=True, n_threads=n_threads)

    # TODO: remove root_ids from lexicon and put them into lexicon_roots
    lexicon_roots = Lexicon()


    # parse_context_batch(nif_context_datas=nif_context_datas, failed=failed, nlp=_nlp,
    #                    filename=fn, t_query=t_query)
    #forest, failed_parse = create_contexts_forest(nif_context_datas, lexicon=lexicon, nlp=_nlp,
    #                                              n_threads=num_threads_parse_pipe,
    #                                              batch_size=batch_size_parse)
    #failed.extend(failed_parse)

    #save_current_forest(forest=forest,
    #                    failed=failed,
    #                    # resource_hashes_failed=np.array(resource_hashes_failed, dtype=DTYPE_HASH),
    #                    lexicon=lexicon, filename=fn, t_parse=datetime.now() - t_start,
    #                    t_query=t_query)

    return forest, lexicon, lexicon_roots