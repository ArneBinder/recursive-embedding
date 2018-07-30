import copy
import logging
import json
import os
import tensorflow as tf

import mytools

default_config = {'train_data_path': ['DEFINE_string',
                                      # '/media/arne/WIN/Users/Arne/ML/data/corpora/ppdb/process_sentence3_ns1/PPDB_CMaggregate',
                                      # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence2/SICK_CMaggregate',
                                      '/media/arne/WIN/ML/data/corpora/SICK/process_sentence3_marked/SICK_CMaggregate',
                                      # SICK default
                                      # '/media/arne/WIN/Users/Arne/ML/data/corpora/STSBENCH/process_sentence3/STSBENCH_CMaggregate',	# STSbench default
                                      # '/media/arne/WIN/Users/Arne/ML/data/corpora/ANNOPPDB/process_sentence3/ANNOPPDB_CMaggregate',   # ANNOPPDB default
                                      # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence2/SICK_tt_CMsequence_ICMtree',
                                      # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence3/SICK_tt_CMsequence_ICMtree',
                                      # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence4/SICK_tt_CMsequence_ICMtree',
                                      # '/media/arne/WIN/Users/Arne/ML/data/corpora/debate_cluster/process_sentence3/HASAN_CMaggregate',
                                      # '/media/arne/WIN/Users/Arne/ML/data/corpora/debate_cluster/process_sentence3/HASAN_CMaggregate_NEGSAMPLES0',
                                      # '/media/arne/WIN/Users/Arne/ML/data/corpora/debate_cluster/process_sentence3/HASAN_CMsequence_ICMtree_NEGSAMPLES0',
                                      #   '/media/arne/WIN/Users/Arne/ML/data/corpora/debate_cluster/process_sentence3/HASAN_CMsequence_ICMtree_NEGSAMPLES1',
                                      'TF Record file containing the training dataset of sequence tuples.',
                                      'data'],
                  'additional_vecs': ['DEFINE_string',
                                      '',
                                       # '/mnt/DATA/ML/data/embeddings/biomed/converted/biolex_lowercase',
                                      'Path to lexicon containing additional vecs, e.g. biomedical embeddings., that are '
                                      'concatinated with default embeddings.',
                                      None
                                     ],
                  'batch_size': ['DEFINE_integer',
                                 100,
                                 'How many samples to read per batch.',
                                 'bs'],
                  'batch_iter': ['DEFINE_string',
                                    '',
                                    # '50,100,50',
                                    'Batch iterator used in do_epoch. If not set, choose one in do_epoch.',
                                    'biter'
                                    ],
                  # DEPRECATED use only batch_iter
                  #'batch_iter_test': ['DEFINE_string',
                  #               '',
                  #               # '50,100,50',
                  #               'DEPRECATED Test batch iterator used in do_epoch. If not set, use batch_iter.',
                  #               'bitert'
                  #               ],
                  'epochs': ['DEFINE_integer',
                             1000000,
                             'The number of epochs.',
                             None],
                  'dev_file_index': ['DEFINE_integer',
                                     1,
                                     'Which file of the train data files should be used as test data.',
                                     'devfidx'],
                  #'sim_measure': ['DEFINE_string',
                  #                'sim_cosine',
                  #                'similarity measure implementation (tensorflow) from model_fold for similarity score '
                  #                'calculation. Currently implemented:'
                  #                '"sim_cosine" -> cosine'
                  #                '"sim_layer" -> similarity measure similar to the one defined in [Tai, Socher 2015]'
                  #                '"sim_manhattan" -> l1-norm based similarity measure (taken from MaLSTM) [Mueller et al., 2016]',
                  #                'sm'],
                  'tree_embedder': ['DEFINE_string',
                                    'TreeEmbedding_FLAT_AVG',
                                    'TreeEmbedder implementation from model_fold that produces a tensorflow fold block on '
                                    'calling which accepts a sequence tree and produces an embedding. '
                                    'Currently implemented (see model_fold.py):'
                                    '"TreeEmbedding_TREE_LSTM"           -> TreeLSTM'
                                    '"TreeEmbedding_HTU_GRU"             -> Headed Tree Unit, using a GRU for order aware '
                                    '                                       and summation for order unaware composition'
                                    '"TreeEmbedding_FLAT_AVG"            -> Averaging applied to first level children '
                                    '                                       (discarding the root)'
                                    '"TreeEmbedding_FLAT_AVG_2levels"    -> Like TreeEmbedding_FLAT_AVG, but concatenating first'
                                    '                                       second level children (e.g. dep-edge embedding) to '
                                    '                                       the first level children (e.g. token embeddings)'
                                    '"TreeEmbedding_FLAT_LSTM"           -> LSTM applied to first level children (discarding the'
                                    '                                       root)'
                                    '"TreeEmbedding_FLAT_LSTM_2levels"   -> Like TreeEmbedding_FLAT_LSTM, but concatenating '
                                    '                                       first second level children (e.g. dependency-edge '
                                    '                                       type embedding) to the first level children '
                                    '                                       (e.g. token embeddings)',
                                    'te'
                                    ],
                  'leaf_fc_size': ['DEFINE_integer',
                                   0,
                                   # 50,
                                   'If not 0, apply a fully connected layer with this size before composition',
                                   'leaffc'
                                   ],
                  'root_fc_sizes': ['DEFINE_string',
                                    '0',
                                    # '50,100,50',
                                    'Apply fully connected layers with these sizes after composition. '
                                    'Format: String containing a comma separated list of positive integers.',
                                    'rootfc'
                                    ],
                  'fc_sizes': ['DEFINE_string',
                                    '1000',
                                    # '50,100,50',
                                    'Apply fully connected layers with these sizes to the concateneated embeddings '
                                    '(before final softmax). '
                                    'Format: String containing a comma separated list of positive integers.',
                                    'fc'
                                    ],
                  'state_size': ['DEFINE_integer',
                                 50,
                                 'size of the composition layer',
                                 'state'],
                  'learning_rate': ['DEFINE_float',
                                    0.02,
                                    # 'tanh',
                                    'learning rate',
                                    'learn_r'],
                  'optimizer': ['DEFINE_string',
                                'AdadeltaOptimizer',
                                'optimizer',
                                'opt'],
                  'early_stopping_window': ['DEFINE_integer',
                                       50,
                                       'If not 0, stop training when current test loss is smaller then last queued '
                                       'previous losses',
                                       None],
                  'keep_prob': ['DEFINE_float',
                                0.7,
                                'Keep probability for dropout layer',
                                'kp'
                                ],
                  #'auto_restore': ['DEFINE_boolean',
                  #                 False,
                  #                 #   True,
                  #                 'Iff enabled, restore from last checkpoint if no improvements during epoch on test data.',
                  #                 'restore'],
                  'extensions': ['DEFINE_string',
                                 '',
                                 'extensions of the files to use as train/test files (appended to .idx.<NR> file names)',
                                 'xt'],
                  'clipping': ['DEFINE_float',
                               5.0,
                               'global norm threshold for clipping gradients',
                               'clp'],
                  'max_depth': ['DEFINE_integer',
                                15,
                                'maximum depth of embedding trees',
                                'dpth'
                                ],
                  'context': ['DEFINE_integer',
                              0,
                              'maximum depth of added context trees',
                              'cntxt'
                              ],
                  'link_cost_ref': ['DEFINE_integer',
                                    -1,     # negative indicates no link following
                                    'How much following a link will cost. Negative values disable link following',
                                    'lc'
                                    ],
                  'model_type': ['DEFINE_string',
                                 'simtuple',
                                 'type of model',
                                 'mt'],
                  'concat_mode': ['DEFINE_string',
                                  'tree',
                                  'how to concatenate the tokens (tree: use tree structure, sequence: as ordered '
                                  'sequence, aggregate: bag-of-tokens)',
                                  'cm'],
                  'neg_samples': ['DEFINE_integer',
                                  0,
                                  'count of negative samples per tree',
                                  'ns'
                                  ],
                  'neg_samples_test': ['DEFINE_string',
                                  '',
                                  'count of negative samples per tree',
                                  'nst'
                                  ],
                  'cut_indices': ['DEFINE_integer',
                                  None,
                                  'If not None, use only the first cut_indices entries in the data to train the REROOT model',
                                  'nds'],
                  'no_fixed_vecs': ['DEFINE_boolean',
                                    False,
                                    'If enabled, train all embedding vecs (including these in .fix.npy file).',
                                    'nfx'],
                  }

ALLOWED_TYPES = ['string', 'float', 'integer', 'boolean']
MODEL_PARAMETERS = ['additional_vecs', 'tree_embedder', 'leaf_fc_size', 'root_fc_sizes', 'state_size', 'max_depth',
                    'context', 'link_cost_ref', 'concat_mode', 'no_fixed_vecs', 'batch_iter']


class Config(object):
    def __init__(self, logdir_continue=None, logdir_pretrained=None, values=None):
        if logdir_continue is not None:
            logging.info('load flags from logdir: %s', logdir_continue)
            with open(os.path.join(logdir_continue, 'flags.json'), 'r') as infile:
                self.__dict__['__values'] = json.load(infile)
        elif logdir_pretrained is not None:
            logging.info('load flags from logdir_pretrained: %s', logdir_pretrained)
            # new_train_data_path = default_config['train_data_path']
            # new_extensions = default_config['extensions']
            with open(os.path.join(logdir_pretrained, 'flags.json'), 'r') as infile:
                self._values = json.load(infile)
            self.__dict__['__values']['train_data_path'] = default_config['train_data_path']
            self.__dict__['__values']['extensions'] = default_config['extensions']
        elif values is not None:
            self.__dict__['__values'] = values
        else:
            self.__dict__['__values'] = default_config

    def __getattr__(self, name):
        """Retrieves the 'value' attribute of the entry name."""

        if name not in self.__dict__['__values']:
            raise AttributeError(name)
        return self.__dict__['__values'][name][1]

    def as_dict(self):
        return {k: self.__getattr__(k) for k in self}

    def __str__(self):
        return json.dumps(self.as_dict(), sort_keys=True)

    __repr__ = __str__

    def __setattr__(self, key, value):
        """Sets the 'value' attribute of the entry name."""

        if key not in self.__dict__['__values']:
            raise AttributeError(key)
        self.__dict__['__values'][key][1] = value

    def __iter__(self):
        return iter(self.__dict__['__values'])

    def add_entry(self, name, value, _type='string', description='', short_name=None):
        if name in self.__dict__['__values']:
            raise AttributeError('entry with name=%s already exists' % name)
        assert _type in ALLOWED_TYPES, '_type=%s, but has to be one of: %s' % (_type, ', '.join(ALLOWED_TYPES))
        self.__dict__['__values'][name] = ['DEFINE_%s' % _type, value, description]
        if short_name is not None:
            self.__dict__['__values'][name].append(short_name)

    def dump(self, logdir):
        # write flags for current run
        filename = os.path.join(logdir, 'flags.json')
        with open(filename, 'w') as outfile:
            json.dump(self.__dict__['__values'], outfile, indent=2, sort_keys=True)

    def serialize(self, filter_flags=None):
        keys = [k for k in self.__dict__['__values'].keys() if filter_flags is None or k in filter_flags]
        res = []
        for flag in sorted(keys):
            # get real flag value
            # new_value = getattr(FLAGS, flag)
            # default_config[flag][1] = new_value
            # value = config[flag][1]
            value = getattr(self, flag)
            entry_values = self.__dict__['__values'][flag]

            # collect run description
            # if 'run_description' not in config:
            # if a short flag name is set, use it. if it is set to None, add this flag not to the run_descriptions
            if len(entry_values) < 4 or entry_values[3]:
                if len(entry_values) >= 4:
                    flag_name = entry_values[3]
                else:
                    flag_name = flag
                flag_name = flag_name.replace('_', '')
                flag_value = str(value).replace('_', '').replace(',', '-')
                # if flag_value is a path, take only the last two subfolders
                flag_value = ''.join(flag_value.split(os.sep)[-2:])
                res.append(flag_name.lower() + flag_value.upper())
        return '_'.join(res)

    def get_model_description(self):
        return self.serialize(filter_flags=MODEL_PARAMETERS)

    def set_run_description(self):
        if 'run_description' not in self.__dict__['__values']:
            run_desc = self.serialize()
            self.__dict__['__values']['run_description'] = ['DEFINE_string',
                                                            run_desc,
                                                            'short string description of the current run',
                                                            None]
            logging.debug('set run description: %s' % self.run_description)
            # if 'run_description' not in config:
            #    config['run_description'] = ['DEFINE_string', '_'.join(run_desc), 'short string description of the current run', None]
            #    logging.info('serialized run description: ' + config['run_description'][1])

    def update_with_flags(self, flags):
        for flag in self:
            # get real flag value
            if flag in flags.__dict__['__flags']:
                new_value = getattr(flags, flag)
                self.__setattr__(flag, new_value)

    def init_flags(self):
        for flag in self.__dict__['__values'].keys():
            v = self.__dict__['__values'][flag]
            getattr(tf.flags, v[0])(flag, v[1], v[2])

    def __deepcopy__(self):
        newone = type(self)(values=copy.deepcopy(self.__dict__['__values']))
        return newone

    def explode(self, value_dict):
        res = []
        for d in mytools.dict_product(value_dict):
            temp = self.__deepcopy__()
            for k in d.keys():
                temp.__setattr__(key=k, value=d[k])
            res.append((temp, d))
        return value_dict.keys(), res

    def create_new_configs(self, config_dicts_list):
        """
        Creates multiple new configs
        Use this config as default and create for every config dict a new config
        :param config_dicts_list: list of dicts to update teh default config with
        :return: parameter_keys, list of tuples(config, selected_parameter_dict)
        """
        res = []
        parameter_keys_ = []
        for d in config_dicts_list:
            parameter_keys_.extend(d.keys())
        parameter_keys = set(parameter_keys_)
        for d in config_dicts_list:
            temp = self.__deepcopy__()
            for k in parameter_keys:
                if k in d.keys():
                    temp.__setattr__(key=k, value=d[k])
                else:
                    d[k] = self.__getattr__(k)
            res.append((temp, d))
        return parameter_keys, res
