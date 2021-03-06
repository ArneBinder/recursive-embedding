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
                                      'Path to lexicon containing additional vecs, e.g. biomedical embeddings., that are '
                                      'concatenated with default embeddings.',
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
                                    None
                                    ],
                  'epochs': ['DEFINE_integer',
                             1000000,
                             'The number of epochs.',
                             None],
                  'dev_file_indices': ['DEFINE_string',
                                     "0",
                                     'Which file(s) of the train data files should be used as test data.',
                                     'dfidx'],
                  'tree_embedder': ['DEFINE_string',
                                    'FLAT_AVG',
                                    'TreeEmbedder implementation from model_fold that produces a tensorflow fold block on '
                                    'calling which accepts a sequence tree and produces an embedding. '
                                    'Currently implemented (see model_fold.py):'
                                    '"TREE_LSTM"           -> TreeLSTM'
                                    '"HTU_GRU"             -> Headed Tree Unit, using a GRU for order aware '
                                    '                                       and summation for order unaware composition'
                                    '"FLAT_AVG"            -> Averaging applied to first level children '
                                    '                                       (discarding the root)'
                                    '"FLAT_AVG_2levels"    -> Like TreeEmbedding_FLAT_AVG, but concatenating first'
                                    '                                       second level children (e.g. dep-edge embedding) to '
                                    '                                       the first level children (e.g. token embeddings)'
                                    '"FLAT_LSTM"           -> LSTM applied to first level children (discarding the'
                                    '                                       root)'
                                    '"FLAT_LSTM_2levels"   -> Like TreeEmbedding_FLAT_LSTM, but concatenating '
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
                  'state_size': ['DEFINE_string',
                                 "50",
                                 'size(s) of the composition layer. Multiple entries have to be separated by comma.',
                                 'st'],
                  'learning_rate': ['DEFINE_float',
                                    0.02,
                                    # 'tanh',
                                    'learning rate',
                                    'lr'],
                  'optimizer': ['DEFINE_string',
                                'AdamOptimizer',
                                'optimizer',
                                None],
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
                  'keep_prob_blank': ['DEFINE_float',
                                     1.0,
                                     'Keep probability for full node dropout (blanking)',
                                     'kpb'
                                     ],
                  'keep_prob_node': ['DEFINE_float',
                                1.0,
                                'Keep probability for full node dropout (subtree dropping)',
                                'kpn'
                                ],
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
                  'task': ['DEFINE_string',
                           None,
                           'the task',
                           'tk'],
                  'blank': ['DEFINE_string',
                           "",
                           'stop tree construction at all tokens with these prefixes (split by ",")',
                           'b'],
                  'add_heads': ['DEFINE_string',
                           "",
                           'add nodes with these prefixes to there parents (split by ",")',
                           'a'],
                  'sample_method': ['DEFINE_string',
                           "U",
                           'default sample method for reroot model. U: uniform, F: frequency, N: nearest; UA, FA and NA sample from all selected lexicon / data indices, not just from the class of the correct head',
                           'sm'],
                  'concat_mode': ['DEFINE_string',
                                  'tree',
                                  'how to concatenate the tokens (tree: use tree structure, sequence: as ordered '
                                  'sequence, aggregate: bag-of-tokens)',
                                  'cm'],
                  'sequence_length': ['DEFINE_integer',
                                      1000,
                                      'maximum number of tokens for plain structures (flatconcat and TFIDF)',
                                      'sl'],
                  'neg_samples': ['DEFINE_string',
                                  "",
                                  'count of negative samples per tree',
                                  'ns'
                                  ],
                  'nbr_trees': ['DEFINE_string',
                                "",
                                'number of trees for one epoch when training the REROOT model',
                                'n'],
                  'nbr_trees_test': ['DEFINE_string',
                                     "",
                                     'If > 0, number of trees for one test epoch when training the REROOT model. '
                                     'Otherwise, use nbr_trees',
                                     None],
                  'no_fixed_vecs': ['DEFINE_boolean',
                                    False,
                                    'If enabled, train all embedding vecs (including these in .fix.npy file).',
                                    'nfv'],
                  'all_vecs_fixed': ['DEFINE_boolean',
                                    False,
                                    'If enabled, do not train any embedding vecs.',
                                    'avf'],
                  'var_vecs_zero': ['DEFINE_boolean',
                                    False,
                                    'If enabled, initialize trainable embedding vecs with zero.',
                                    'vvz'],
                  'var_vecs_random': ['DEFINE_boolean',
                                    False,
                                    'If enabled, initialize trainable embedding vecs with random values.',
                                    'vvr'],
                  'dump_trees': ['DEFINE_boolean',
                                    False,
                                    'If enabled, dump compiled trees in cache dir.',
                                    'dt'],
                  'use_tfidf': ['DEFINE_boolean',
                                      False,
                                      'If enabled, append TF-IDF embeddings to the tree model output.',
                                      'tfidf'],
                  'bidirectional': ['DEFINE_boolean',
                                    False,
                                    'If enabled, follow edges in opposite direction. Works only with tree models',
                                    'd'],
                  'exclude_class': ['DEFINE_string',
                                    '',
                                    'exclude class for multi class prediction. '
                                    'NOTE: Setting this, sets exclusive_classes=True',
                                    'ec'],
                  'use_circular_correlation': [
                      'DEFINE_boolean',
                      False,
                      'If enabled, use circular self correlation in TreeTuple- and TreeSingle models.',
                      'cc'],
                  'merge_factor': ['DEFINE_string',
                                   '1',
                                   'merge multiple embeddings (only for FLAT models), depends on used sentence_processor',
                                   None],
                  }

ALLOWED_TYPES = ['string', 'float', 'integer', 'boolean']

# used for compile trees
TREE_MODEL_PARAMETERS = ['additional_vecs', 'leaf_fc_size', 'root_fc_sizes', 'state_size', 'tree_embedder']
# used for compile trees
TREE_STRUCTURE_PARAMETERS = ['max_depth', 'context', 'link_cost_ref', 'concat_mode', 'sequence_length',
                             'no_fixed_vecs', 'all_vecs_fixed', 'add_heads', 'blank', 'bidirectional']
MODEL_PARAMETERS = TREE_MODEL_PARAMETERS + ['fc_sizes', 'use_tfidf', 'use_circular_correlation', 'model_type']
DESCRIPTION_PARAMETERS = MODEL_PARAMETERS + TREE_STRUCTURE_PARAMETERS + ['var_vecs_zero', 'var_vecs_random', 'sample_method', 'task']
FLAGS_FN = 'flags.json'

VALUES_SHORT = {'TRUE': 'T', 'FALSE': 'F', 'REROOT': 'LM'}


class Config(object):
    def __init__(self, logdir=None, logdir_pretrained=None, values=None):
        if logdir:
            self.set_from_logdir(logdir)
        elif logdir_pretrained:
            self.set_from_logdir(logdir_pretrained)
        elif values is not None:
            self.__dict__['__values'] = values
        else:
            self.__dict__['__values'] = default_config

    def set_from_logdir(self, logdir, parameter_whitelist=None):
        with open(os.path.join(logdir, FLAGS_FN), 'r') as infile:
            values_dict = json.load(infile)
        if parameter_whitelist is not None:
            values_dict = {k: values_dict[k] for k in values_dict if k in parameter_whitelist}

        # add (new) default config values and keys excluded via whitelist
        for p in list(set(default_config) - set(values_dict)):
            values_dict[p] = default_config[p]
        self.__dict__['__values'] = values_dict

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
        filename = os.path.join(logdir, FLAGS_FN)
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
                # if flag_value is a path, take only the last folder name
                if os.sep in flag_value:
                    flag_value = flag_value.split(os.sep)[-2]
                flag_value = flag_value.upper()
                flag_value = VALUES_SHORT.get(flag_value, flag_value)
                res.append(flag_name.lower() + flag_value)
        return '_'.join(res)

    def get_description(self):
        return self.serialize(filter_flags=DESCRIPTION_PARAMETERS)

    def get_serialization_for_compile_trees(self):
        return self.serialize(filter_flags=TREE_MODEL_PARAMETERS + TREE_STRUCTURE_PARAMETERS)

    def get_serialization_for_calculate_tfidf(self):
        return self.serialize(filter_flags=['sequence_length'])

    def set_run_description(self):
        run_desc = self.serialize()
        self.__dict__['__values']['run_description'] = ['DEFINE_string',
                                                        run_desc,
                                                        'short string description of the current run',
                                                        None]
        logging.debug('set run description: %s' % self.run_description)

    def update_with_flags(self, flags, keep_model_parameters=True):
        blacklist = []
        if keep_model_parameters:
            if getattr(flags, 'model_type') != self.model_type:
                blacklist = TREE_MODEL_PARAMETERS
            else:
                blacklist = MODEL_PARAMETERS

        for flag in self:
            # get real flag value
            if flag in flags.__dict__['__flags'] and flag not in blacklist:
                new_value = getattr(flags, flag)
                self.__setattr__(flag, new_value)

    def init_flags(self):
        for flag in self.__dict__['__values'].keys():
            v = self.__dict__['__values'][flag]
            getattr(tf.flags, v[0])(flag, v[1], v[2])

    def __deepcopy__(self, memo):
        result = type(self)(values=copy.deepcopy(self.__dict__['__values'], memo))
        memo[id(self)] = result
        return result

    def explode(self, value_dict, previous_fieldnames=None):
        previous_dict = {}
        if previous_fieldnames is not None:
            previous_dict = {fnl: self.__dict__['__values'][fnl][1] for fnl in previous_fieldnames if fnl in self.__dict__['__values'].keys()}
        res = []
        for d in mytools.dict_product(value_dict):
            temp = copy.deepcopy(self)
            for k in d.keys():
                temp.__setattr__(key=k, value=d[k])
            d.update({k: previous_dict[k] for k in previous_dict.keys() if k not in d.keys()})
            res.append((temp, d))
        return list(set(value_dict.keys() + previous_dict.keys())), res

    def create_new_configs(self, config_dicts_list, previous_fieldnames=None):
        """
        Creates multiple new configs
        Use this config as default and create for every config dict a new config
        :param config_dicts_list: list of dicts to update teh default config with
        :return: parameter_keys, list of tuples(config, selected_parameter_dict)
        """
        previous_dict = {}
        if previous_fieldnames is not None:
            previous_dict = {fnl: self.__dict__['__values'][fnl][1] for fnl in previous_fieldnames if
                             fnl in self.__dict__['__values'].keys()}
        res = []
        parameter_keys_ = []
        for d in config_dicts_list:
            parameter_keys_.extend(d.keys())
        parameter_keys_.extend(previous_dict.keys())
        parameter_keys = set(parameter_keys_)
        for d in config_dicts_list:
            temp = copy.deepcopy(self)
            for k in parameter_keys:
                if k in d.keys():
                    temp.__setattr__(key=k, value=d[k])
                else:
                    d[k] = self.__getattr__(k)
            res.append((temp, d))
        return parameter_keys, res
