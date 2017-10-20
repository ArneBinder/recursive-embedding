from __future__ import absolute_import
import shutil

import os
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from google.protobuf import text_format
import tensorflow as tf

tf.flags.DEFINE_string('dir',
                       os.getcwd(),
                       'Root directory for looking recursively downwards for checkpoint files. Defaults to the current working derectory.')

FLAGS = tf.flags.FLAGS


def get_all_checkpoint_states(root_dir):
    check_list = [[os.path.join(root, filename) for filename in filenames if filename == 'checkpoint'] for
                  root, directories, filenames in os.walk(root_dir)]
    return [fn_list[0] for fn_list in check_list if len(fn_list) > 0]


def update_checkpoint_state(check_file):

    check_dir = os.path.dirname(os.path.abspath(check_file))
    # Read the existing checkpoint file.
    current_checkpoint_state = tf.train.get_checkpoint_state(check_dir)
    # collect model checkpoint names (look for .meta files)
    file_list = [os.path.join(check_dir, os.path.splitext(file)[0]) for file in os.listdir(check_dir) if file.endswith(".meta")]
    if len(file_list) == 0:
        print('WARNING: no model checkpoints found in "%s"' % check_dir)
        return
    # sort by step
    file_list.sort(key=lambda x: int(x.split('-')[-1]))

    new_checkpoint_state = CheckpointState(model_checkpoint_path=file_list[-1], all_model_checkpoint_paths=file_list)
    if current_checkpoint_state == new_checkpoint_state:
        return

    # backup, if not already done (preserves original)
    if not os.path.isfile(check_file + '.backup'):
        shutil.copyfile(check_file, check_file + '.backup')

    with open(check_file, 'w') as f:
        f.write(text_format.MessageToString(new_checkpoint_state))
    print('updated %s' % check_dir)


if __name__ == '__main__':
    all_checkpoints = get_all_checkpoint_states(FLAGS.dir)
    for ckpt in all_checkpoints:
        update_checkpoint_state(ckpt)
