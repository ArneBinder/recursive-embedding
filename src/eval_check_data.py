
import numpy as np
import plac


def get_data_spans(data, start_pos, indices):
    res = []
    _start_pos = start_pos[indices]
    _end_pos = start_pos[indices + 1]
    for i, idx in enumerate(indices):
        data_span = data[_start_pos[i]:_end_pos[i]]
        # blank id
        data_span[1] = 0
        res.append((idx, data_span))
    return res


def data_spans_overlap(data_spans, other_data_spans):
    indices_similar = []
    for idx, span in data_spans:
        for idx_other, other_span in other_data_spans:
            if np.array_equal(span, other_span):
                indices_similar.append((idx, idx_other))
    return indices_similar


@plac.annotations(
    path=('path to recemb data', 'option', 'p', str),
    train_index_files=('comma separated list of index file suffixes, e.g. "train.0,train.1"', 'option', 'a', str),
    test_index_files=('comma separated list of index file suffixes, e.g. "test.0,test.1"', 'option', 'e', str),
)
def check_data_overlap(path, train_index_files, test_index_files):

    train_indices = np.sort(np.concatenate([np.load('%s.idx.%s.npy' % (path, prefix_with_idx)) for prefix_with_idx in train_index_files.split(',')]))
    test_indices = np.sort(np.concatenate([np.load('%s.idx.%s.npy' % (path, prefix_with_idx)) for prefix_with_idx in test_index_files.split(',')]))
    assert not any(np.isin(train_indices, test_indices)), 'train indices are in test indices!'
    assert not any(np.isin(test_indices, train_indices)), 'test indices are in train indices!'
    data = np.load('%s.data.npy' % path)
    start_pos = np.load('%s.root.pos.npy' % path)
    start_pos = np.concatenate((start_pos, [len(data)]))
    data_spans_train = get_data_spans(data, start_pos, indices=train_indices)
    data_spans_test = get_data_spans(data, start_pos, indices=test_indices)
    #assert not any([any([np.array_equal(data_span, other_data_span) for other_data_span in data_spans_test]) for data_span in data_spans_train]), 'train data span in test data spans'
    #assert not any([(data_span in data_spans_test) for data_span in data_spans_train]), 'train data span in test data spans'
    #assert not any([(data_span in data_spans_train) for data_span in data_spans_test]), 'test data span in train data spans'
    indices_similar = data_spans_overlap(data_spans_train, data_spans_test)
    assert len(indices_similar) == 0, 'train data span in test data spans: %s' % str(indices_similar)


if __name__ == '__main__':
    plac.call(check_data_overlap)
    print('done')
