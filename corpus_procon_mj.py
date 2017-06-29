import json
import pprint
import re
import numpy as np
import httplib2
import logging
from pprint import pformat
from sklearn.metrics import pairwise_distances

import sheets_connector

spreadsheet_debate = '16oLBPKeqtIwtC7uzoA7kB9cBA81iPI-P05q9ItVmx1Y'
range_debate = 'selected_columns_labeled'
range_arg_mappings = 'unique_labels'


def posts_and_labels():
    raw_posts_iter = sheets_connector.get_sheet_values(spreadsheet_id=spreadsheet_debate, range_name=range_debate,
                                                       as_dict=True)
    raw_label_mappings_iter = sheets_connector.get_sheet_values(spreadsheet_id=spreadsheet_debate,
                                                                range_name=range_arg_mappings, as_dict=True)

    pattern = re.compile(r"\s+")

    def to_list(s):
        return pattern.sub("", s).split(',')

    label_mappings = {}
    for label_mapping in raw_label_mappings_iter:
        # has to have associated values
        if 'labels_mapped' in label_mapping:
            label_mappings[label_mapping['labels']] = to_list(label_mapping['labels_mapped'])

    posts = {}
    labels_set = set()
    for post in raw_posts_iter:
        post_id = int(post['post_id'])
        # take all columns except post_id and labels. post_id will be the dict key and labels are processed further
        posts[post_id] = {k: post[k] for k in post.keys() if k != 'post_id' and k != 'labels'}
        labels_mapped = []
        if 'labels' in post:
            labels = to_list(post['labels'])
            for label in labels:
                if label != u'':
                    if label not in label_mappings:
                        logging.warn('no mapping found for label='+label)
                        #print('WARNING: no mapping found for label='+label)
                    else:
                        current_labels_mapped = label_mappings[label]
                        labels_mapped.extend(current_labels_mapped)
                        labels_set |= set(current_labels_mapped)

        posts[post_id]['labels'] = labels_mapped

    return posts, labels_set


def label_distance(post1_v, post2_v):
    distance = pairwise_distances([post1_v], [post2_v], metric='euclidean')
    return distance


def label_vector(post, label_mapping):
    v = np.zeros(shape=(len(label_mapping)), dtype=np.int32)
    for label in post['labels']:
        idx = label_mapping[label]
        v[idx] += 1
    return v


def request_embeddings(sequences, sentence_processor='process_sentence7', concat_mode='sequence', url='http://127.0.0.1:5000/api/embed'):
    logging.info('request embeddings ...')
    http_obj = httplib2.Http()
    resp, content = http_obj.request(
        uri=url,
        method='POST',
        headers={'Content-Type': 'application/json; charset=UTF-8'},
        body=json.dumps({'sequences': sequences, 'sentence_processor':sentence_processor, 'concat_mode':concat_mode}),
    )
    logging.info('response embeddings:')
    logging.info(pformat(resp))
    return json.loads(content)['embeddings']


def main():
    pp = pprint.PrettyPrinter(indent=2)
    posts, labels = posts_and_labels()
    labels_list = list(labels)
    m = {v: i for i, v in enumerate(labels_list)}

    posts_content = [posts[p_id]['content'] for p_id in posts]
    print(json.dumps(posts_content))
    posts_vecs_label = [label_vector(posts[p_id], m) for p_id in posts]
    #posts_vecs_embedding = request_embeddings(posts_content)

    # TODO: cache vecs

    distances_labels = pairwise_distances(posts_vecs_label, metric='euclidean')
    #distances_embeddings = pairwise_distances(posts_vecs_embedding, metric='euclidean')

    # TODO: evaluate embeddings

    #print('')


    for i, post_id in enumerate(sorted(posts.keys())):
        #if i > 10:
        #    break
        labels_mapped_joined = ", ".join(posts[post_id]['labels'])
        #print(str(post_id) + '\t' + labels_mapped_joined)
        #post = posts[post_id]
        #pp.pprint(post)
        #v = label_vector(post, m)
        #print(v)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
