import csv
import pickle

import tools


def write_dict(out_path, mapping, vecs, vocab_nlp=None, vocab_manual=None):
    print('dump embeddings to: ' + out_path + '.vecs ...')
    vecs.dump(out_path + '.vecs')
    print('dump mappings to: ' + out_path + '.mapping ...')
    with open(out_path + '.mapping', "wb") as f:
        pickle.dump(mapping, f)
    print('vecs.shape: ' + str(vecs.shape) + ', len(mapping): ' + str(len(mapping)))

    if vocab_nlp is not None:
        print('write tsv dict: ' + out_path + '.tsv ...')
        rev_map = tools.revert_mapping(mapping)
        with open(out_path + '.tsv', 'wb') as csvfile:
            fieldnames = ['label', 'id_orig']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)

            writer.writeheader()
            for i in range(len(rev_map)):
                id_orig = rev_map[i]
                if id_orig >= 0:
                    label = vocab_nlp[id_orig].orth_
                else:
                    label = vocab_manual[id_orig]
                writer.writerow({'label': label.encode("utf-8"), 'id_orig': str(id_orig)})