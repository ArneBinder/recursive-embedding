# conda install backports.csv

from backports import csv
from io import open

fn_scores = '/mnt/DATA/ML/data/corpora_in/BIOSSES-Dataset/SICK_format/scores.tsv'
fn_sentences = '/mnt/DATA/ML/data/corpora_in/BIOSSES-Dataset/SICK_format/sentences.tsv'


def loadtsv(fn):
    with open(fn, encoding='utf-8') as tsvf:
        reader = csv.DictReader(tsvf, dialect='excel-tab')
        return list(reader)


def clean_sentence(sentence):
    sentence = sentence.strip()
    if sentence.endswith('.'):
        sentence = sentence[:-1]
    return sentence


if __name__ == '__main__':
    fn_scores = '/mnt/DATA/ML/data/corpora_in/BIOSSES/SICK_format/scores.tsv'
    fn_sentences = '/mnt/DATA/ML/data/corpora_in/BIOSSES/SICK_format/sentences_cleaned.tsv'
    fn_out = '/mnt/DATA/ML/data/corpora_in/BIOSSES/SICK_format/data.tsv'

    scores = loadtsv(fn_scores)
    sentences = loadtsv(fn_sentences)
    scores_dict = {s['pair_ID'].strip(): s for s in scores}

    for i in range(len(sentences)):
        sentences[i].update(scores_dict[sentences[i]['pair_ID'].strip()])
        sentences[i]['sentence_A'] = clean_sentence(sentences[i]['sentence_A'])
        sentences[i]['sentence_B'] = clean_sentence(sentences[i]['sentence_B'])
        score_mean = 0.0
        annot_scores = [int(sentences[i][k]) for k in sentences[i] if k.startswith('Annotator_')]
        assert len(annot_scores) == 5, 'wrong number of annotators: %i, expected 5.' % len(annot_scores)
        sentences[i]['relatedness_score'] = sum(annot_scores) / 5.0 + 1.0
        sentences[i]['entailment_judgment'] = 'DUMMY'
        sentences[i]['pair_ID'] = 'B' + sentences[i]['pair_ID']

    with open(fn_out, 'w', encoding='utf-8') as tsvf:
        reader = csv.DictWriter(tsvf, fieldnames=['pair_ID', 'sentence_A', 'sentence_B', 'relatedness_score', 'entailment_judgment'],
                                dialect='excel-tab', extrasaction='ignore', quotechar='|', quoting=csv.QUOTE_NONE)
        reader.writeheader()
        reader.writerows(sentences)

    print('done')