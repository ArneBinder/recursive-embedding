import corpus
import preprocessing
import tools
import spacy
import constants
import visualize

nlp = spacy.load('en')
nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]
print('extract word embeddings from spaCy...')
vecs, mapping = preprocessing.get_word_embeddings(nlp.vocab)
# for processing parser output
#data_maps = {constants.WORD_EMBEDDING: mapping}
data_maps2 = mapping
# data vectors
#data_vecs = {constants.WORD_EMBEDDING: vecs}

def read_sentence2(sentence, vis = False):
    seq_data, seq_parents, root = preprocessing.read_data(preprocessing.string_reader, preprocessing.process_sentence5, nlp, data_maps2, args={'content': sentence})
    print('root: ' + str(root))

    data_maps_reverse = corpus.revert_mapping(data_maps2)

    #print( 'counts: ' +str(getCounts(seq_data, nlp.vocab, constants.vocab_manual, data_maps_reverse)))
    # print('counts: ' + str(tools.getFromDicts(nlp.vocab, constants.vocab_manual,)))

    if vis:
        visualize.visualize('forest_temp.png', (seq_data, seq_parents), data_maps_reverse, nlp.vocab, constants.vocab_manual)
        #img = Image('forest_temp.png')
        #display(img)

    return seq_data, seq_parents, root

(seq_data, seq_parents, root) = read_sentence2('London is a big city in the United Kingdom. I like this.', True)
#(seq_data, seq_parents, root) = read_sentence2('London is a big city in the United Kingdom.', True)