
from src.lexicon import Lexicon
from src.sequence_trees import Forest

if __name__ == '__main__':
    lexicon = Lexicon(
        filename='/mnt/WIN/ML/data/corpora-test3/SICK/process_sentence10/SICK_cmSEQUENCE_icmTREE_onehotdep')

    data = [0, 12, 9, 95, 7, 26, 2, 163, 8, 10, 3, 29, 7, 17, 0, 11, 2, 350, 19, 20, 0, 61, 14, 309, 9, 16, 8, 10, 18,
            42, 7, 17, 0, 15, 2, 251, 1, 5, 4, 6]
    parents = [2, -1, 8, -1, -2, -1, -2, -1, 2, -1, 28, -1, -2, -1, 2, -1, -4, -1, -8, -1, 4, -1, 2, -1, 4, -1, 2, -1,
               -18, -1, -2, -1, 2, -1, -4, -1, -8, -1, 1, 0]

    forest1 = Forest(data=data, parents=parents, lexicon=lexicon)

    #forest.visualize(filename='test.svg')
    forest1.reset_cache_values()
    forest1.children_dict_to_arrays()
    #forest.children_array_to_dict()

    forest1.dump(filename='forest1')
    forest2 = Forest(filename='forest1', lexicon=lexicon)
    forest2_children = forest2._children.tolist()
    forest2_children_offset = forest2._children_pos.tolist()
    forest2.visualize(filename='test_arrays.svg')

    Forest.concatenate([forest1, forest2]).dump(filename='forest3')
    forest3 = Forest(filename='forest3', lexicon=lexicon)
    forest3.visualize(filename='test_conc.svg')

    forest3.extend([forest1])
    forest3.dump(filename='forest4')
    forest4 = Forest(filename='forest4', lexicon=lexicon)
    forest4.visualize(filename='test_extend.svg')
    print('')
