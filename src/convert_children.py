
from sequence_trees import Forest
import glob
import os

if __name__ == '__main__':

    p_dir = '/mnt/WIN/ML/data/corpora/DBPEDIANIF/1000_dep'
    os.chdir(p_dir)
    for file in glob.glob("*.parent"):
        fn = os.path.join(p_dir, file[:-len('.parent')])
        print(fn)
        forest = Forest(filename=fn)
        forest.children_dict_to_arrays()
        forest.dump(fn)
