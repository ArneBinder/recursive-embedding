#p = '/mnt/DATA/ML/data/corpora_in/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/'
p = ''
lines_in = list(open(p+'TEST_FILE_FULL.TXT').readlines())
res = []
for i in range(0, len(lines_in), 4):
    res.append((lines_in[i].split('\t')[0], lines_in[i+1].strip()))
open(p+'TEST_FILE_KEY_DIRECTION.TXT', 'w').writelines(('%s\t%s\n' % (_id, _rel) for _id, _rel in res))
