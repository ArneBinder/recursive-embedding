import os
import subprocess
import numpy as np

from constants import TYPE_RELATION, SEMEVAL_RELATION, TACRED_RELATION


def format_rel(rel):
    if rel.startswith(SEMEVAL_RELATION):
        rel_split = rel.strip().split('=')
        return rel_split[1]
    if rel.startswith(TACRED_RELATION):
        rel_split = rel.strip().split('=')
        rel_string = rel_split[1]
        if rel_string == 'no_relation':
            return 'Other'
        rel_string = rel_string.replace(':', '-').replace('/', '_')
        return rel_string + '(e1,e2)'
    rel_split = rel.strip().split('/')
    if rel_split[-1] == 'Other':
        return rel_split[-1]
    if rel_split[-2] == 'FW':
        return '%s(e1,e2)' % rel_split[-1]
    if rel_split[-2] == 'BW':
        return '%s(e2,e1)' % rel_split[-1]
    if rel_split[0] == TYPE_RELATION:
        return rel_split[1]
    raise Exception('unknown format for relation: %s' % rel)


def ids_to_str(sc, fn_in, fn_out):
    v = np.load(fn_in)
    open(fn_out, 'w').writelines(
        ('%i\t%s\n' % (i, format_rel(sc[idx])) for i, idx in enumerate(np.argmax(v, axis=1))))


def convert_values(path, class_strings):
    sc = None
    for class_string in class_strings:
        _fn = (os.path.join(path, 'data.%s.classes.strings' % class_string))
        if os.path.exists(_fn):
            sc = list((s[:-1] for s in open(_fn)))
            break
    assert sc is not None, 'No classes strings file found (looked for %s)' % str(['data.%s.classes.strings' % s for s in class_strings])

    fn_predicted_tsv = os.path.join(path, 'strings_predicted.tsv')
    fn_gold_tsv = os.path.join(path, 'strings_gold.tsv')
    ids_to_str(sc=sc, fn_in=os.path.join(path, 'values_predicted.np'), fn_out=fn_predicted_tsv)
    ids_to_str(sc=sc, fn_in=os.path.join(path, 'values_gold.np'), fn_out=fn_gold_tsv)
    return fn_predicted_tsv, fn_gold_tsv


def eval(path_dir, class_strings=(SEMEVAL_RELATION, TACRED_RELATION, 'RELS', 'RELT'),
         fn_script='~/recursive-embedding/docker/create-corpus/semeval2010task8/data/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2_tacred.pl'):

    fn_gold_strings = os.path.join(path_dir, 'values_gold_strings')
    fn_max_strings = os.path.join(path_dir, 'values_predicted_strings')
    if os.path.exists(fn_gold_strings + '.txt') and os.path.exists(fn_max_strings + '.txt'):
        fn_gold_tsv = fn_gold_strings + '.tsv'
        fn_predicted_tsv = fn_max_strings + '.tsv'
        open(fn_gold_tsv, 'w').writelines(('%i\t%s\n' % (i, format_rel(l)) for i, l in enumerate(open(fn_gold_strings + '.txt').readlines())))
        open(fn_predicted_tsv, 'w').writelines(('%i\t%s\n' % (i, format_rel(l)) for i, l in enumerate(open(fn_max_strings + '.txt').readlines())))
    else:
        fn_predicted_tsv, fn_gold_tsv = convert_values(path=path_dir, class_strings=class_strings)

    check_script = 'perl %s %s %s' % (fn_script, fn_predicted_tsv, fn_gold_tsv)
    perl_result = subprocess.check_output(check_script, shell=True)
    open(os.path.join(path_dir, 'eval.txt'), 'w').write(perl_result)
    last_line = perl_result.split('\n')[-2]
    score_str = last_line.replace('<<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = ', '').replace('% >>>', '')
    f1 = float(score_str) / 100
    return f1


if __name__ == "__main__":
    import plac
    print(plac.call(eval))