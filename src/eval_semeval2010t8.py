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


def ids_to_str(sc, fn_in, fn_out, exclude_class=None, threshold=0.5):
    v = np.load(fn_in)
    open(fn_out, 'w').writelines(
        ('%i\t%s\n' % (i, format_rel(sc[idx] if exclude_class is None or v[i, idx] >= threshold else exclude_class)) for i, idx in enumerate(np.argmax(v, axis=1))))


def convert_values(path, class_strings, fn_gold_tsv, fn_predicted_tsv, exclude_class=None, threshold=0.5):
    sc = None
    for class_string in class_strings:
        _fn = (os.path.join(path, 'data.%s.classes.strings' % class_string))
        if os.path.exists(_fn):
            sc = list((s[:-1] for s in open(_fn)))
            break
    assert sc is not None, 'No classes strings file found (looked for %s)' % str(['data.%s.classes.strings' % s for s in class_strings])

    #fn_predicted_tsv = os.path.join(path, 'strings_predicted.tsv')
    #fn_gold_tsv = os.path.join(path, 'strings_gold.tsv')
    ids_to_str(sc=sc, fn_in=os.path.join(path, 'values_predicted.np'), fn_out=fn_predicted_tsv, exclude_class=exclude_class, threshold=threshold)
    ids_to_str(sc=sc, fn_in=os.path.join(path, 'values_gold.np'), fn_out=fn_gold_tsv, exclude_class=exclude_class, threshold=threshold)
    return fn_predicted_tsv, fn_gold_tsv


def eval(path_dir, exclude_class=None, threshold=0.5, class_strings=(SEMEVAL_RELATION, TACRED_RELATION, 'RELS', 'RELT'),
         fn_script='~/recursive-embedding/docker/preprocessing/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2_tacred.pl'):

    fn_gold_strings = os.path.join(path_dir, 'values_gold_strings')
    fn_max_strings = os.path.join(path_dir, 'values_predicted_strings')
    if exclude_class is not None:
        print('exclude class: %s, use threshold: %f' % (exclude_class, threshold))
        fn_gold_strings += '_t%.2f' % threshold
        fn_max_strings += '_t%.2f' % threshold
    fn_gold_tsv = fn_gold_strings + '.tsv'
    fn_predicted_tsv = fn_max_strings + '.tsv'
    if not os.path.exists(fn_gold_tsv) or not os.path.exists(fn_predicted_tsv):
        if os.path.exists(fn_gold_strings + '.txt') and os.path.exists(fn_max_strings + '.txt'):
            open(fn_gold_tsv, 'w').writelines(('%i\t%s\n' % (i, format_rel(l)) for i, l in enumerate(open(fn_gold_strings + '.txt').readlines()) if l.startswith(SEMEVAL_RELATION) or l.startswith(TACRED_RELATION)))
            open(fn_predicted_tsv, 'w').writelines(('%i\t%s\n' % (i, format_rel(l)) for i, l in enumerate(open(fn_max_strings + '.txt').readlines()) if l.startswith(SEMEVAL_RELATION) or l.startswith(TACRED_RELATION)))
        else:
            print('%s or %s not found, create from probabilities...' % (fn_max_strings, fn_gold_strings))
            convert_values(path=path_dir, fn_gold_tsv=fn_gold_tsv, fn_predicted_tsv=fn_predicted_tsv,
                           class_strings=class_strings, exclude_class=exclude_class, threshold=threshold)

    check_script = 'perl %s %s %s' % (fn_script, fn_predicted_tsv, fn_gold_tsv)
    perl_result = subprocess.check_output(check_script, shell=True)
    open(os.path.join(path_dir, 'eval.txt'), 'w').write(perl_result)
    perl_result_lines = perl_result.split('\n')
    f1_micro_idx = perl_result_lines.index('Micro-averaged result (excluding Other):') + 1
    f1_micro_str = perl_result_lines[f1_micro_idx].split('=')[-1].replace('%', '')
    f1_micro_score = float(f1_micro_str) / 100
    last_line = perl_result_lines[-2]
    score_str = last_line.replace('<<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = ', '').replace('% >>>', '')
    f1_score = float(score_str) / 100
    return f1_score, f1_micro_score


if __name__ == "__main__":
    import plac
    print(plac.call(eval))