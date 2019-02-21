import csv
import os
import re

import plac


def to_number(s):
    try:
        return float(s)
    except ValueError:
        return None


def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)


def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss


def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5


def move_to_front(l, entries):
    return entries + [e for e in l if e not in entries]


def collect_subpaths_with_fn(path, fn):
    res = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == fn:
                res.append(root)
    return res


@plac.annotations(
    out=('output file name', 'option', 'o', str),
    fn=('input file name', 'option', 'f', str),
    semeval=('output file name', 'flag', 'e', bool),
    exclude_class=('excluded class', 'option', 'c', str),
    threshold=('probability threshold to use exclude_class', 'option', 't', float),
    paths=('paths to folders', 'positional', None, str),
)
def load_and_merge_scores(out, fn='scores.tsv', semeval=False, exclude_class=None, threshold=0.5, *paths):
    data = []
    assert len(paths) > 0, 'no folders given'
    all_paths = []
    for path in paths:
        _paths = collect_subpaths_with_fn(path, fn)
        print('%s -> %s' % (path, _paths))
        all_paths.extend(_paths)
    for path in all_paths:
        print('read %s ...' % os.path.join(path, fn))
        dir_name = os.path.basename(path)
        new_data = list(csv.DictReader(open(os.path.join(path, fn)), delimiter='\t'))
        nbr_found = 0
        for d in new_data:
            d['dir'] = dir_name
            d['path'] = path
            if semeval:
                try:
                    from eval_semeval2010t8 import eval
                    run_dir = os.path.join(path, d['run_description'])
                    #assert os.path.exists(run_dir), 'path not found: %s' % run_dir
                    if not os.path.exists(run_dir):
                        #print('WARNING: path not found, skip: %s' % run_dir)
                        continue
                    #else:
                    #    print('XXX')
                    t_string = '_t%.2f' % threshold if exclude_class is not None else ''
                    d['f1_wo_norelation_macro' + t_string], d['f1_wo_norelation_micro' + t_string] = eval(path_dir=run_dir,
                                                                                    exclude_class=exclude_class,
                                                                                    threshold=threshold)
                    nbr_found += 1
                except Exception as e:
                    print('ERROR while getting semeval scores for %s:\n%s' % (dir_name, e))
            rd = d['run_description'].split('/')
            # ATTENTION: assume that dir_name is of format: something_SENTENCEPROCESSOR
            # the sentence processor is added to the run_desc used for arranging same settings
            d['run_desc'] = rd[0] + '_sp' + dir_name.split('_')[-1].upper()
            data.append({k: d[k] for k in d if d[k] != ''})
        if semeval:
            print('found prediction results for %i of %i runs' % (nbr_found, len(new_data)))

    fieldnames = sorted(list(set([item for sublist in data for item in sublist])))
    with open('%s.tsv' % out, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(data)

    data_arranged = {}
    for d in data:
        data_arranged.setdefault(d['run_desc'], []).append(d)
    stats = {}
    for run_desc in data_arranged:
        stats[run_desc] = {}
        runs = data_arranged[run_desc]
        runs_keys = set([item for sublist in runs for item in sublist])

        for k in runs_keys:
            try:
                current_values = [float(run[k]) for run in runs]
                stats[run_desc]['%s_mean' % k] = sum(current_values) / len(current_values)
                stats[run_desc]['%s_std' % k] = stddev(current_values)
            except (ValueError, KeyError):
                continue

        stats[run_desc]['nbr'] = len(runs)
        split = re.split(r'(_|^)([a-z]+)', run_desc)
        new_entries = {split[i-1]: split[i] for i in range(3, len(split), 3)}
        stats[run_desc].update(new_entries)

    fieldnames = sorted(list(set([item for sublist in stats.values() for item in sublist])))
    score_fields = move_to_front([f for f in fieldnames if f.endswith('_mean') or f.endswith('_std')], ['steps_train_mean', 'steps_train_std', 'time_s_mean', 'time_s_std'])
    other_fields = [f for f in fieldnames if not (f.endswith('_mean') or f.endswith('_std'))]
    fieldnames = move_to_front(score_fields + other_fields, ['te', 'sp', 'data'])
    with open('%s.merged.tsv' % out, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(stats.values())


if __name__ == '__main__':
    plac.call(load_and_merge_scores)
    print('done')
