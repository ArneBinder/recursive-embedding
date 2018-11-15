from os.path import join
import subprocess

"""
Assume two relation type prediction (r1, r2) with e1 -> r0 -> e2 and e2 -> r1 -> e1.
Artificial relation types are suffixed with "/REV" to indicate reverted relations. But there is _no_ "Other/REV"!
"""

def load_values(fn):
    lines = [l[:-1].split('/')[2:] for l in open(fn).readlines()]
    return [(lines[i + 1], lines[i]) for i in range(0, len(lines), 2)]


def get_forward_indices(lines):
    res = []
    for i, (l1, l2) in enumerate(lines):
        assert len(l1) == 1 or len(l2) == 1, 'two REV found: %i' % i
        assert not (len(l1) == 1 and len(l2) == 1) or (
                l1 == ['Other'] and l2 == ['Other']), 'both should be Other: %i' % i
        if len(l1) > 1:
            res.append(1)
        elif len(l2) > 1:
            res.append(0)
        else:
            res.append(2)
    return res


def get_entry(t, idx):
    _t = t[idx]
    if _t[0] == 'Other':
        return 'Other'
    _dir = '(e1,e2)' if (idx == 0 and len(_t) == 1) or (idx == 1 and len(_t) == 2) else '(e2,e1)'
    return _t[0] + _dir


def create_eval_lines(lines, indices, handle_other=None):
    res = []
    for i, l in enumerate(lines):
        idx = indices[i]
        if idx < 2:
            res.append(get_entry(l, idx))
        else:
            if handle_other is not None:
                if handle_other == 'best':
                    if l[0][0] == 'Other' or l[1][0] == 'Other':
                        res.append('Other')
                    else:
                        res.append(get_entry(l, 0))
                elif handle_other == 'worst':
                    if l[0][0] != 'Other':
                        res.append(get_entry(l, 0))
                    elif l[1][0] != 'Other':
                        res.append(get_entry(l, 1))
                    else:
                        res.append('Other')
    return res


def dump(lines, fn):
    open(fn, 'w').writelines(('%i\t%s\n' % (i, l) for i, l in enumerate(lines)))


def eval(fn_predicted_tsv, fn_gold_tsv, script_dir):
    check_script = 'perl %s %s %s' % (
    join(script_dir, 'semeval2010_task8_scorer-v1.2.pl'), fn_predicted_tsv, fn_gold_tsv)
    perl_result = subprocess.check_output(check_script, shell=True)
    last_line = perl_result.split('\n')[-2]
    score_str = last_line.replace(
        '<<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = ',
        '').replace('% >>>', '')
    f1 = float(score_str) / 100
    return f1


def main(p='/mnt/DATA/ML/training/supervised/log/DEBUG/SEMEVAL/REROOT_rel/DEBUG_rev/avfFALSE_bs100_bRELATION_clp5.0_cmTREE_cntxt0_dfidx0_dtFALSE_fc0_kp0.9_kpb1.0_kpn1.0_leaffc0_lr0.003_lc-1_dpth7_mtREROOT_n_ns8_nfvFALSE_rootfc0_sl1000_st150_tkR-T_dataMERGED_teHTUBATCHEDHEADREDUCESUMMAPGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/'):

    l_g = load_values(join(p, 'values_gold_strings.txt'))
    l_m = load_values(join(p, 'values_max_strings.txt'))
    f_g = get_forward_indices(l_g)

    r_g = create_eval_lines(l_g, f_g)
    r_m = create_eval_lines(l_m, f_g)

    r_g_best = create_eval_lines(l_g, f_g, handle_other='best')
    r_m_best = create_eval_lines(l_m, f_g, handle_other='best')

    r_g_worst = create_eval_lines(l_g, f_g, handle_other='worst')
    r_m_worst = create_eval_lines(l_m, f_g, handle_other='worst')

    assert r_g_best == r_g_worst, 'gold best / worst do not match'

    dump(r_g, join(p, 'v_g_none.txt'))
    dump(r_m, join(p, 'v_m_none.txt'))
    dump(r_g_best, join(p, 'v_g_best.txt'))
    dump(r_m_best, join(p, 'v_m_best.txt'))
    dump(r_m_worst, join(p, 'v_m_worst.txt'))

    f_none = eval(join(p, 'v_m_none.txt'), join(p, 'v_g_none.txt'), p)
    f_best = eval(join(p, 'v_m_best.txt'), join(p, 'v_g_best.txt'), p)
    f_worst = eval(join(p, 'v_m_worst.txt'), join(p, 'v_g_best.txt'), p)
    print('f_none: %f' % f_none)
    print('f_best: %f' % f_best)
    print('f_worst: %f' % f_worst)
    print('done')


if __name__ == '__main__':
    import plac; plac.call(main)
