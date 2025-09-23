import torch

'''
Taken directly from pytorch-cortex: https://github.com/prescient-design/cortex/blob/main/cortex/model/leaf/_regressor_leaf.py
'''

def diag_gaussian_cumulant(canon_param):
    res = -1.0 * canon_param[0].pow(2) / (4 * canon_param[1]) - 0.5 * (-2.0 * canon_param[1]).log()
    return res

def diag_natural_gaussian_kl_divergence(canon_param_p, canon_param_q):
    # https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf
    var_p = -0.5 / canon_param_p[1]
    mean_p = canon_param_p[0] * var_p.clamp_min(1e-4)

    exp_suff_stat = torch.stack(
        [
            mean_p,
            var_p + mean_p.pow(2),
        ]
    )

    term_1 = ((canon_param_p - canon_param_q) * exp_suff_stat).sum(0)
    term_2 = -1.0 * diag_gaussian_cumulant(canon_param_p)
    term_3 = diag_gaussian_cumulant(canon_param_q)
    return term_1 + term_2 + term_3