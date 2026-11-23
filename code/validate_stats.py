"""Independent checks of the statistical procedures used in 'No Universal AI Dominance'.
Re-implements each procedure in a different way and compares with analysis.py / results.json."""
import json, itertools, math
import numpy as np
from scipy import stats
from paths import RESULTS_JSON, VALIDATION_JSON
R = json.load(open(RESULTS_JSON)); R6 = R["rev6"]
rng = np.random.default_rng(2026)
report = {}

# 1. Kendall's W: compare with the Friedman statistic (tie-corrected), W = chi2_F / (k (n - 1))
M = R["matrix"]; dims = list(M); labs = list(M[dims[0]])
X = np.array([[M[d][l] for l in labs] for d in dims])          # k raters (dimensions) x n objects (vendors)
chi2, p_as = stats.friedmanchisquare(*[X[:, j] for j in range(X.shape[1])])
k, n = X.shape
report["kendall_W"] = dict(script=R["kendall_primary"]["W"], via_friedman=round(chi2/(k*(n - 1)), 3), p_perm=R["kendall_primary"]["p_perm"], p_asymptotic=round(float(p_as), 4))

# 2. Pooled shortfall test: all vendors, larger simulation, different seed
def T(sh, tau): return sum(max(0.0, math.log(tau/s)) for s in sh)
def null_T(sigma, rho, kk=5, nn=4_000_000):
    e = sigma*(math.sqrt(rho)*rng.standard_normal((nn, 1)) + math.sqrt(1 - rho)*rng.standard_normal((nn, kk)))
    return np.sort(np.maximum(0, -e).sum(1))
nulls = {(0.35, 0.0): null_T(0.35, 0.0), (0.35, 0.5): null_T(0.35, 0.5)}
def pval(t, key): a = nulls[key]; return float(1 - np.searchsorted(a, t - 1e-12, side="left")/len(a))
prim = {l: [M[d][l] for d in dims] for l in labs}
fav = {l: list(v.values()) for l, v in R["favourable_matrix"].items()}
pv = {}
for name, mat in (("primary", prim), ("favourable", fav)):
    pv[name] = {}
    for tau in (50, 40, 30):
        per = {l: (round(pval(T(sh, tau), (0.35, 0.0)), 5), round(pval(T(sh, tau), (0.35, 0.5)), 5)) for l, sh in mat.items()}
        worst = max(per, key=lambda l: per[l][0])
        pv[name][tau] = dict(max_vendor=worst, p_indep=per[worst][0], p_corr=per[worst][1], second=sorted(per.values())[-2][0])
report["pooled_all_vendors"] = pv

# 3. Size of the pooled test at the least favourable point and inside H0 (should be <= alpha)
def size(true_share, tau, sigma, rho, reps=200_000):
    e = sigma*(math.sqrt(rho)*rng.standard_normal((reps, 1)) + math.sqrt(1 - rho)*rng.standard_normal((reps, 5)))
    obs = np.minimum(true_share*np.exp(e), 100.0)
    t = np.maximum(0, np.log(tau/obs)).sum(1)
    crit = np.quantile(nulls[(sigma, rho)], 0.95)
    return round(float((t > crit).mean()), 4)
report["pooled_size"] = dict(at_boundary_indep=size(50, 50, 0.35, 0.0), at_boundary_corr=size(50, 50, 0.35, 0.5),
                             interior_60_indep=size(60, 50, 0.35, 0.0), mixed_boundary=None)
e = 0.35*rng.standard_normal((200_000, 5)); true = np.array([50, 50, 80, 80, 80.0])
t = np.maximum(0, np.log(50/np.minimum(true*np.exp(e), 100))).sum(1)
report["pooled_size"]["mixed_boundary"] = round(float((t > np.quantile(nulls[(0.35, 0.0)], 0.95)).mean()), 4)
# power: a vendor like OpenAI's primary row taken as the truth
t = np.maximum(0, np.log(50/np.minimum(np.array(prim["OpenAI"])*np.exp(e), 100))).sum(1)
report["pooled_power_if_primary_true"] = round(float((t > np.quantile(nulls[(0.35, 0.0)], 0.95)).mean()), 3)

# 4. Error rate of rule 2 (kappa* = 2): formula against simulation
obs = 50*np.exp(0.35*rng.standard_normal((1_000_000, 5)))
report["rule2_size"] = dict(formula=R6["error_rates"]["size_rule_kstar2"][1], simulated=round(float((obs.min(1) < 25).mean()), 4))

# 5. HHI upper bound: brute-force check that no admissible split of the residual exceeds it
def hhi_hi(named, total=100.0):
    s = np.array(named, float); resid = total - s.sum(); m = s.min(); kk = int(resid // m); rem = resid - kk*m
    return (s**2).sum() + kk*m*m + rem*rem
named = [26, 19, 16]; best = 0
for _ in range(200_000):
    parts = rng.dirichlet(np.ones(rng.integers(3, 12)))*39
    if parts.max() <= 16: best = max(best, (parts**2).sum())
report["hhi_bound"] = dict(upper=round(hhi_hi(named)), brute_force_max=round(sum(x*x for x in named) + best))

# 6. Instability index with the unlisted residual treated as one entity (shares then sum to 100)
P = R6["panel"]
def instab(series):
    A = np.array(list(series.values())); res = 100 - A.sum(0); A2 = np.vstack([A, res])
    d_named = 0.5*np.abs(np.diff(A, axis=1)).sum(0); d_full = 0.5*np.abs(np.diff(A2, axis=1)).sum(0)
    return dict(named_mean=round(float(d_named.mean()), 2), full_mean=round(float(d_full.mean()), 2), full_max=round(float(d_full.max()), 2),
                residual_range=(round(float(res.min()), 1), round(float(res.max()), 1)))
report["instability"] = dict(web=instab(P["web_series"]), dev=instab(P["dev_series"]))

# 7. Spearman correlations of the open-weight layers: exact permutation p-values (n = 6)
O = R["open_segment"]["matrix"]; fam = R["open_segment"]["concordance"]["families"]
def exact_spearman(x, y):
    r0 = stats.spearmanr(x, y)[0]; cnt = tot = 0
    for perm in itertools.permutations(range(len(y))):
        r = stats.spearmanr(x, [y[i] for i in perm])[0]; tot += 1; cnt += abs(r) >= abs(r0) - 1e-12
    return round(float(r0), 2), round(cnt/tot, 3)
dl = [O["downloads"][f] for f in fam]; gw = [O["gateway"][f] for f in fam]; en = [O["enterprise"][f] for f in fam]
report["spearman_exact"] = dict(dl_gw=exact_spearman(dl, gw), dl_ent=exact_spearman(dl, en), gw_ent=exact_spearman(gw, en))

# 8. Partial-ownership HHI: general MHHI formula against the closed form beta*s_j*s_k (silent interest)
def mhhi_general(s, beta, gamma):       # beta[i][j] financial share of owner i in firm j; gamma control share
    s = np.array(s, float); nf = len(s); tot = 0
    for j in range(nf):
        for kx in range(nf):
            num = sum(gamma[i][j]*beta[i][kx] for i in range(len(beta))); den = sum(gamma[i][j]*beta[i][j] for i in range(len(beta)))
            tot += s[j]*s[kx]*num/den
    return tot
s3 = [24.65, 11.65, 63.7]; b = 0.137       # Google, Anthropic, rest (as one independent firm, for the algebra only)
beta = [[1, b, 0], [0, 1 - b, 0], [0, 0, 1]]; gamma = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
report["mhhi"] = dict(general_delta=round(mhhi_general(s3, beta, gamma) - sum(x*x for x in s3), 2), closed_form=round(b*24.65*11.65, 2))

# 9. Monte Carlo error of the smallest reported p-value
report["mc_error_p_primary"] = dict(p=pv["primary"][50]["p_indep"], se=round(math.sqrt(2e-4*(1 - 2e-4)/4e6), 6))
json.dump(report, open(VALIDATION_JSON, "w"), indent=1)
print(json.dumps(report, indent=1))

# 10. Exclusivity bounds: re-derive the analytic interval by linear programming over
#     user types (which subset of brands each user holds). Confirms both bounds are sharp.
import itertools as _it
from scipy.optimize import linprog as _lp
_B={'ChatGPT':34.0,'Gemini':24.0,'Copilot':11.0,'Claude':7.0}; _ANY=51.0; _LEAD='ChatGPT'
_brands=list(_B); _types=[t for r in range(1,len(_brands)+1) for t in _it.combinations(_brands,r)]
def _feas(e):
    A=[[1.0]*len(_types)]+[[1.0 if k in t else 0.0 for t in _types] for k in _brands] \
      +[[1.0 if t==(_LEAD,) else 0.0 for t in _types]]
    rhs=[_ANY]+[_B[k] for k in _brands]+[e]
    return _lp([0]*len(_types),A_eq=np.array(A),b_eq=np.array(rhs),
               bounds=[(0,None)]*len(_types),method="highs").success
_ex=R["multihoming"]["exclusivity"]
_lo,_hi=_ex["excl_lo_pop"],_ex["excl_hi_pop"]
report["exclusivity_bounds"]=dict(
    analytic=[_lo,_hi],
    lo_sharp=(not _feas(_lo-0.1)) and _feas(_lo),      # just below infeasible, at bound feasible
    hi_sharp=_feas(_hi) and (not _feas(_hi+0.1)),
    interior_feasible=_feas((_lo+_hi)/2))
json.dump(report, open(VALIDATION_JSON, "w"), indent=1)
print("exclusivity bounds:", json.dumps(report["exclusivity_bounds"]))
