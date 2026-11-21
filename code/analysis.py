"""Reproducible computations for 'No Universal AI Dominance'.
All inputs are published shares transcribed from the cited sources (see paper, Table 1)."""
import json, itertools, random
import numpy as np
from paths import RESULTS_JSON

TAU = 50.0

def hhi_bounds(named, total=100.0):
    """named: dict of shares in %, residual = total - sum(named).
    lower: residual atomistic; upper: residual split in blocks no larger than smallest named share."""
    s = np.array(list(named.values()), float)
    resid = max(total - s.sum(), 0.0)
    lo = float((s**2).sum())
    m = s.min()
    k = int(resid // m); rem = resid - k*m
    hi = lo + k*m*m + rem*rem
    return round(lo), round(hi), round(resid, 2)

def cr(named, k):
    return round(sum(sorted(named.values(), reverse=True)[:k]), 1)

def hill(named, q):
    """Hill number of order q: the number of equally sized firms that would give
    the same concentration. The family spans the usual indices, so reporting a
    profile over q shows how far a verdict depends on how heavily the leader is
    weighted: q=0 counts firms, q=1 is exp(Shannon entropy), q=2 is 1/HHI and
    q=inf is 1/s_(1). Computed on the named shares renormalised to 100, which is
    the concentration-maximising convention: the unattributed residual can only
    add competitors, never remove them."""
    p = np.array([v for v in named.values() if v > 0], float)
    p = p/p.sum()
    if np.isinf(q):          return float(1.0/p.max())
    if abs(q - 1.0) < 1e-12: return float(np.exp(-(p*np.log(p)).sum()))
    return float(((p**q).sum())**(1.0/(1.0 - q)))

# ---------------- datasets (shares in %) ----------------
D = {}
D["A  Consumer web visits (Similarweb, 7 assistants, Oct 2026)"] = dict(OpenAI=56.6, Google=24.7, Anthropic=11.7, DeepSeek=3.5, xAI=1.9, Perplexity=0.8, Microsoft=0.7)
mau = dict(Meta=1200, OpenAI=1000, Google=1000, ByteDance=345, Alibaba=320, Baidu=220, DeepSeek=145)
mau_total = sum(mau.values()) + 559
D["B  Monthly active users (Counterpoint, Apr 2026)"] = {k: 100*v/mau_total for k, v in mau.items()}
D["C  LLM revenue (Counterpoint, Q1 2026)"] = dict(Anthropic=31.4, OpenAI=29.0, Google=12.1, Microsoft=7.2, Tencent=4.8, Baidu=3.6, Alibaba=2.9, Meta=1.4, xAI=1.4, Perplexity=1.0, DeepSeek=0.2, ByteDance=0.1)
D["D  Developer gateway requests (OpenRouter, wk of 9 Nov 2026)"] = dict(DeepSeek=31.8, OpenAI=19.0, Google=15.1, Zai=7.0, Anthropic=6.3, Alibaba=6.2, Mistral=2.6, Tencent=2.3, NVIDIA=1.6)
D["E  Enterprise LLM API spend, US (Menlo, 2026)"] = dict(Anthropic=55.0, OpenAI=31.0, Google=9.0)
D["D' Developer tokens excl. BYOK (AIMultiple/OpenRouter, Oct 2026)"] = dict(Google=30.39, OpenAI=22.46, Alibaba=10.11, Anthropic=9.80, Xiaomi=8.12, DeepSeek=5.80, Moonshot=2.26, Zhipu=2.06, Meta=1.80, MiniMax=0.77)
D["E' Business token spend, US (Ramp, Oct 2026)"] = dict(Anthropic=49.3, OpenAI=46.1, xAI=2.4, Google=1.7)
D["F  App+web audience (Sensor Tower, May 2026)"] = dict(OpenAI=46.4, Google=27.7, Anthropic=10.3)
cn = dict(ByteDance=419.08, Alibaba=158.17, DeepSeek=122.65, Tencent=37.33, Ant=34.79)
D["G  China AI-native app MAU, top 5 (QuestMobile, Sep 2026)"] = {k: 100*v/sum(cn.values()) for k, v in cn.items()}

out = {"concentration": []}
for name, d in D.items():
    lo, hi, resid = hhi_bounds(d)
    leader = max(d, key=d.get); srt = sorted(d.values(), reverse=True)
    out["concentration"].append(dict(metric=name, leader=leader, s1=round(d[leader], 1), s2=round(srt[1], 1),
        lead_ratio=round(srt[0]/srt[1], 2), cr4=cr(d, 4), hhi_lo=lo, hhi_hi=hi, residual=resid,
        dominant50=d[leader] >= 50, dominant40=d[leader] >= 40,
        # 1/HHI restated as a count of equally sized firms; the bounds invert the HHI bounds
        eff_n_lo=round(10000.0/hi, 2), eff_n_hi=round(10000.0/lo, 2),
        hill_q0=round(hill(d, 0), 2), hill_q1=round(hill(d, 1), 2),
        hill_q2=round(hill(d, 2), 2), hill_qinf=round(hill(d, float("inf")), 2)))

# ---------------- cross-dimension vendor matrix (primary: 5 global dimensions) ----------------
labs = ["OpenAI", "Google", "Anthropic", "DeepSeek", "Meta", "Alibaba"]
M = {
 "A web visits":   dict(OpenAI=56.23, Google=24.65, Anthropic=11.65, DeepSeek=3.31, Meta=0.50, Alibaba=0.31),   # AIMultiple by-lab (Similarweb), Oct 2026
 "B MAU":          dict(OpenAI=100*1000/mau_total, Google=100*1000/mau_total, Anthropic=100*134/mau_total, DeepSeek=100*145/mau_total, Meta=100*1200/mau_total, Alibaba=100*320/mau_total),
 "C revenue":      dict(OpenAI=29.0, Google=12.1, Anthropic=31.4, DeepSeek=0.2, Meta=1.4, Alibaba=2.9),
 "D gateway req.": dict(OpenAI=19.0, Google=15.1, Anthropic=6.3, DeepSeek=31.8, Meta=0.8, Alibaba=6.2),          # Meta unlisted (<1.6): midpoint 0.8
 "E enterprise":   dict(OpenAI=31.0, Google=9.0, Anthropic=55.0, DeepSeek=4/3, Meta=4/3, Alibaba=4/3),           # open-source total 4% split evenly (tie)
}
out["matrix"] = {m: {l: round(v[l], 1) for l in labs} for m, v in M.items()}
out["leaders_primary"] = {m: max(v, key=v.get) for m, v in M.items()}
out["robustness"] = []
for l in labs:
    vals = {m: M[m][l] for m in M}
    mmin = min(vals, key=vals.get)
    out["robustness"].append(dict(lab=l, max_share=round(max(vals.values()), 1), max_dim=max(vals, key=vals.get),
        min_share=round(vals[mmin], 1), min_dim=mmin, kappa=round(TAU/vals[mmin], 1),
        dims_ge50=sum(v >= 50 for v in vals.values()), dims_ge40=sum(v >= 40 for v in vals.values()),
        dims_ge30=sum(v >= 30 for v in vals.values()), dims_led=sum(out["leaders_primary"][m] == l for m in M)))

def rankdata_desc(x):
    from scipy.stats import rankdata
    return rankdata([-v for v in x], method="average")

def kendall_w(R):
    """R: k raters x n objects rank matrix (ties allowed, corrected)."""
    k, n = R.shape
    Rj = R.sum(0); S = ((Rj - Rj.mean())**2).sum()
    T = 0.0
    for row in R:
        _, cnt = np.unique(row, return_counts=True)
        T += (cnt**3 - cnt).sum()
    return 12*S / (k*k*(n**3 - n) - k*T)

def w_test(Mx, labs, nperm=200000, seed=7):
    R = np.array([rankdata_desc([Mx[m][l] for l in labs]) for m in Mx])
    w = kendall_w(R)
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(nperm):
        Rp = np.array([rng.permutation(r) for r in R])
        if kendall_w(Rp) >= w - 1e-12: cnt += 1
    return round(float(w), 3), (cnt + 1)/(nperm + 1), R

w6, p6, R6 = w_test(M, labs)
out["kendall_primary"] = dict(n_labs=6, k_dims=5, W=w6, p_perm=round(p6, 4), mean_rank={l: round(float(r), 2) for l, r in zip(labs, R6.mean(0))})
labs4 = ["OpenAI", "Google", "Anthropic", "DeepSeek"]
w4, p4, R4 = w_test({m: {l: M[m][l] for l in labs4} for m in M}, labs4)
out["kendall_big4"] = dict(n_labs=4, k_dims=5, W=w4, p_perm=round(p4, 4), mean_rank={l: round(float(r), 2) for l, r in zip(labs4, R4.mean(0))},
                           worst_rank={l: float(r) for l, r in zip(labs4, R4.max(0))})
# extended: add secondary metrics D', E', F for the four labs
M8 = {m: {l: M[m][l] for l in labs4} for m in M}
M8["D' tokens excl BYOK"] = dict(OpenAI=22.46, Google=30.39, Anthropic=9.80, DeepSeek=5.80)
M8["E' Ramp token spend"] = dict(OpenAI=46.1, Google=1.7, Anthropic=49.3, DeepSeek=0.0)
M8["F audience"] = dict(OpenAI=46.4, Google=27.7, Anthropic=10.3, DeepSeek=2.5)   # DeepSeek "<5%": midpoint
w8, p8, R8 = w_test(M8, labs4)
out["kendall_ext"] = dict(n_labs=4, k_metrics=8, W=w8, p_perm=round(p8, 4), mean_rank={l: round(float(r), 2) for l, r in zip(labs4, R8.mean(0))},
                          leaders={m: max(v, key=v.get) for m, v in M8.items()})

# ---------------- paradigm level: open-weight share by metric ----------------
P = [("Gateway tokens, OpenRouter (Sep 2026)", 72.0), ("Gateway tokens, Vercel (Sep 2026)", 71.5),
     ("Gateway spend, OpenRouter (Sep 2026)", 27.0), ("Gateway spend, Vercel (Sep 2026)", 23.0),
     ("Enterprise usage, US (Menlo 2026)", 19.0), ("Enterprise metered consumption (Futurum, Oct 2026)", 5.0),
     ("Enterprise spend, US (Menlo 2026)", 4.0), ("Consumer web visits, open-weight lab (Oct 2026)", 3.5)]
out["paradigm"] = dict(metrics=P, open_ge50=sum(v >= 50 for _, v in P), closed_ge50=sum(100 - v >= 50 for _, v in P),
                       min_open=min(v for _, v in P), max_open=max(v for _, v in P))

# ---------------- origin level: US- vs China-origin labs ----------------
us_rev = 31.4+29.0+12.1+7.2+1.4+1.4+1.0; cn_rev = 4.8+3.6+2.9+0.2+0.1
us_mau = 100*(1200+1000+1000)/mau_total; cn_mau = 100*(345+320+220+145)/mau_total
O = [("Consumer web visits (AIMultiple, Oct 2026)", 95.73, 4.21), ("Enterprise API spend, US (Menlo)", 95.0, None),
     ("Revenue (Counterpoint Q1 2026)", round(us_rev, 1), round(cn_rev, 1)), ("MAU (Counterpoint Apr 2026)", round(us_mau, 1), round(cn_mau, 1)),
     ("Dev tokens excl. BYOK (AIMultiple, Oct 2026)", 65.30, 29.15), ("Dev tokens excl. BYOK (AIMultiple, Jul 2026)", 21.05, 76.99),
     ("HF downloads (Feb 2025-Feb 2026)", 36.5, 41.0), ("Gateway tokens (OpenRouter, Sep 2026)", None, 69.0)]
out["origin"] = O

# ---------------- user geography: assistant shares by country ----------------
cols = ["ChatGPT", "Gemini", "Claude", "DeepSeek", "Grok", "Perplexity", "Copilot"]
C = {"United States":[62.9,16.5,14.8,1.0,2.6,1.1,1.1], "India":[64.1,20.2,11.7,1.1,1.7,0.9,0.2], "Japan":[55.6,35.5,6.3,0.1,1.1,0.4,1.0],
 "Brazil":[65.1,26.3,4.7,1.6,1.5,0.4,0.5], "Indonesia":[49.4,34.8,11.0,2.2,1.7,0.8,0.2], "United Kingdom":[68.2,14.3,13.0,1.0,1.5,0.9,1.1],
 "South Korea":[56.5,30.6,9.8,0.4,2.0,0.5,0.2], "Germany":[62.6,21.5,9.8,1.9,1.9,1.6,0.7], "France":[62.5,21.4,12.0,0.9,1.1,1.3,0.7],
 "Canada":[64.3,19.3,11.6,1.1,1.8,1.1,0.8], "Vietnam":[46.8,43.5,3.4,2.6,3.3,0.1,0.2], "Mexico":[47.1,39.2,7.3,2.6,2.5,0.6,0.7],
 "China":[17.7,5.5,1.3,73.2,1.8,0.5,0.1], "Philippines":[62.2,25.7,8.4,1.3,0.9,0.8,0.7], "Spain":[55.0,29.7,10.0,1.7,1.5,1.1,1.0],
 "Italy":[43.5,44.8,7.8,1.2,0.8,0.9,1.0], "Australia":[66.6,12.5,15.5,0.8,1.9,1.0,1.7], "Turkey":[48.4,43.0,5.8,1.0,1.3,0.2,0.2],
 "Russia":[43.6,13.7,4.5,32.1,2.3,3.6,0.2]}
geo = []
for c, v in C.items():
    v = np.array(v); i = int(v.argmax())
    geo.append(dict(country=c, leader=cols[i], s1=float(v[i]), s2=float(np.sort(v)[-2]), hhi=round(float((v**2).sum())), sum=round(float(v.sum()), 1)))
out["geo"] = geo
out["geo_summary"] = dict(n=len(geo), leaders={k: sum(g["leader"] == k for g in geo) for k in cols if any(g["leader"] == k for g in geo)},
    chatgpt_ge50=sum(g["leader"] == "ChatGPT" and g["s1"] >= 50 for g in geo), any_ge50=sum(g["s1"] >= 50 for g in geo),
    hhi_min=min(g["hhi"] for g in geo), hhi_max=max(g["hhi"] for g in geo), hhi_median=float(np.median([g["hhi"] for g in geo])),
    close_lead=[g["country"] for g in geo if g["s1"]/g["s2"] < 1.2])

# ---------------- scale and reconciliation arithmetic ----------------
or_day = 128.4/7; goog_day = 3200/30; doubao_day = 180.0
out["scale"] = dict(openrouter_T_per_day=round(or_day, 1), google_T_per_day=round(goog_day, 1), doubao_T_per_day=doubao_day,
    openrouter_vs_sum_pct=round(100*or_day/(goog_day+doubao_day), 1), openrouter_vs_doubao_pct=round(100*or_day/doubao_day, 1),
    google_api_T_per_day=round(32e9*1440/1e12, 1))
wap = 5.3e9
out["dedup"] = dict(mau_sum_M=mau_total, microsoft_users_B=round(0.188*wap/1e9, 2), ratio_cnnic_ms_china=round(50/17.5, 2),
    ratio_survey_ms_us=round(49/33.0, 2), india_share_chatgpt_pct=round(100*100/900, 1))
# capability benchmarks: top-10 slots by lab
out["capability"] = dict(benchlm_top10=dict(OpenAI=5, Google=3, Anthropic=2), hle_top10=dict(Anthropic=7, OpenAI=1, Moonshot=1, DeepSeek=1),
                         benchlm_open_in_top25=2, hf_collection_qwen_org=9, hf_collection_n=40)
json.dump(out, open(RESULTS_JSON, "w"), indent=1)
print(json.dumps(out, indent=1))

# ---------------- sensitivity: matrix most favourable to H0 (max over alternative sources per dimension) ----------------
FAV = {
 "OpenAI":    dict(A=max(56.6, 56.23, 79.4), B=max(100*1000/mau_total, 46.4), C=29.0, D=max(19.0, 22.46), E=max(31.0, 46.1, 38.5, 56.0)),   # E: Menlo, Ramp, Moss, a16z
 "Google":    dict(A=max(24.7, 24.65), B=max(100*1000/mau_total, 27.7), C=12.1, D=max(15.1, 30.39), E=max(9.0, 1.7)),
 "Anthropic": dict(A=max(11.7, 11.65), B=max(100*134/mau_total, 10.3), C=31.4, D=max(6.3, 9.80), E=max(55.0, 49.3, 52.8, 58.7)),         # E: Menlo, Ramp Oct, Moss, Ramp Nov
 "DeepSeek":  dict(A=max(3.5, 3.31), B=100*145/mau_total, C=0.2, D=max(31.8, 5.80), E=4.0),
 "Meta":      dict(A=0.50, B=100*1200/mau_total, C=1.4, D=1.80, E=4.0),
}
fav = []
for lab, v in FAV.items():
    srt = sorted(v.items(), key=lambda kv: kv[1])
    row = dict(lab=lab, **{k: round(x, 1) for k, x in v.items()}, weakest=srt[0][0], second_weakest=srt[1][0])
    for tau in (50, 40, 30):
        row[f"kappa{tau}"] = round(tau/srt[0][1], 2)
        row[f"fail{tau}"] = sum(x < tau for x in v.values())
        row[f"kappa2nd{tau}"] = round(tau/srt[1][1], 2)
    fav.append(row)
out["favourable"] = fav
json.dump(out, open(RESULTS_JSON, "w"), indent=1)
print(json.dumps(fav, indent=1))

# ================= additions (revision 2) =================
# ---- Monte Carlo measurement-error simulation ----
def mc_h0(matrix, taus=(50, 40, 30), sigmas=None, n=100000, rho=0.0, seed=11):
    """P(exists vendor with all dimensions >= tau) when each share is multiplied by exp(eps),
    eps ~ N(0, sigma^2); rho = share of error variance common to all dimensions of a vendor."""
    rng = np.random.default_rng(seed)
    S = np.array([list(v.values()) for v in matrix.values()], float)      # vendors x dims
    sigmas = np.round(np.linspace(0, 1.0, 21), 2) if sigmas is None else sigmas
    res = {t: [] for t in taus}
    for sg in sigmas:
        zc = rng.standard_normal((n, S.shape[0], 1)); zi = rng.standard_normal((n,) + S.shape)
        eps = sg*(np.sqrt(rho)*zc + np.sqrt(1 - rho)*zi)
        X = np.minimum(S[None]*np.exp(eps), 100.0)
        mn = X.min(axis=2)                                                   # weakest dimension per vendor
        for t in taus:
            res[t].append(float((mn >= t).any(axis=1).mean()))
    return [float(s) for s in sigmas], res

FAVM = {k: {d: v[d] for d in "ABCDE"} for k, v in FAV.items()}
sig, r_ind = mc_h0(FAVM, rho=0.0)
_, r_cor = mc_h0(FAVM, rho=0.5)
PRIM = {l: {m: M[m][l] for m in M} for l in labs}
_, r_prim = mc_h0(PRIM, rho=0.0)
def first_cross(sig, p, level=0.05):
    for s, v in zip(sig, p):
        if v >= level: return s
    return None
i35 = sig.index(0.35)
out["mc"] = dict(sigmas=sig, n=100000,
    favourable_indep={str(t): v for t, v in r_ind.items()}, favourable_corr={str(t): v for t, v in r_cor.items()},
    primary_indep={str(t): v for t, v in r_prim.items()},
    at_sigma_035=dict(fav_indep={str(t): round(r_ind[t][i35], 4) for t in r_ind}, fav_corr={str(t): round(r_cor[t][i35], 4) for t in r_cor},
                      prim_indep={str(t): round(r_prim[t][i35], 4) for t in r_prim}),
    sigma_at_5pct=dict(fav_indep={str(t): first_cross(sig, r_ind[t]) for t in r_ind}, fav_corr={str(t): first_cross(sig, r_cor[t]) for t in r_cor}),
    factor95_at_035=round(float(np.exp(1.96*0.35)), 2))

# ---- implied closed/open price ratio from token and spend shares ----
def price_ratio(tok_open, spend_open):
    t, s = tok_open/100, spend_open/100
    return round((t/(1 - t))/(s/(1 - s)), 1)
out["price_ratio"] = dict(openrouter=price_ratio(72.0, 27.0), vercel=price_ratio(71.5, 23.0))

# ---- multi-homing lower bound (Reuters Institute 2026, weekly use, six countries) ----
brands = dict(ChatGPT=34, Gemini=24, Copilot=11, Claude=7)
mh = sum(brands.values())/51
out["multihoming"] = dict(sum_brands=sum(brands.values()), any_use=51, tools_per_user_lb=round(mh, 2))

# Exclusivity-adjusted concentration. A share counts a multi-homing user once per
# brand, so to first order the exclusive-user share is s/mh and the index scales by
# 1/mh^2. Applied only to the consumer metrics, where multi-homing is measured;
# enterprise buyers mostly single-home, so no adjustment is warranted there. The
# correction divides every vendor by the same factor, while a leader plausibly holds
# more exclusive users than a challenger, so the adjusted figure is a LOWER BOUND on
# concentration.
out["multihoming"]["adjusted_hhi"] = {
    row["metric"].split()[0]: dict(hhi=row["hhi_lo"], adjusted=round(row["hhi_lo"]/mh**2),
                                   band_before="highly" if row["hhi_lo"] > 1800 else "moderately",
                                   band_after="highly" if row["hhi_lo"]/mh**2 > 1800 else "moderately")
    for row in out["concentration"] if row["metric"].split()[0] in ("A", "F")}

# ---- adoption depends on the measure ----
out["adoption_measures"] = [
 ("Organisations reporting any AI use, global survey (AI Index, 2025)", 88.0),
 ("US businesses paying for AI, card platform (Ramp, Oct 2026)", 56.8),
 ("Individuals, weekly use, six countries (Reuters Institute, 2026)", 51.0),
 ("Individuals, China (CNNIC, mid-2026)", 50.0),
 ("Workers, work-related use, US (Fed/RPS, Nov 2025)", 41.0),
 ("Individuals 16-74, last 3 months, EU (Eurostat, 2025)", 32.7),
 ("Internet users, Brazil (TIC Domicilios, 2025)", 32.0),
 ("EU enterprises using AI (Eurostat, 2025)", 20.0),
 ("World working-age population (Microsoft, Q2 2026)", 18.8),
 ("US firms using AI in production (Census BTOS, late 2023)", 3.8)]
# ---- time series ----
out["series"] = dict(
 vercel_tokens=[("Dec 25", 7), ("Apr 26", 11), ("Jun 26", 29), ("Jul 26", 36), ("Aug 26", 56), ("Sep 26", 71.5)],
 vercel_spend=[("Jun 26", 4), ("Jul 26", 8.6), ("Aug 26", 14), ("Sep 26", 23)],
 menlo=dict(Anthropic=[(2023, 12), (2025, 40), (2026, 55)], OpenAI=[(2023, 50), (2025, 27), (2026, 31)], Google=[(2025, 21), (2026, 9)]))
out["microsoft"] = [("United Arab Emirates", 73.3, "e"), ("Singapore", 64.3, "e"), ("Ireland", 49.9, "e"), ("France", 49.6, "e"),
 ("United States", 33.0, "e"), ("Global North", 28.8, "a"), ("Japan", 24.7, "e"), ("Brazil", 20.2, "e"), ("World", 18.8, "a"),
 ("India", 18.5, "e"), ("China", 17.5, "e"), ("Global South", 16.2, "a")]
out["favourable_matrix"] = {k: {d: round(x, 1) for d, x in v.items()} for k, v in FAVM.items()}
json.dump(out, open(RESULTS_JSON, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("price_ratio", "multihoming")}, indent=1))
print({k: v for k, v in out["mc"].items() if k in ("at_sigma_035", "sigma_at_5pct", "factor95_at_035")})
for t in ("50", "40", "30"): print(t, [round(x, 3) for x in out["mc"]["favourable_indep"][t]])
for t in ("50", "40", "30"): print("corr", t, [round(x, 3) for x in out["mc"]["favourable_corr"][t]])


# ================= additions (revision 3) =================
# ---- access counterfactual: DeepSeek share by market access ----
ms_ds = {"China": 89, "Belarus": 56, "Cuba": 49, "Russia": 43}                 # Microsoft, H2 2025, "DeepSeek market share"
sw_ds = {c: v[3] for c, v in C.items()}                                         # Similarweb, Sep/Oct 2026, share among seven assistants
restricted = ["China", "Russia"]
south = ["India", "Brazil", "Indonesia", "Vietnam", "Mexico", "Philippines", "Turkey"]
north = [c for c in sw_ds if c not in restricted + south]
open_mk = [c for c in sw_ds if c not in restricted]
out["access"] = dict(microsoft=ms_ds, similarweb=sw_ds, restricted=restricted, south=south, north=north,
    median_open=float(np.median([sw_ds[c] for c in open_mk])), max_open=max(sw_ds[c] for c in open_mk), n_open=len(open_mk),
    median_south=float(np.median([sw_ds[c] for c in south])), median_north=float(np.median([sw_ds[c] for c in north])),
    leader_share_south={c: C[c][0] for c in south}, min_chatgpt_south=min(C[c][0] for c in south),
    us_origin_south={c: round(100 - C[c][3], 1) for c in south})
# ---- two trackers of the same construct (consumer chat share) ----
sw = dict(ChatGPT=56.6, Gemini=24.7, Claude=11.7, DeepSeek=3.5, Perplexity=0.8, Copilot=0.7)      # Similarweb visits, Oct 2026
sc = dict(ChatGPT=79.4, Gemini=10.9, Claude=2.57, DeepSeek=0.02, Perplexity=4.31, Copilot=2.79)   # StatCounter referrals, Aug 2026
out["trackers"] = {k: dict(similarweb=sw[k], statcounter=sc[k], ratio=round(max(sw[k], sc[k])/min(sw[k], sc[k]), 1)) for k in sw}
# ---- china top-five apps by paradigm of the lab ----
tot = sum(cn.values())
out["china_apps"] = dict(bytedance=round(100*cn["ByteDance"]/tot, 1), open_weight_labs=round(100*(cn["Alibaba"] + cn["DeepSeek"])/tot, 1))
# ---- vintage replication of the layer ordering (open-weight / China-origin share) ----
out["vintage"] = dict(y2025=dict(tokens=30, enterprise_usage=13, consumer_web=7), y2026=dict(tokens=72, enterprise_usage=19, consumer_web=4.2))
json.dump(out, open(RESULTS_JSON, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("access", "trackers", "china_apps")}, indent=1)[:2600])
print({k: v for k, v in out["mc"].items() if k in ("at_sigma_035", "sigma_at_5pct")})
print([r for r in out["favourable"] if r["lab"] == "OpenAI"])

# =====================================================================================
# Inside the open-weight segment (revision 5)
# =====================================================================================
from scipy.stats import spearmanr
OWS = {}
# -- Layer 1: Hub downloads. Share (%) of trailing-30-day downloads among the 300 most-downloaded
#    text-generation repositories on Hugging Face, attributed by architecture (VIDraft leaderboard).
dl = {"2026-09-17": {"Qwen": 55.8, "GPT/OPT": 12.1, "Llama": 6.5, "Gemma": 5.6, "DeepSeek": 4.6, "Nemotron": 2.6, "GLM": 2.1},
      "2026-11-04": {"Qwen": 50.9, "GPT/OPT": 15.4, "DeepSeek": 6.7, "Llama": 5.3, "GLM": 3.4, "Nemotron": 3.2, "Gemma": 2.5}}
dl_origin = {"2026-09-17": dict(China=66.1, US=31.5, EU=1.2, Korea=0.5, unattributed=0.7),
             "2026-11-04": dict(China=64.1, US=30.0, EU=1.6, Korea=0.5, unattributed=3.9)}
#    ATOM report: cumulative Hub downloads of ~1.5K mainline open language models, Nov 2023 - Mar 2026 (millions)
atom_region = dict(China=1150.0, US=723.0, EU=163.0)
atom_family = dict(Qwen=942.1, Llama=476.0, DeepSeek=128.2, Mistral=124.3)       # DeepSeek, Mistral as of Jan 2026
atom_tot = sum(atom_region.values())
# -- Layer 2: derivatives and local inference (Hugging Face, State of Open Models, Aug 2026)
deriv = dict(Qwen=151448, Google=82506, Meta=151448/2.6, Llama=151448/4.7)       # Meta and Llama implied by the stated ratios
gguf = dict(Qwen=39.6, Gemma=20.8, Llama=7.5)                                    # GGUF downloads per month, millions
# -- Layer 3: gateway. Open-weight tokens on a large public router, 2-8 Nov 2026 (American Compute), trillions and %
gw_T = {"DeepSeek": 81.1, "Hunyuan": 18.8, "GLM": 17.9, "Nemotron": 8.16, "Kimi": 6.47, "MiniMax": 5.70, "MiMo": 3.44,
        "Qwen": 2.27, "Gemma": 0.892, "gpt-oss": 0.531, "Mistral": 0.519, "Llama": 0.097}
gw = {"DeepSeek": 53.5, "Hunyuan": 12.4, "GLM": 11.8, "Nemotron": 5.4, "Kimi": 4.3, "MiniMax": 3.8, "MiMo": 2.3,
      "Qwen": 1.5, "Gemma": 0.6, "gpt-oss": 0.4, "Mistral": 0.3, "Llama": round(100*0.097/(81.1/0.535), 2)}
lab_served = {"Hunyuan": 99, "Nemotron": 89, "MiniMax": 82, "MiMo": 70, "Qwen": 51, "Kimi": 48, "Mistral": 34, "DeepSeek": 14,
              "GLM": 8, "Gemma": 0, "gpt-oss": 0, "Llama": 0}                    # 0: the lab runs no API for the family
fam_origin = {"DeepSeek": "China", "Hunyuan": "China", "GLM": "China", "Kimi": "China", "MiniMax": "China", "MiMo": "China",
              "Qwen": "China", "Nemotron": "US", "Gemma": "US", "gpt-oss": "US", "Llama": "US", "Mistral": "EU"}
# third-party providers: (share of third-party-served tokens %, HQ as coded by the source)
prov = [("Wafer", 12.5, "US"), ("DeepInfra", 11.8, "US"), ("Together", 11.0, "US"), ("Relace", 8.6, "n/l"), ("inference.net", 7.5, "US"),
        ("DigitalOcean", 5.3, "US"), ("Morph", 5.1, "US"), ("CoreWeave", 4.6, "US"), ("Open Inference", 3.0, "US"), ("Fireworks", 2.8, "US"),
        ("Crusoe", 2.0, "US"), ("StreamLake", 1.9, "China"), ("Parasail", 1.8, "US"), ("Modal", 1.6, "US"), ("Cohere", 1.6, "US"),
        ("NovitaAI", 1.5, "US"), ("Baseten", 1.5, "US"), ("Prime Intellect", 1.5, "n/l"), ("Baidu Qianfan", 1.5, "China"),
        ("Alibaba Cloud Int.", 1.5, "Singapore"), ("Particle AI", 1.4, "US"), ("GMICloud", 1.2, "US"), ("Ionstream", 0.8, "US"),
        ("Cloudflare", 0.7, "US"), ("SiliconFlow", 0.7, "Singapore"), ("StepFun", 0.6, "n/l"), ("Makora", 0.6, "n/l"),
        ("ModelRun", 0.6, "US"), ("Venice", 0.5, "US"), ("Decart", 0.5, "US"), ("Sail Research", 0.4, "US"), ("Phala", 0.4, "US"),
        ("Nebius", 0.3, "Netherlands"), ("Reka AI", 0.3, "n/l"), ("Mistral", 0.3, "France"), ("DekaLLM", 0.3, "Indonesia"),
        ("AtlasCloud", 0.3, "US"), ("Inceptron", 0.3, "Sweden"), ("NEAR AI", 0.2, "n/l"), ("Friendli", 0.2, "US")]
T3 = 11.8/0.125                                                                  # third-party-served tokens implied by the top row
# -- Layer 4: enterprise production (Futurum, Oct 2026): % of 83 production users running each family (multiple response)
ent = dict(Gemma=55, Phi=39, Llama=35, DeepSeek=25, Nemotron=25, Mistral=24, Qwen=10, Kimi=6, GLM=1)

# ---- concentration by layer
lo, hi, resid = hhi_bounds(gw)
OWS["gateway_family"] = dict(leader="DeepSeek", s1=53.5, cr4=cr(gw, 4), hhi_lo=lo, hhi_hi=hi, residual=resid,
                             china_origin=round(sum(v for k, v in gw.items() if fam_origin[k] == "China"), 1),
                             us_origin=round(sum(v for k, v in gw.items() if fam_origin[k] == "US"), 1),
                             eu_origin=round(sum(v for k, v in gw.items() if fam_origin[k] == "EU"), 1))
OWS["downloads_family"] = {}
for d, v in dl.items():
    lo, hi, resid = hhi_bounds(v)
    OWS["downloads_family"][d] = dict(leader="Qwen", s1=v["Qwen"], cr4=cr(v, 4), hhi_lo=lo, hhi_hi=hi, residual=resid, origin=dl_origin[d])
OWS["atom"] = dict(total_M=atom_tot, region_share={k: round(100*v/atom_tot, 1) for k, v in atom_region.items()},
                   family_share={k: round(100*v/atom_tot, 1) for k, v in atom_family.items()},
                   qwen_over_llama=round(atom_family["Qwen"]/atom_family["Llama"], 2))
OWS["derivatives"] = {k: round(v) for k, v in deriv.items()}
OWS["gguf"] = dict(monthly_M=gguf, qwen_share_of_three=round(100*gguf["Qwen"]/sum(gguf.values()), 1))

# ---- who sells the tokens: family (model author) versus seller (lab API or third-party host)
lab_T = {f: gw_T[f]*lab_served[f]/100 for f in gw_T}
TOT = T3 + sum(lab_T.values())
seller = {f"{f} (lab)": 100*t/TOT for f, t in lab_T.items() if t > 0}
seller.update({p: s*T3/TOT for p, s, _ in prov})
hhi_seller = sum(v*v for v in seller.values())
top_sellers = sorted(seller.items(), key=lambda t: -t[1])[:6]
lab_hq = {"DeepSeek": "China", "Hunyuan": "China", "GLM": "China", "Kimi": "China", "MiniMax": "China", "MiMo": "China", "Qwen": "China",
          "Nemotron": "US", "Mistral": "EU"}
hq = {}
for f, t in lab_T.items():
    if t > 0: hq[lab_hq[f]] = hq.get(lab_hq[f], 0) + 100*t/TOT
for p, s, h in prov:
    key = h if h in ("US", "China", "n/l") else ("Singapore" if h == "Singapore" else "Other")
    if h == "France": key = "EU"
    hq[key] = hq.get(key, 0) + s*T3/TOT
third = {}
for p, s, h in prov: third[h] = third.get(h, 0) + s
OWS["seller"] = dict(third_party_T=round(T3, 1), lab_served_T=round(sum(lab_T.values()), 1), total_T=round(TOT, 1),
    family_sum_T=round(sum(gw_T.values()), 1), family_implied_total_T=round(81.1/0.535, 1),
    lab_served_pct=round(100*sum(lab_T.values())/TOT, 1), hhi_family=OWS["gateway_family"]["hhi_lo"], hhi_seller=round(hhi_seller),
    top_sellers=[(k, round(v, 1)) for k, v in top_sellers], largest_seller=round(max(seller.values()), 1),
    deepseek_lab_share_of_all=round(seller["DeepSeek (lab)"], 1),
    hq_all_tokens={k: round(v, 1) for k, v in sorted(hq.items(), key=lambda t: -t[1])},
    third_party_hq={k: round(v, 1) for k, v in sorted(third.items(), key=lambda t: -t[1])},
    third_party_us=round(third["US"], 1), third_party_china=round(third["China"], 1),
    third_party_china_incl_affiliates=round(third["China"] + third["Singapore"] + 1.5, 1),      # + Alibaba Cloud Int., SiliconFlow, NovitaAI
    lab_served_by_family=lab_served)

# ---- rank concordance of families across layers (six families observed in all three)
fam6 = ["Qwen", "DeepSeek", "Llama", "GLM", "Nemotron", "Gemma"]
L3 = {"Hub downloads": {f: dl["2026-11-04"][f] for f in fam6}, "Gateway tokens": {f: gw[f] for f in fam6},
      "Enterprise production": {f: ent[f] for f in fam6}}
wO, pO, RO = w_test(L3, fam6)
pairs = {}
ks = list(L3)
for i in range(3):
    for j in range(i + 1, 3):
        rho, p = spearmanr([L3[ks[i]][f] for f in fam6], [L3[ks[j]][f] for f in fam6])
        pairs[f"{ks[i]} ~ {ks[j]}"] = dict(rho=round(float(rho), 2), p=round(float(p), 3))
OWS["concordance"] = dict(families=fam6, layers=ks, W=wO, p_perm=round(pO, 4), spearman=pairs,
    ranks={k: {f: float(r) for f, r in zip(fam6, row)} for k, row in zip(ks, RO)},
    leaders={k: max(v, key=v.get) for k, v in L3.items()})
# robustness margin within the segment (share-type layers only: downloads and gateway tokens)
OWS["kappa"] = {f: dict(min_share=min(dl["2026-11-04"][f], gw[f]), max_share=max(dl["2026-11-04"][f], gw[f]),
                        kappa=round(50/min(dl["2026-11-04"][f], gw[f]), 1)) for f in fam6}
OWS["matrix"] = dict(downloads=dl["2026-11-04"], gateway=gw, enterprise=ent,
                     derivatives_k={k: round(v/1000, 1) for k, v in deriv.items()}, gguf=gguf)

# ---- how far behind: open versus closed frontier, by ruler
OWS["lag_months"] = [  # (ruler, low, high, source key)
    ("Epoch Capabilities Index, Jan 2023 - Oct 2025", 3, 3, "epoch"),
    ("Epoch Capabilities Index, since Jan 2026", 4, 4, "epoch"),
    ("Public benchmarks (17 selected, by type)", 4, 6, "epoch"),
    ("Cyber capabilities index", 6, 7, "epoch"),
    ("Furniture-assembly benchmark (visual, spatial)", 7, 7, "epoch"),
    ("Private benchmarks (17 selected, by type)", 8, 10, "epoch"),
    ("Training compute (2024 estimate, 90% CI 6-22)", 15, 15, "epoch2024"),
    ("Benchmark performance (2024 estimate, 90% CI)", 5, 22, "epoch2024")]
OWS["aa_index"] = [dict(date="Apr 2025", closed=35, open=22, closed_model="Claude 3.7 Sonnet", open_model="DeepSeek V3"),
                   dict(date="Apr 2026", closed=60, open=54, closed_model="GPT-5.5", open_model="Kimi K2.6, MiMo V2.5 Pro"),
                   dict(date="Oct 2026 (v5.0)", closed=58, open=32, closed_model="Claude Opus 5.5", open_model="DeepSeek V4.1 Flash, GLM-5.3",
                        unreleased=46, unreleased_model="Kimi K3.1 Pro")]
for r in OWS["aa_index"]: r["gap"] = r["closed"] - r["open"]
OWS["aa_price_frontier"] = dict(open_on_frontier=9, frontier_n=13)

# ---- two ecosystems: size of what is downloaded, licences, by lab and origin (Hugging Face, Jan - Aug 2026)
OWS["share_above_70B"] = [("MiniMax", 100, "China"), ("Moonshot", 88, "China"), ("DeepSeek", 55, "China"), ("Z.ai", 39, "China"),
                          ("NVIDIA", 14, "US"), ("Meta", 9, "US"), ("Google", 0, "US"), ("Microsoft", 0, "US"), ("IBM", 0, "US")]
OWS["above_70B_all"] = 3.0
OWS["licences_above_20B"] = dict(China=dict(n=178, permissive=59 + 22, apache=59, mit=22, other=100 - 81),
                                 US=dict(permissive=29, custom=41, none_declared=30))
OWS["hub"] = dict(models_M=2.96, pct_models_under_200_dl=85.6, pct_repos_for_99_2_dl=1.5, top25_from_2022=13, top25_from_2026=0,
                  top25_overlap_likes=1, qwen_2026_M=2045, moonshot_2026_M=37, qwen_over_moonshot=round(2045/37))
out["open_segment"] = OWS
json.dump(out, open(RESULTS_JSON, "w"), indent=1)
print(json.dumps({k: OWS[k] for k in ("gateway_family", "downloads_family", "atom", "seller", "concordance", "kappa", "gguf", "derivatives")}, indent=1))

# ---- lower bound on the share of China-origin family tokens sold by US-headquartered third-party hosts
cn_T = sum(t for f, t in gw_T.items() if fam_origin[f] == "China")
non_cn_third = sum(gw_T[f]*(1 - lab_served[f]/100) for f in gw_T if fam_origin[f] != "China")
us_third_T = T3*third["US"]/100
OWS["seller"]["china_family_T"] = round(cn_T, 1)
OWS["seller"]["china_family_lab_served_pct"] = round(100*sum(lab_T[f] for f in gw_T if fam_origin[f] == "China")/cn_T, 1)
OWS["seller"]["china_family_sold_by_us_hosts_min_pct"] = round(100*(us_third_T - non_cn_third)/cn_T, 1)
out["open_segment"] = OWS
json.dump(out, open(RESULTS_JSON, "w"), indent=1)
print({k: OWS["seller"][k] for k in ("china_family_T", "china_family_lab_served_pct", "china_family_sold_by_us_hosts_min_pct")})


# =====================================================================================
# Revision 6: relevant markets, second sources, durability panel, ownership blocs,
#             seller-based dimension D, common denominators, error rates of the decision rule
# =====================================================================================
from scipy.stats import norm
R6 = {}

# ---- (1) market-level readings: every available source per candidate relevant market
MK = {
 "Consumer assistants outside China": [
    ("Similarweb web visits, Oct 2026", "OpenAI", 56.6), ("StatCounter page views, Oct 2026", "OpenAI", 51.58),
    ("StatCounter page views, Aug 2026", "OpenAI", 79.4), ("Sensor Tower app and web audience, May 2026", "OpenAI", 46.4)],
 "Enterprise model spend": [
    ("Menlo survey, US, 2026", "Anthropic", 55.0), ("Ramp transactions, US, Oct 2026", "Anthropic", 49.3),
    ("Ramp transactions, US, week to 15 Nov 2026", "Anthropic", 58.7), ("Moss card spend, Europe, Jan 2024-May 2026", "Anthropic", 52.8),
    ("a16z survey, Global 2000 CIOs, Jan 2026", "OpenAI", 56.0)],
 "Developer inference through gateways": [
    ("OpenRouter requests by model author, Nov 2026", "DeepSeek", 31.8), ("OpenRouter paid tokens by author, Oct 2026", "Google", 30.39),
    ("Requests by seller (reconstructed), Nov 2026", "OpenAI", 19.0)],
 "Consumer AI apps in China": [
    ("QuestMobile top five, Sep 2026", "ByteDance", 54.3), ("AICPB five largest Chinese assistants by firm, Jun 2026", "Alibaba", 42.0),
    ("AICPB excluding Quark, Jun 2026", "ByteDance", 39.6)]}
R6["markets"] = {}
for m, rows in MK.items():
    leaders = [r[1] for r in rows]; shares = [r[2] for r in rows]
    R6["markets"][m] = dict(readings=rows, n=len(rows), distinct_leaders=sorted(set(leaders)), n_ge50=sum(x >= 50 for x in shares),
        n_ge40=sum(x >= 40 for x in shares), min=min(shares), max=max(shares),
        same_leader_ge50=max(sum(1 for l, x in zip(leaders, shares) if l == L and x >= 50) for L in set(leaders)))
aic = dict(Doubao=324.31, Qwen=251.13, Quark=159.24, DeepSeek=139.08, Yuanbao=103.92)
R6["aicpb"] = dict(doubao_of_five=round(100*aic["Doubao"]/sum(aic.values()), 1), alibaba_of_five=round(100*(aic["Qwen"] + aic["Quark"])/sum(aic.values()), 1),
                   doubao_ex_quark=round(100*aic["Doubao"]/(sum(aic.values()) - aic["Quark"]), 1))

# ---- (2) StatCounter by region (page-view based), latest month available
SC = {"Worldwide (Oct)": dict(ChatGPT=51.58, Gemini=28.86, Perplexity=15.04, Claude=2.3, Copilot=2.18, DeepSeek=0.05),
      "Africa (Oct)": dict(ChatGPT=45.93, Gemini=38.97, Perplexity=9.23, Copilot=3.08, Claude=2.71, DeepSeek=0.07),
      "Asia (Oct)": dict(ChatGPT=55.8, Gemini=24.64, Perplexity=17.04, Claude=1.7, Copilot=0.77, DeepSeek=0.04),
      "Europe (Oct)": dict(ChatGPT=43.38, Gemini=38.64, Perplexity=10.38, Copilot=4.11, Claude=3.48, DeepSeek=0.01),
      "North America (Sep)": dict(ChatGPT=74.24, Gemini=14.64, Copilot=6.27, Perplexity=3.10, Claude=1.69, DeepSeek=0.07),
      "South America (Oct)": dict(Gemini=49.26, ChatGPT=37.05, Perplexity=8.63, Claude=2.85, Copilot=2.2, DeepSeek=0.01),
      "Oceania (Jul)": dict(ChatGPT=70.2, Copilot=9.57, Gemini=9.57, Perplexity=5.42, Claude=5.24, DeepSeek=0.01)}
R6["statcounter_regions"] = {k: dict(leader=max(v, key=v.get), s1=max(v.values()), hhi=round(sum(x*x for x in v.values())), shares=v) for k, v in SC.items()}
reg = [k for k in SC if not k.startswith("World")]
R6["statcounter_summary"] = dict(regions=len(reg), chatgpt_leads=sum(max(SC[k], key=SC[k].get) == "ChatGPT" for k in reg),
    leader_ge50=sum(max(SC[k].values()) >= 50 for k in reg), world_aug=79.4, world_oct=51.58)

# ---- (3) durability: thirteen-month panel (AIMultiple; Similarweb visits and OpenRouter paid tokens by lab)
months = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
WEB = {"OpenAI": [75.05, 70.91, 66.54, 64.77, 62.78, 55.98, 54.02, 53.07, 53.06, 54.23, 55.89, 56.77, 56.23],
       "Google": [14.40, 16.46, 20.79, 23.34, 24.68, 25.33, 27.10, 27.66, 28.25, 26.73, 25.96, 25.43, 24.65],
       "Anthropic": [2.40, 2.62, 2.85, 3.01, 4.21, 6.55, 8.75, 9.59, 9.79, 10.10, 9.41, 9.71, 11.65],
       "DeepSeek": [4.15, 4.43, 4.26, 3.59, 3.27, 3.56, 3.95, 4.19, 3.92, 3.76, 3.71, 3.52, 3.31],
       "xAI": [0.01, 0.14, 0.04, 0.07, 0.07, 3.21, 2.73, 2.33, 2.09, 2.09, 2.08, 1.97, 1.89],
       "Perplexity": [2.66, 2.28, 2.15, 1.91, 1.79, 1.63, 1.52, 1.32, 1.14, 1.03, 0.94, 0.88, 0.82],
       "Meta": [0.28, 0.24, 0.26, 0.26, 0.23, 0.23, 0.30, 0.33, 0.30, 0.41, 0.56, 0.50, 0.50],
       "Qwen": [0.40, 0.49, 0.41, 0.38, 0.41, 0.53, 0.40, 0.39, 0.34, 0.36, 0.35, 0.33, 0.31]}
DEV = {"Google": [14.24, 16.70, 25.20, 33.10, 25.15, 20.94, 22.09, 26.65, 15.09, 8.97, 13.29, 19.88, 30.39],
       "OpenAI": [9.30, 12.21, 12.99, 12.47, 13.98, 13.73, 16.75, 18.54, 11.52, 7.36, 9.12, 16.74, 22.46],
       "Anthropic": [16.34, 15.30, 19.32, 19.69, 18.08, 17.90, 19.87, 21.16, 12.94, 2.69, 2.68, 1.96, 9.80],
       "Qwen": [42.55, 39.36, 21.44, 4.31, 3.64, 4.30, 5.22, 7.05, 5.33, 3.43, 3.03, 6.13, 10.11],
       "Xiaomi": [0, 0, 0.00, 0.18, 0.65, 14.40, 6.79, 2.80, 41.00, 70.34, 63.08, 42.59, 8.12],
       "DeepSeek": [0.11, 0.08, 3.89, 8.72, 9.40, 8.41, 8.75, 9.25, 5.89, 0.83, 2.80, 4.62, 5.80],
       "Moonshot": [0, 0, 0, 0, 0, 0, 0, 0, 0.71, 0.44, 1.30, 1.25, 2.26],
       "Zhipu": [11.72, 11.41, 8.04, 5.99, 0, 0, 0, 0, 0.93, 0.93, 1.02, 1.11, 2.06],
       "Meta": [1.01, 0.72, 0.96, 1.23, 1.33, 1.16, 0.99, 1.36, 1.06, 0.71, 0.79, 1.28, 1.80],
       "MiniMax": [0, 0, 0.51, 1.98, 17.44, 12.87, 12.88, 8.36, 3.26, 1.00, 1.03, 0.81, 0.77]}
ORI = {"web_US": [94.80, 92.69, 92.67, 93.38, 93.78, 92.98, 94.44, 94.31, 94.73, 94.78, 94.84, 95.26, 95.73],
       "dev_US": [45.60, 48.93, 65.22, 76.89, 67.12, 58.86, 64.93, 70.56, 41.69, 21.05, 26.60, 40.49, 65.30],
       "dev_CN": [54.39, 50.86, 33.89, 21.23, 31.17, 40.01, 33.65, 27.47, 57.14, 76.99, 72.26, 56.52, 29.15]}
def panel_stats(P):
    T = len(months); labs_ = list(P)
    lead = [max(labs_, key=lambda l: P[l][t]) for t in range(T)]
    s1 = [P[lead[t]][t] for t in range(T)]
    changes = sum(lead[t] != lead[t - 1] for t in range(1, T))
    res = [100 - sum(P[l][t] for l in labs_) for t in range(T)]              # unlisted labs, treated as one entity so that shares sum to 100
    inst = [0.5*(sum(abs(P[l][t] - P[l][t - 1]) for l in labs_) + abs(res[t] - res[t - 1])) for t in range(1, T)]
    spells, cur = [], 0
    for t in range(T):
        if s1[t] >= 50: cur += 1
        else:
            if cur: spells.append(cur)
            cur = 0
    if cur: spells.append(cur)
    by = {}
    for l in lead: by[l] = by.get(l, 0) + 1
    return dict(leaders=lead, leader_share=s1, distinct_leaders=len(set(lead)), months_led=by, leader_changes=changes,
                months_ge50=sum(x >= 50 for x in s1), months_ge40=sum(x >= 40 for x in s1), longest_spell_ge50=max(spells) if spells else 0,
                same_firm_ge50_all=len(set(l for l, x in zip(lead, s1) if x >= 50)) == 1 and all(x >= 50 for x in s1),
                instability_mean=round(float(np.mean(inst)), 2), instability_max=round(float(max(inst)), 2), hhi=[round(sum(P[l][t]**2 for l in labs_)) for t in range(T)])
R6["panel"] = dict(months=months, web=panel_stats(WEB), dev=panel_stats(DEV), web_series=WEB, dev_series=DEV, origin=ORI,
    openai_web_change=round(WEB["OpenAI"][-1] - WEB["OpenAI"][0], 2), openai_web_min=min(WEB["OpenAI"]),
    openai_web_slope_first8=round(float(np.polyfit(range(9), WEB["OpenAI"][:9], 1)[0]), 2), openai_web_slope_last5=round(float(np.polyfit(range(5), WEB["OpenAI"][8:], 1)[0]), 2),
    dev_origin_majority_flips=sum((ORI["dev_US"][t] >= 50) != (ORI["dev_US"][t - 1] >= 50) for t in range(1, 13)),
    dev_china_ge50=sum(x >= 50 for x in ORI["dev_CN"]), dev_us_ge50=sum(x >= 50 for x in ORI["dev_US"]))

# ---- (4) ownership blocs. Stakes: Microsoft about 24% of OpenAI (as converted); Google 13.7% and Amazon 14.7% of Anthropic (non-voting)
resid_B = 100*(559 - 134)/mau_total                                  # upper bound for any unlisted vendor's MAU share (Counterpoint residual less Anthropic)
BASE = {l: {d: PRIM[l][m] for d, m in zip("ABCDE", M)} for l in labs}
def bloc(mat, a, b, extra=None):
    out_ = {k: dict(v) for k, v in mat.items() if k not in (a, b)}
    out_[a + "+" + b] = {d: mat[a][d] + (mat[b][d] if b in mat else 0) + (extra or {}).get(d, 0) for d in "ABCDE"}
    return out_
MSFT = dict(A=0.7, B=resid_B, C=7.2, D=1.6, E=0.0)                   # Microsoft: Copilot visits, residual MAU (upper bound), revenue, <1.6% of requests
S1 = {k: dict(v) for k, v in BASE.items()}; S1["OpenAI+Microsoft"] = {d: BASE["OpenAI"][d] + MSFT[d] for d in "ABCDE"}; del S1["OpenAI"]
S2 = bloc(BASE, "Google", "Anthropic")
S3 = bloc(S1, "Google", "Anthropic")
FAVB = {k: dict(v) for k, v in FAVM.items()}
FAV_S1 = {k: dict(v) for k, v in FAVB.items()}
FAV_S1["OpenAI+Microsoft"] = dict(A=79.4 + 2.79, B=max(BASE["OpenAI"]["B"] + resid_B, 46.4), C=29.0 + 7.2, D=22.46 + 1.6, E=56.0); del FAV_S1["OpenAI"]
FAV_S2 = {k: dict(v) for k, v in FAVB.items() if k not in ("Google", "Anthropic")}
FAV_S2["Google+Anthropic"] = dict(A=max(24.7 + 11.7, 28.86 + 2.3), B=max(BASE["Google"]["B"] + BASE["Anthropic"]["B"], 27.7 + 10.3),
                                  C=12.1 + 31.4, D=max(15.1 + 6.3, 30.39 + 9.80), E=max(55.0 + 9.0, 49.3 + 1.7, 58.7 + 1.4))
FAV_S3 = {k: dict(v) for k, v in FAV_S1.items() if k not in ("Google", "Anthropic")}; FAV_S3["Google+Anthropic"] = FAV_S2["Google+Anthropic"]
def test_matrix(mat):
    rows = {}
    for k, v in mat.items():
        srt = sorted(v.items(), key=lambda kv: kv[1])
        rows[k] = dict(shares={d: round(x, 1) for d, x in v.items()}, weakest=srt[0][0], min=round(srt[0][1], 1),
                       kappa50=round(50/srt[0][1], 2), kappa40=round(40/srt[0][1], 2), kappa30=round(30/srt[0][1], 2),
                       fail50=sum(x < 50 for x in v.values()), fail40=sum(x < 40 for x in v.values()), fail30=sum(x < 30 for x in v.values()))
    best = min(rows, key=lambda k: rows[k]["kappa50"])
    return dict(rows=rows, best=best, kappa50=rows[best]["kappa50"], kappa40=rows[best]["kappa40"], kappa30=rows[best]["kappa30"])
def mc35(mat, rho):
    _, r = mc_h0(mat, sigmas=[0.35], rho=rho, n=200000, seed=23)
    return {str(t): round(v[0], 4) for t, v in r.items()}
R6["blocs"] = {}
for name, pm, fm in (("S0 no bloc", BASE, FAVB), ("S1 OpenAI+Microsoft", S1, FAV_S1), ("S2 Google+Anthropic", S2, FAV_S2), ("S3 both blocs", S3, FAV_S3)):
    tp, tf = test_matrix(pm), test_matrix(fm)
    R6["blocs"][name] = dict(primary=dict(best=tp["best"], kappa50=tp["kappa50"], kappa40=tp["kappa40"], kappa30=tp["kappa30"], row=tp["rows"][tp["best"]]),
        favourable=dict(best=tf["best"], kappa50=tf["kappa50"], kappa40=tf["kappa40"], kappa30=tf["kappa30"], row=tf["rows"][tf["best"]]),
        mc_primary=mc35(pm, 0.0), mc_fav_indep=mc35(fm, 0.0), mc_fav_corr=mc35(fm, 0.5))
# partial-ownership adjustment of the HHI (O'Brien and Salop): silent financial interest beta*s_j*s_k; full merger 2*s_j*s_k
def mhhi(d, stakes):
    silent = sum(b*d.get(j, 0)*d.get(k, 0) for j, k, b in stakes); merger = sum(2*d.get(j, 0)*d.get(k, 0) for j, k, b in stakes)
    return round(silent), round(merger)
stakes = [("Microsoft", "OpenAI", 0.24), ("Google", "Anthropic", 0.137)]
R6["mhhi"] = {}
for name, d in D.items():
    lo_, hi_, _ = hhi_bounds(d); sil, mer = mhhi(d, stakes)
    R6["mhhi"][name[:2].strip()] = dict(hhi_lo=lo_, silent_delta=sil, merger_delta=mer, hhi_silent=lo_ + sil, hhi_merger=lo_ + mer)

# ---- (5) dimension D by seller instead of model author
lab_share_req = {"DeepSeek": 0.14, "Alibaba": 0.51, "Zai": 0.08}      # share of the family's tokens served by the lab's own API (open-weight families)
Dsel = dict(D["D  Developer gateway requests (OpenRouter, wk of 9 Nov 2026)"])
for k, f in lab_share_req.items(): Dsel[k] = Dsel[k]*f
Msel = {m: dict(M[m]) for m in M}; dkey = [m for m in M if m.startswith("D")][0]
Msel[dkey]["DeepSeek"] = Dsel["DeepSeek"]; Msel[dkey]["Alibaba"] = Dsel["Alibaba"]
lead_sel = {m: max(Msel[m], key=Msel[m].get) for m in Msel}
wS, pS, _ = w_test(Msel, labs)
R6["d_by_seller"] = dict(shares={k: round(v, 1) for k, v in sorted(Dsel.items(), key=lambda t: -t[1])}, leader=max(Dsel, key=Dsel.get),
    leaders=lead_sel, distinct_leaders=len(set(lead_sel.values())), W=wS, p_perm=round(pS, 4),
    openai_kappa50=round(50/min(Msel[m]["OpenAI"] for m in Msel), 2))

# ---- (6) common denominator: each dimension renormalised over the six labs of the matrix
CD = {m: {l: 100*M[m][l]/sum(M[m][x] for x in labs) for l in labs} for m in M}
cd_rows = {l: dict(min=round(min(CD[m][l] for m in CD), 1), kappa50=round(50/min(CD[m][l] for m in CD), 2), ge50=sum(CD[m][l] >= 50 for m in CD)) for l in labs}
R6["common_denominator"] = dict(matrix={m: {l: round(v, 1) for l, v in CD[m].items()} for m in CD}, rows=cd_rows,
    leaders={m: max(CD[m], key=CD[m].get) for m in CD}, best=min(cd_rows, key=lambda l: cd_rows[l]["kappa50"]))

# ---- (7) error rates of the decision rule under the log-normal error model, at the least favourable point of H0
def p_reject_boundary(kstar, sigma, k=5):       # true share = tau in every dimension; rule rejects when min observed < tau/kstar
    return 1 - (1 - norm.cdf(-np.log(kstar)/sigma))**k
def p_obs(min_obs, tau, sigma, k=5):            # P(min of k observed shares <= min_obs | all true shares = tau)
    return 1 - (1 - norm.cdf(np.log(min_obs/tau)/sigma))**k
sg = [0.2, 0.35, 0.5, 0.7]
R6["error_rates"] = dict(sigmas=sg, size_rule_kstar2=[round(float(p_reject_boundary(2, s)), 4) for s in sg],
    kstar_for_5pct=[round(float(np.exp(-s*norm.ppf(1 - 0.95**0.2))), 2) for s in sg],
    p_primary={str(t): [round(float(p_obs(19.0, t, s)), 4) for s in sg] for t in (50, 40, 30)},
    p_favourable={str(t): [round(float(p_obs(22.46, t, s)), 4) for s in sg] for t in (50, 40, 30)})

# ---- (8) upstream layers (context, not part of the test)
cloud = dict(Amazon=26, Microsoft=19, Google=16)
lo_, hi_, res_ = hhi_bounds(cloud)
R6["upstream"] = dict(cloud=cloud, cloud_hhi_lo=lo_, cloud_hhi_hi=hi_, cloud_cr3=61, nvidia_lo=75, nvidia_hi=86)
out["rev6"] = R6
out["favourable_matrix"] = {k: {d: round(x, 1) for d, x in v.items()} for k, v in FAVM.items()}
json.dump(out, open(RESULTS_JSON, "w"), indent=1)
show = {k: R6[k] for k in ("aicpb", "statcounter_summary", "d_by_seller", "error_rates", "upstream")}
show["markets"] = {m: {k: v for k, v in r.items() if k != "readings"} for m, r in R6["markets"].items()}
show["panel"] = {k: {kk: vv for kk, vv in R6["panel"][k].items() if kk not in ("hhi",)} for k in ("web", "dev")}
show["panel_misc"] = {k: v for k, v in R6["panel"].items() if k.startswith("openai") or k.startswith("dev_")and k!="dev_series"}
show["blocs"] = R6["blocs"]; show["mhhi"] = R6["mhhi"]; show["cd"] = dict(rows=cd_rows, leaders=R6["common_denominator"]["leaders"])
print("REV6"); print(json.dumps(show, indent=0)[:9000])
print("MC035", out["mc"]["at_sigma_035"], out["mc"]["sigma_at_5pct"])
print("FAVROW", [r for r in out["favourable"] if r["lab"] in ("OpenAI", "Anthropic")])

# ---- (9) pooled shortfall test of H0(tau) at its least favourable point (all true shares equal tau)
def p_pooled(shares, tau, sigma, rho=0.0, n=400000, seed=31):
    """T = sum_d max(0, ln(tau/s_d)). Under H0 every true share is >= tau; T is stochastically largest when all equal tau.
    Returns P(T >= T_obs) for log-normal errors with scale sigma, a share rho of whose variance is common to the vendor."""
    rng = np.random.default_rng(seed); k = len(shares)
    t_obs = sum(max(0.0, float(np.log(tau/s))) for s in shares)
    eps = sigma*(np.sqrt(rho)*rng.standard_normal((n, 1)) + np.sqrt(1 - rho)*rng.standard_normal((n, k)))
    T = np.maximum(0.0, -eps).sum(axis=1)
    return round(float(t_obs), 3), float((T >= t_obs - 1e-12).mean())
cases = {"Primary matrix, OpenAI": list(BASE["OpenAI"].values()), "Most favourable matrix, OpenAI": list(FAVB["OpenAI"].values()),
         "Common denominator, OpenAI": [CD[m]["OpenAI"] for m in CD],
         "Bloc OpenAI+Microsoft, primary": list(S1["OpenAI+Microsoft"].values()),
         "Bloc OpenAI+Microsoft, most favourable": list(FAV_S1["OpenAI+Microsoft"].values()),
         "Bloc Google+Anthropic, primary": list(S2["Google+Anthropic"].values()),
         "Bloc Google+Anthropic, most favourable": list(FAV_S2["Google+Anthropic"].values())}
R6["pooled"] = {}
for name, sh in cases.items():
    R6["pooled"][name] = {}
    for tau in (50, 40, 30):
        t_, p0 = p_pooled(sh, tau, 0.35, 0.0); _, p5 = p_pooled(sh, tau, 0.35, 0.5); _, pw = p_pooled(sh, tau, 0.5, 0.0)
        R6["pooled"][name][str(tau)] = dict(T=t_, p_indep=round(p0, 4), p_corr=round(p5, 4), p_sigma05=round(pw, 4))
out["rev6"] = R6
json.dump(out, open(RESULTS_JSON, "w"), indent=1)
for k, v in R6["pooled"].items(): print(k, v)


# =====================================================================================
# Revision 7: statistical validation. The error model is applied to every null, including the local,
#             paradigm and origin nulls that earlier revisions decided on point estimates alone.
# =====================================================================================
V = {}
def q95_S(k, rho, n=2_000_000, seed=77):
    rng = np.random.default_rng(seed)
    z = np.sqrt(rho)*rng.standard_normal((n, 1)) + np.sqrt(1 - rho)*rng.standard_normal((n, k))
    return float(np.quantile(np.maximum(0, -z).sum(1), 0.95))
def p_T(t_obs, k, sigma=0.35, rho=0.0, n=1_000_000, seed=78):
    rng = np.random.default_rng(seed)
    z = np.sqrt(rho)*rng.standard_normal((n, 1)) + np.sqrt(1 - rho)*rng.standard_normal((n, k))
    return float((sigma*np.maximum(0, -z).sum(1) >= t_obs - 1e-12).mean())
# (a) largest error scale at which the primary rejection of H0(50) still holds at the 5% level
t50 = sum(max(0.0, float(np.log(50/x))) for x in BASE["OpenAI"].values())
V["sigma_star"] = dict(T=round(t50, 3), indep=round(t50/q95_S(5, 0.0), 2), corr=round(t50/q95_S(5, 0.5), 2))
V["sigma_star"]["factor95_indep"] = round(float(np.exp(1.96*V["sigma_star"]["indep"])), 1)
V["sigma_star"]["factor95_corr"] = round(float(np.exp(1.96*V["sigma_star"]["corr"])), 1)
# (b) local nulls: one-dimensional version of the same test. p = Phi(ln(s1/tau)/sigma) when the leader is below tau.
def p_local(s1, tau=50, sigma=0.35): return 1.0 if s1 >= tau else float(norm.cdf(np.log(s1/tau)/sigma))
V["local_critical_share"] = {str(t): round(float(t*np.exp(norm.ppf(0.05)*0.35)), 1) for t in (50, 40, 30)}
V["local_metrics"] = [dict(metric=r["metric"][:2].strip(), leader=r["leader"], s1=r["s1"], p50=round(p_local(r["s1"]), 3)) for r in out["concentration"]]
V["local_metrics_rejected"] = [r["metric"] for r in V["local_metrics"] if r["p50"] < 0.05]
V["local_markets"] = {m: [(lab, sh, round(p_local(sh), 3)) for _, lab, sh in rows] for m, rows in MK.items()}
V["local_regions"] = {k: (v["leader"], v["s1"], round(p_local(v["s1"]), 3)) for k, v in R6["statcounter_regions"].items()}
# (c) paradigm level: eight metrics, and the same collapsed into five groups (gateway tokens, gateway spend, enterprise usage, enterprise spend, consumer)
open_sh = [v for _, v in P]
groups = [[72.0, 71.5], [27.0, 23.0], [19.0], [5.0, 4.0], [3.5]]
def side_test(shares, label):
    T_ = sum(max(0.0, float(np.log(50/x))) for x in shares); k = len(shares)
    return dict(side=label, k=k, min=round(min(shares), 1), kappa50=round(50/min(shares), 2), below50=sum(x < 50 for x in shares), T=round(T_, 3),
                p_indep=round(p_T(T_, k, rho=0.0), 4), p_corr=round(p_T(T_, k, rho=0.5), 4))
V["paradigm"] = dict(eight=[side_test(open_sh, "open"), side_test([100 - x for x in open_sh], "closed")],
                     five=[side_test([float(np.mean(g)) for g in groups], "open"), side_test([100 - float(np.mean(g)) for g in groups], "closed")])
# (d) origin level: latest reading of each of seven metrics; a missing side is bounded by 100 minus the other side
us = [95.73, 95.0, 83.5, 66.8, 65.3, 36.5, 100 - 69.0]
cn = [4.21, 100 - 95.0, 11.6, 21.5, 29.15, 41.0, 69.0]
V["origin"] = [side_test(us, "US"), side_test(cn, "China")]
out["rev7"] = V
out["rev6"]["panel"]["web"] = panel_stats(WEB); out["rev6"]["panel"]["dev"] = panel_stats(DEV)
json.dump(out, open(RESULTS_JSON, "w"), indent=1)
print("REV7", json.dumps(V, indent=0))
print("INST", out["rev6"]["panel"]["web"]["instability_mean"], out["rev6"]["panel"]["web"]["instability_max"], out["rev6"]["panel"]["dev"]["instability_mean"], out["rev6"]["panel"]["dev"]["instability_max"])
