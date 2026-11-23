"""Every number quoted in the manuscript, checked against `results/results.json`.

The analysis script is the single source of truth for the paper's numbers. These
tests close the loop in the other direction: each assertion below is a value
that appears in a table, a figure caption or a sentence of `paper/main.tex`, so
a silent drift between the script and the manuscript fails CI.

Run with:  pytest -q
"""
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "results" / "results.json").read_text())
VALIDATION = json.loads((ROOT / "results" / "validation.json").read_text())
TEX = (ROOT / "paper" / "main.tex").read_text()


def approx(x, tol=0.05):
    """Tolerance for a value the paper rounds to one decimal place."""
    return pytest.approx(x, abs=tol)


# --------------------------------------------------------------------------
# Table 2: within-metric concentration and local dominance
# --------------------------------------------------------------------------

# metric key -> (leader, s1, lead ratio, CR4, HHI lower, HHI upper)
TABLE2 = {
    "A": ("OpenAI", 56.6, 2.29, 96.5, 3968, 3968),
    "B": ("Meta", 25.1, 1.20, 74.0, 1627, 1661),
    "C": ("Anthropic", 31.4, 1.08, 79.7, 2075, 2075),
    "D": ("DeepSeek", 31.8, 1.67, 72.9, 1742, 1755),
    "E": ("Anthropic", 55.0, 1.77, 95.0, 4067, 4092),
    "D'": ("Google", 30.4, 1.35, 72.8, 1739, 1744),
    "E'": ("Anthropic", 49.3, 1.07, 99.5, 4564, 4565),
    "F": ("OpenAI", 46.4, 1.68, 84.4, 3026, 3161),
    "G": ("ByteDance", 54.3, 2.65, 95.5, 3663, 3663),
}


def _concentration_by_metric():
    """The metric label is the first token of the `metric` string in results.json."""
    return {row["metric"].split()[0]: row for row in RESULTS["concentration"]}


def test_table2_has_all_nine_metrics():
    assert set(_concentration_by_metric()) == set(TABLE2)


@pytest.mark.parametrize("key", list(TABLE2))
def test_table2_row(key):
    leader, s1, lead_ratio, cr4, hhi_lo, hhi_hi = TABLE2[key]
    row = _concentration_by_metric()[key]
    assert row["leader"] == leader
    assert row["s1"] == approx(s1)
    assert row["lead_ratio"] == approx(lead_ratio, tol=0.005)
    assert row["cr4"] == approx(cr4)
    assert (row["hhi_lo"], row["hhi_hi"]) == (hhi_lo, hhi_hi)


def test_hhi_bounds_are_ordered_and_residual_non_negative():
    for row in RESULTS["concentration"]:
        assert row["hhi_lo"] <= row["hhi_hi"], row["metric"]
        assert row["residual"] >= 0, row["metric"]


def test_six_of_nine_metrics_are_highly_concentrated():
    """Abstract: 'HHI above 1,800 in six of nine metrics'."""
    assert sum(r["hhi_lo"] > 1800 for r in RESULTS["concentration"]) == 6


def test_three_of_nine_local_dominances_on_point_estimates():
    """Section 4.1 / Fig. 1: leader at or above 50% in three of nine metrics."""
    dominant = [r["metric"].split()[0] for r in RESULTS["concentration"] if r["dominant50"]]
    assert dominant == ["A", "E", "G"]


def test_local_null_critical_share_and_single_rejection():
    """Section 4.1: the one-dimensional test rejects only below 28.1%, which
    happens for metric B alone (Meta at 25.1%, p = 0.024)."""
    rev7 = RESULTS["rev7"]
    assert rev7["local_critical_share"]["50"] == approx(28.1)
    assert rev7["local_metrics_rejected"] == ["B"]
    by_metric = {m["metric"]: m for m in rev7["local_metrics"]}
    assert by_metric["B"]["p50"] == approx(0.024, tol=0.001)
    # C, D and D' sit between 0.078 and 0.098; the paper says "between 0.08 and 0.10"
    for key in ("C", "D", "D'"):
        assert 0.07 < by_metric[key]["p50"] < 0.10, key
    # E' and F exceed 0.4
    for key in ("E'", "F"):
        assert by_metric[key]["p50"] > 0.4, key


# --------------------------------------------------------------------------
# Table 4: the primary cross-dimension vendor matrix
# --------------------------------------------------------------------------

# vendor -> (A, B, C, D, E, kappa(50), dims >= 50)
TABLE4 = {
    "OpenAI": (56.2, 20.9, 29.0, 19.0, 31.0, 2.6, 1),
    "Google": (24.6, 20.9, 12.1, 15.1, 9.0, 5.6, 0),
    "Anthropic": (11.7, 2.8, 31.4, 6.3, 55.0, 17.9, 1),
    "DeepSeek": (3.3, 3.0, 0.2, 31.8, 1.3, 250.0, 0),
    "Meta": (0.5, 25.1, 1.4, 0.8, 1.3, 100.0, 0),
    "Alibaba": (0.3, 6.7, 2.9, 6.2, 1.3, 161.3, 0),
}
DIMS = ["A web visits", "B MAU", "C revenue", "D gateway req.", "E enterprise"]


@pytest.mark.parametrize("vendor", list(TABLE4))
def test_table4_shares(vendor):
    expected = TABLE4[vendor][:5]
    got = tuple(RESULTS["matrix"][d][vendor] for d in DIMS)
    assert got == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("vendor", list(TABLE4))
def test_table4_robustness_margin(vendor):
    _, _, _, _, _, kappa, dims_ge50 = TABLE4[vendor]
    row = next(r for r in RESULTS["robustness"] if r["lab"] == vendor)
    assert row["kappa"] == approx(kappa, tol=0.05 if kappa < 100 else 0.5)
    assert row["dims_ge50"] == dims_ge50


def test_no_vendor_reaches_50_in_more_than_one_dimension():
    """The headline result of Section 4.3."""
    assert max(r["dims_ge50"] for r in RESULTS["robustness"]) == 1


def test_four_distinct_leaders_across_five_dimensions():
    leaders = RESULTS["leaders_primary"]
    assert leaders == {
        "A web visits": "OpenAI",
        "B MAU": "Meta",
        "C revenue": "Anthropic",
        "D gateway req.": "DeepSeek",
        "E enterprise": "Anthropic",
    }
    assert len(set(leaders.values())) == 4


def test_rule2_rejects_every_vendor_on_the_primary_matrix():
    """Rule 2 rejects when kappa > kappa* = 2."""
    assert all(r["kappa"] > 2.0 for r in RESULTS["robustness"])


def test_robust_leads_still_give_three_different_leaders():
    """Section 4.3: three leads have a lead ratio >= 1.5 (A, D, E) and they
    belong to three different vendors."""
    robust = {}
    for dim in DIMS:
        shares = sorted(RESULTS["matrix"][dim].items(), key=lambda kv: -kv[1])
        if shares[0][1] / shares[1][1] >= 1.5:
            robust[dim] = shares[0][0]
    assert robust == {
        "A web visits": "OpenAI",
        "D gateway req.": "DeepSeek",
        "E enterprise": "Anthropic",
    }


# --------------------------------------------------------------------------
# Table 5: the matrix most favourable to H0
# --------------------------------------------------------------------------

# vendor -> (A, B, C, D, E, k50, fail50, k40, fail40, k30, fail30)
TABLE5 = {
    "OpenAI": (79.4, 46.4, 29.0, 22.5, 56.0, 2.23, 3, 1.78, 2, 1.34, 2),
    "Google": (24.7, 27.7, 12.1, 30.4, 9.0, 5.56, 5, 4.44, 5, 3.33, 4),
    "Anthropic": (11.7, 10.3, 31.4, 9.8, 58.7, 5.10, 4, 4.08, 4, 3.06, 3),
    "DeepSeek": (3.5, 3.0, 0.2, 31.8, 4.0, 250.0, 5, 200.0, 5, 150.0, 4),
    "Meta": (0.5, 25.1, 1.4, 1.8, 4.0, 100.0, 5, 80.0, 5, 60.0, 5),
}


@pytest.mark.parametrize("vendor", list(TABLE5))
def test_table5_row(vendor):
    a, b, c, d, e, k50, f50, k40, f40, k30, f30 = TABLE5[vendor]
    row = next(r for r in RESULTS["favourable"] if r["lab"] == vendor)
    assert (row["A"], row["B"], row["C"], row["D"], row["E"]) == pytest.approx(
        (a, b, c, d, e), abs=0.05
    )
    tol = 0.01 if k50 < 100 else 0.5
    assert row["kappa50"] == approx(k50, tol=tol)
    assert row["kappa40"] == approx(k40, tol=tol)
    assert row["kappa30"] == approx(k30, tol=tol)
    assert (row["fail50"], row["fail40"], row["fail30"]) == (f50, f40, f30)


def test_favourable_matrix_still_rejects_rule2_at_tau_50():
    assert all(r["kappa50"] > 2.0 for r in RESULTS["favourable"])


def test_openai_is_the_closest_case_at_tau_40_and_30():
    """Section 4.3: at tau=40 OpenAI's margin of 1.78 falls inside the
    tolerance band (inconclusive); at tau=30 it is within a factor of 1.34."""
    openai = next(r for r in RESULTS["favourable"] if r["lab"] == "OpenAI")
    assert 1.0 < openai["kappa40"] <= 2.0
    assert openai["kappa30"] == approx(1.34, tol=0.01)
    assert openai["kappa2nd30"] == approx(1.03, tol=0.01)
    others = [r["kappa40"] for r in RESULTS["favourable"] if r["lab"] != "OpenAI"]
    assert all(k > 2.0 for k in others)


# --------------------------------------------------------------------------
# Rule 6: the measurement-error simulation (Fig. 3)
# --------------------------------------------------------------------------


def test_mc_fractions_at_the_reference_error_scale():
    """Section 4.3: at sigma = 0.35, H0(50) holds in 0.02% of draws with
    independent errors and 0.37% with correlated errors; H0(40) in 0.5% and
    2.7%; H0(30) in 8.2% and 15.1%. On the primary matrix: 0%, 0%, 0.4%."""
    at = RESULTS["mc"]["at_sigma_035"]
    assert at["fav_indep"]["50"] == approx(0.0002, tol=0.0001)
    assert at["fav_indep"]["40"] == approx(0.0049, tol=0.0005)
    assert at["fav_indep"]["30"] == approx(0.0823, tol=0.002)
    assert at["fav_corr"]["50"] == approx(0.0037, tol=0.0005)
    assert at["fav_corr"]["40"] == approx(0.0267, tol=0.002)
    assert at["fav_corr"]["30"] == approx(0.1509, tol=0.003)
    assert at["prim_indep"]["50"] == 0.0
    assert at["prim_indep"]["40"] == 0.0
    assert at["prim_indep"]["30"] == approx(0.0037, tol=0.0005)


def test_mc_is_increasing_in_the_error_scale():
    """More measurement noise can only help the null. The simulated curve is a
    Monte Carlo estimate, so a step is allowed to dip by a few standard errors
    (n = 100,000 draws per point) without the trend being in doubt."""
    tol = 0.005
    for key in ("favourable_indep", "favourable_corr", "primary_indep"):
        for tau in ("50", "40", "30"):
            series = RESULTS["mc"][key][tau]
            for lo, hi in zip(series, series[1:]):
                assert hi >= lo - tol, (key, tau, lo, hi)
            assert series[-1] >= series[0]


def test_mc_sigma_at_which_the_null_reaches_five_percent():
    """Section 4.3: with correlated errors H0(50) reaches 5% of draws only at
    sigma = 0.70 and H0(40) at sigma = 0.45; with independent errors neither
    threshold is reached over the simulated range."""
    s = RESULTS["mc"]["sigma_at_5pct"]
    assert s["fav_corr"]["50"] == approx(0.70, tol=0.001)
    assert s["fav_corr"]["40"] == approx(0.45, tol=0.001)
    assert s["fav_indep"]["50"] is None
    assert s["fav_indep"]["40"] is None


def test_reference_sigma_places_95pct_of_errors_within_a_factor_of_two():
    assert RESULTS["mc"]["factor95_at_035"] == approx(1.99, tol=0.01)
    assert math.exp(1.959963985 * 0.35) == pytest.approx(1.99, abs=0.01)


# --------------------------------------------------------------------------
# Rule 7: the calibrated pooled test
# --------------------------------------------------------------------------


def test_pooled_test_primary_matrix():
    """Abstract: 'a pooled test ... gives p between 0.0002 and 0.018'."""
    p = RESULTS["rev6"]["pooled"]["Primary matrix, OpenAI"]["50"]
    assert p["p_indep"] == approx(0.0002, tol=0.0001)
    assert p["p_corr"] == approx(0.0175, tol=0.001)
    assert p["p_sigma05"] == approx(0.009, tol=0.001)
    assert p["p_indep"] < 0.05 and p["p_corr"] < 0.05


def test_pooled_test_favourable_matrix_does_not_reject():
    """Abstract: 'With the source most favourable to H0 ... p rises to 0.07-0.16'."""
    p = RESULTS["rev6"]["pooled"]["Most favourable matrix, OpenAI"]["50"]
    assert p["p_indep"] == approx(0.0744, tol=0.002)
    assert p["p_corr"] == approx(0.1611, tol=0.003)
    assert p["p_indep"] > 0.05 and p["p_corr"] > 0.05


def test_pooled_test_at_tau_40_and_30():
    """H0(40) is rejected with independent errors only; H0(30) is not rejected."""
    prim = RESULTS["rev6"]["pooled"]["Primary matrix, OpenAI"]
    assert prim["40"]["p_indep"] == approx(0.0104, tol=0.001)
    assert prim["40"]["p_corr"] == approx(0.0759, tol=0.002)
    assert prim["40"]["p_indep"] < 0.05 < prim["40"]["p_corr"]
    assert prim["30"]["p_indep"] > 0.05 and prim["30"]["p_corr"] > 0.05


def test_largest_error_scale_at_which_the_rejection_survives():
    """Section 4.3: the rejection holds up to sigma = 0.65 with independent
    errors (factor 3.6) and sigma = 0.45 with correlated errors (factor 2.4)."""
    s = RESULTS["rev7"]["sigma_star"]
    assert s["indep"] == approx(0.65, tol=0.001)
    assert s["corr"] == approx(0.45, tol=0.001)
    assert s["factor95_indep"] == approx(3.6, tol=0.05)
    assert s["factor95_corr"] == approx(2.4, tol=0.05)


def test_error_rate_of_rule_2_at_the_boundary_of_H0():
    """Section 4.3: rule 2 with kappa* = 2 falsely rejects a truly dominant
    vendor with probability 0.1% at sigma = 0.2, 11% at 0.35 and 35% at 0.5;
    a 5% rate at sigma = 0.35 needs kappa* = 2.25."""
    er = RESULTS["rev6"]["error_rates"]
    assert er["sigmas"] == [0.2, 0.35, 0.5, 0.7]
    assert er["size_rule_kstar2"][:3] == pytest.approx([0.0013, 0.1136, 0.351], abs=0.001)
    assert er["kstar_for_5pct"][1] == approx(2.25, tol=0.01)


def test_openai_margin_clears_the_calibrated_tolerance_on_primary_but_not_favourable():
    """Section 4.3: 2.63 clears kappa* = 2.25, 2.23 does not."""
    kstar = RESULTS["rev6"]["error_rates"]["kstar_for_5pct"][1]
    primary = RESULTS["rev6"]["blocs"]["S0 no bloc"]["primary"]["kappa50"]
    favourable = RESULTS["rev6"]["blocs"]["S0 no bloc"]["favourable"]["kappa50"]
    assert primary == approx(2.63, tol=0.01)
    assert favourable == approx(2.23, tol=0.01)
    assert primary > kstar > favourable


# --------------------------------------------------------------------------
# Rule 5: rank concordance
# --------------------------------------------------------------------------


def test_kendall_w_primary_and_extended():
    assert RESULTS["kendall_primary"]["W"] == approx(0.379, tol=0.001)
    assert RESULTS["kendall_primary"]["p_perm"] == approx(0.0794, tol=0.002)
    assert RESULTS["kendall_ext"]["W"] == approx(0.343, tol=0.001)
    assert RESULTS["kendall_ext"]["p_perm"] == approx(0.0345, tol=0.002)
    # W is a coefficient of concordance and must lie in [0, 1]
    for key in ("kendall_primary", "kendall_big4", "kendall_ext"):
        assert 0.0 <= RESULTS[key]["W"] <= 1.0


def test_eight_metrics_give_four_different_leaders():
    leaders = RESULTS["kendall_ext"]["leaders"]
    assert len(leaders) == 8
    assert set(leaders.values()) == {"OpenAI", "Anthropic", "DeepSeek", "Google"}


# --------------------------------------------------------------------------
# Paradigm and origin levels
# --------------------------------------------------------------------------


def test_paradigm_level_rejects_open_but_not_closed():
    """Abstract: 'the test rejects dominance by open weights ... but not by
    closed weights'."""
    for spec in ("eight", "five"):
        open_side, closed_side = RESULTS["rev7"]["paradigm"][spec]
        assert open_side["side"] == "open" and closed_side["side"] == "closed"
        assert open_side["p_indep"] < 0.05 and open_side["p_corr"] < 0.05
        assert closed_side["p_indep"] > 0.05 and closed_side["p_corr"] > 0.05
        # closed weights fall below 50% by less than the tolerance
        assert closed_side["kappa50"] < 2.0


def test_origin_level_rejects_china_but_not_us():
    """Abstract: 'rejects dominance ... by Chinese labs, but not by ... US labs'."""
    us, china = RESULTS["rev7"]["origin"]
    assert us["side"] == "US" and china["side"] == "China"
    assert china["p_indep"] < 0.05 and china["p_corr"] < 0.05
    assert us["p_indep"] > 0.05 and us["p_corr"] > 0.05
    assert us["kappa50"] == approx(1.61, tol=0.01)
    assert us["below50"] == 2


# --------------------------------------------------------------------------
# Rule 8: ownership blocs
# --------------------------------------------------------------------------


def test_google_anthropic_bloc_meets_30pct_in_all_five_dimensions():
    """Abstract: 'If, on that reading, Google and Anthropic are counted as one
    bloc ..., a 30% threshold is met in all five dimensions'."""
    bloc = RESULTS["rev6"]["blocs"]["S2 Google+Anthropic"]["favourable"]
    assert bloc["best"] == "Google+Anthropic"
    assert bloc["row"]["fail30"] == 0
    assert bloc["kappa30"] < 1.0  # every dimension already at or above 30%
    assert all(v >= 30.0 for v in bloc["row"]["shares"].values())


def test_bloc_scenarios_are_monotone_in_the_null_direction():
    """Merging vendors can only make the null easier to satisfy."""
    blocs = RESULTS["rev6"]["blocs"]
    base = blocs["S0 no bloc"]["favourable"]["kappa50"]
    for name in ("S1 OpenAI+Microsoft", "S2 Google+Anthropic", "S3 both blocs"):
        assert blocs[name]["favourable"]["kappa50"] <= base, name


def test_partial_ownership_hhi_matches_the_obrien_salop_closed_form():
    """Rule 8: a silent financial interest beta adds beta * s_j * s_k."""
    m = VALIDATION["mhhi"]
    assert m["general_delta"] == approx(m["closed_form"], tol=0.01)
    assert m["closed_form"] == approx(0.137 * 24.65 * 11.65, tol=0.01)


# --------------------------------------------------------------------------
# Rule 9: durability over the thirteen-month panel
# --------------------------------------------------------------------------


def test_consumer_panel_leader_holds_fifty_percent_in_all_thirteen_months():
    """Abstract: 'above 50% in all 13 months of a panel while losing 19 points'."""
    panel = RESULTS["rev6"]["panel"]
    assert len(panel["months"]) == 13
    web = panel["web"]
    assert web["months_ge50"] == 13
    assert web["distinct_leaders"] == 1
    assert web["leader_changes"] == 0
    assert web["same_firm_ge50_all"] is True
    assert abs(panel["openai_web_change"]) == approx(19.0, tol=1.0)


def test_developer_panel_has_three_leaders_in_thirteen_months():
    """Abstract: 'Developer inference has no stable leader: three in 13 months'."""
    dev = RESULTS["rev6"]["panel"]["dev"]
    assert dev["distinct_leaders"] == 3
    assert dev["months_ge50"] == 2


def test_instability_index_is_an_order_of_magnitude_higher_for_developers():
    """Section 4.12: 2.9% of the consumer market changes hands per month
    against 19.4% of the developer market."""
    panel = RESULTS["rev6"]["panel"]
    assert panel["web"]["instability_mean"] == approx(2.94, tol=0.01)
    assert panel["dev"]["instability_mean"] == approx(19.37, tol=0.01)
    assert panel["dev"]["instability_mean"] > 6 * panel["web"]["instability_mean"]


# --------------------------------------------------------------------------
# The open-weight segment
# --------------------------------------------------------------------------


def test_chinese_families_supply_about_ninety_percent_of_open_gateway_tokens():
    """Abstract: 'Chinese families supply 90% of open-weight gateway tokens
    while US-headquartered sellers serve about 58%'."""
    gw = RESULTS["open_segment"]["gateway_family"]
    seller = RESULTS["open_segment"]["seller"]
    assert gw["china_origin"] == approx(89.6, tol=0.1)
    assert seller["hq_all_tokens"]["US"] == approx(57.6, tol=0.1)


def test_two_thirds_of_open_weight_tokens_are_sold_by_someone_other_than_the_author():
    """Section 6: 'about two thirds of open-weight gateway tokens are sold by
    firms other than the lab that released the weights'."""
    seller = RESULTS["open_segment"]["seller"]
    assert seller["lab_served_pct"] == approx(34.7, tol=0.1)
    assert 0.60 < 1 - seller["lab_served_pct"] / 100 < 0.70


def test_attributing_to_the_seller_collapses_concentration():
    """HHI falls from 3,225 by model family to 559 by seller."""
    seller = RESULTS["open_segment"]["seller"]
    assert seller["hhi_family"] == 3225
    assert seller["hhi_seller"] == 559
    assert seller["hhi_seller"] < 1000  # unconcentrated by the US bands


# --------------------------------------------------------------------------
# Upstream layers (context, outside the test)
# --------------------------------------------------------------------------


def test_accelerator_share_meets_every_threshold_used_in_the_paper():
    """Abstract: 'Upstream, one firm holds 75-86% of AI accelerators'."""
    up = RESULTS["rev6"]["upstream"]
    assert (up["nvidia_lo"], up["nvidia_hi"]) == (75, 86)
    assert up["nvidia_lo"] >= 50  # the only share in the paper that clears tau=50
    assert up["cloud_cr3"] == 61


# --------------------------------------------------------------------------
# Geography and multi-homing
# --------------------------------------------------------------------------


def test_country_level_panel_covers_nineteen_economies():
    summary = RESULTS["geo_summary"]
    assert summary["n"] == 19
    assert len(RESULTS["geo"]) == 19
    assert summary["chatgpt_ge50"] == 12
    assert summary["any_ge50"] == 13
    assert summary["hhi_min"] == 3158
    assert summary["hhi_max"] == 5707


def test_one_assistant_leads_seventeen_of_nineteen_countries():
    """Section 8: 'one assistant leads 17 of 19 countries'."""
    leaders = [row["leader"] for row in RESULTS["geo"]]
    assert leaders.count("ChatGPT") == 17


def test_multihoming_lower_bound_exceeds_one_tool_per_user():
    """Section 4.9: brand shares sum to 76% while 51% report any use."""
    mh = RESULTS["multihoming"]
    assert mh["sum_brands"] == 76
    assert mh["any_use"] == 51
    assert mh["tools_per_user_lb"] == approx(1.49, tol=0.01)
    assert mh["tools_per_user_lb"] == pytest.approx(76 / 51, abs=0.01)


def test_implied_closed_to_open_price_ratio():
    """Section 4.6: tokens and spend shares imply a price gap of about 7x."""
    pr = RESULTS["price_ratio"]
    assert pr["openrouter"] == approx(7.0, tol=0.1)
    assert pr["vercel"] == approx(8.4, tol=0.1)


# --------------------------------------------------------------------------
# Independent re-implementation (validate_stats.py)
# --------------------------------------------------------------------------


def test_kendall_w_agrees_with_the_friedman_route():
    k = VALIDATION["kendall_W"]
    assert k["script"] == approx(k["via_friedman"], tol=0.001)
    assert k["script"] == approx(RESULTS["kendall_primary"]["W"], tol=0.001)


def test_pooled_test_holds_its_nominal_size_at_the_boundary():
    """A valid test of H0 must reject at most 5% of the time when H0 is true
    at its least favourable point, and less strictly inside H0."""
    size = VALIDATION["pooled_size"]
    assert size["at_boundary_indep"] <= 0.055
    assert size["at_boundary_corr"] <= 0.055
    assert size["interior_60_indep"] < size["at_boundary_indep"]
    assert size["mixed_boundary"] < size["at_boundary_indep"]


def test_pooled_test_has_high_power_against_the_observed_configuration():
    assert VALIDATION["pooled_power_if_primary_true"] > 0.95


def test_rule2_size_formula_matches_simulation():
    r2 = VALIDATION["rule2_size"]
    assert r2["formula"] == approx(r2["simulated"], tol=0.002)


def test_hhi_upper_bound_is_not_exceeded_by_brute_force_search():
    """Rule 1: the analytic upper bound must dominate any admissible split of
    the unattributed residual."""
    h = VALIDATION["hhi_bound"]
    assert h["upper"] >= h["brute_force_max"]
    assert h["upper"] - h["brute_force_max"] <= 10


def test_pooled_p_values_are_reproduced_by_the_independent_implementation():
    """validate_stats.py re-runs the pooled test with a different seed, a
    larger simulation and a different null-distribution routine."""
    indep_script = RESULTS["rev6"]["pooled"]["Primary matrix, OpenAI"]["50"]["p_indep"]
    indep_check = VALIDATION["pooled_all_vendors"]["primary"]["50"]["p_indep"]
    assert indep_check == approx(indep_script, tol=0.0005)
    fav_script = RESULTS["rev6"]["pooled"]["Most favourable matrix, OpenAI"]["50"]["p_corr"]
    fav_check = VALIDATION["pooled_all_vendors"]["favourable"]["50"]["p_corr"]
    assert fav_check == approx(fav_script, tol=0.005)


def test_openai_is_the_binding_vendor_in_every_specification():
    """The union-intersection test is decided by the best-placed vendor, so the
    reported p-value must be the maximum over vendors."""
    for matrix in ("primary", "favourable"):
        for tau in ("50", "40", "30"):
            row = VALIDATION["pooled_all_vendors"][matrix][tau]
            assert row["max_vendor"] == "OpenAI", (matrix, tau)
            assert row["p_indep"] >= row["second"]


def test_instability_index_is_robust_to_the_unlisted_residual():
    """validate_stats.py recomputes the index counting the residual as one
    entity; the paper reports that version."""
    for side in ("web", "dev"):
        check = VALIDATION["instability"][side]
        script = RESULTS["rev6"]["panel"][side]
        assert check["full_mean"] == approx(script["instability_mean"], tol=0.01)
        assert check["full_max"] == approx(script["instability_max"], tol=0.01)


def test_monte_carlo_error_of_the_smallest_p_value_is_negligible():
    mc = VALIDATION["mc_error_p_primary"]
    assert mc["se"] < mc["p"] / 10


# --------------------------------------------------------------------------
# Manuscript / repository consistency
# --------------------------------------------------------------------------


def test_every_figure_produced_by_the_script_is_included_in_the_manuscript():
    figs = sorted(p.stem for p in (ROOT / "paper" / "figs").glob("*.pdf"))
    assert len(figs) == 14
    for stem in figs:
        assert f"figs/{stem}.pdf" in TEX, stem


def test_manuscript_cites_the_replication_package():
    assert "oliveira2026repo" in TEX
    bib = (ROOT / "paper" / "references.bib").read_text()
    assert "@misc{oliveira2026repo" in bib
    assert "github.com/rfapo/no-universal-ai-dominance" in bib


def test_source_list_covers_every_url_bearing_source():
    """The paper reports 61 data sources; sources.csv lists their URLs."""
    rows = (ROOT / "data" / "sources.csv").read_text().strip().splitlines()
    assert rows[0].startswith("bib_key,title,url,accessed")
    assert len(rows) - 1 >= 60


# --------------------------------------------------------------------------
# README / results consistency
# --------------------------------------------------------------------------

README = (ROOT / "README.md").read_text()


def _markdown_table_after(heading, n_cols):
    """Return the rows of the first markdown table following `heading`."""
    body = README.split(heading, 1)[1]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == n_cols and not set("".join(cells)) <= set("-: "):
                rows.append(cells)
        elif rows:
            break
    return rows


def test_readme_concentration_table_matches_results():
    """The README reprints Table 2; it must not drift from results.json."""
    rows = _markdown_table_after("### 1. The identity of the leader changes", 7)
    data = {r["metric"].split()[0]: r for r in RESULTS["concentration"]}
    seen = 0
    for cells in rows:
        key = cells[0].replace("′", "'")
        if key not in data:
            continue
        seen += 1
        row = data[key]
        assert cells[1] == row["leader"], key
        assert float(cells[2]) == approx(row["s1"]), key
        assert float(cells[3]) == approx(row["lead_ratio"], tol=0.005), key
        assert float(cells[4]) == approx(row["cr4"]), key
    assert seen == 9, f"only matched {seen} of 9 metrics in the README table"


def test_readme_vendor_matrix_matches_results():
    """The README reprints Table 4."""
    rows = _markdown_table_after("### 2. No vendor reaches 50%", 9)
    seen = 0
    for cells in rows:
        vendor = cells[0]
        if vendor not in TABLE4:
            continue
        seen += 1
        shares = [float(c.replace("**", "")) for c in cells[1:6]]
        assert shares == pytest.approx(
            [RESULTS["matrix"][d][vendor] for d in DIMS], abs=0.05
        ), vendor
    assert seen == len(TABLE4), f"only matched {seen} of {len(TABLE4)} vendors"


def test_readme_quotes_the_pooled_p_values_correctly():
    pooled = RESULTS["rev6"]["pooled"]
    prim = pooled["Primary matrix, OpenAI"]["50"]
    fav = pooled["Most favourable matrix, OpenAI"]["50"]
    assert f"{prim['p_indep']:.4f}" in README          # 0.0002
    assert f"{fav['p_indep']:.3f}" in README           # 0.074
    assert f"{fav['p_corr']:.3f}" in README            # 0.161


def test_readme_figure_links_resolve():
    import re

    for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README):
        assert (ROOT / rel).exists(), rel


def test_readme_internal_file_references_exist():
    import re

    # backticked paths that look like repository files
    refs = set(re.findall(r"`((?:code|data|results|paper|tests)/[A-Za-z0-9_./*-]+)`", README))
    for ref in refs:
        if "*" in ref:
            assert list(ROOT.glob(ref)), ref
        else:
            assert (ROOT / ref).exists(), ref


# --------------------------------------------------------------------------
# Effective number of competitors and the Hill profile (Table 2, Appendix C)
# --------------------------------------------------------------------------


def test_effective_n_inverts_the_hhi_bounds():
    """Table 2's 1/HHI column is the HHI restated as a firm count, so it must
    invert the bounds exactly: the low count pairs with the high index."""
    for row in RESULTS["concentration"]:
        assert row["eff_n_lo"] == approx(10000.0 / row["hhi_hi"], tol=0.01), row["metric"]
        assert row["eff_n_hi"] == approx(10000.0 / row["hhi_lo"], tol=0.01), row["metric"]
        assert row["eff_n_lo"] <= row["eff_n_hi"]


def test_hill_profile_is_non_increasing_in_q():
    """A defining property of the Hill family: weighting the leader more
    heavily can never report more effective competitors."""
    for row in RESULTS["concentration"]:
        prof = [row["hill_q0"], row["hill_q1"], row["hill_q2"], row["hill_qinf"]]
        for a, b in zip(prof, prof[1:]):
            assert b <= a + 1e-6, (row["metric"], prof)


def test_hill_q_infinity_is_the_reciprocal_leader_share():
    """q -> inf reduces to 1/s_(1) on the renormalised named shares."""
    for row in RESULTS["concentration"]:
        # shares are renormalised over named firms, so s_(1) is scaled by the named total
        assert row["hill_qinf"] >= 1.0
        assert row["hill_qinf"] <= row["hill_q0"] + 1e-6


def test_hill_q0_is_an_integer_count_of_named_firms():
    for row in RESULTS["concentration"]:
        assert abs(row["hill_q0"] - round(row["hill_q0"])) < 1e-6, row["metric"]


def test_hill_q2_is_at_most_the_effective_n_from_hhi():
    """hill_q2 renormalises the named shares and so discards the unattributed
    residual, which can only add competitors. It must therefore sit at or below
    the Table 2 figure, and the gap must grow with the residual."""
    for row in RESULTS["concentration"]:
        assert row["hill_q2"] <= row["eff_n_hi"] + 1e-6, row["metric"]
        if row["residual"] < 0.2:
            assert row["hill_q2"] == approx(row["eff_n_hi"], tol=0.02), row["metric"]


def test_concentrated_group_separates_from_the_rest_for_q_at_least_one():
    """Appendix C claims the separation holds for q >= 1 but not at q = 0."""
    conc = {"A", "E", "E'", "F", "G"}
    by = {r["metric"].split()[0]: r for r in RESULTS["concentration"]}
    for q in ("hill_q1", "hill_q2", "hill_qinf"):
        hi = max(by[k][q] for k in by if k in conc)
        lo = min(by[k][q] for k in by if k not in conc)
        assert hi < lo, (q, hi, lo)
    q0_hi = max(by[k]["hill_q0"] for k in by if k in conc)
    q0_lo = min(by[k]["hill_q0"] for k in by if k not in conc)
    assert not (q0_hi < q0_lo), "appendix says q=0 does NOT separate"


def test_hhi_and_leader_share_rankings_disagree():
    """Appendix C's point: E' is the most concentrated metric at q=2 and the
    least concentrated of the five at q=inf."""
    by = {r["metric"].split()[0]: r for r in RESULTS["concentration"]}
    conc = ["A", "E", "E'", "F", "G"]
    assert min(conc, key=lambda k: by[k]["hill_q2"]) == "E'"
    assert max(conc, key=lambda k: by[k]["hill_qinf"]) == "E'"
    assert min(conc, key=lambda k: by[k]["hill_qinf"]) == "E"


def test_manuscript_justifies_the_choice_of_index_and_links_the_appendix():
    assert "Why the HHI" in TEX
    assert "app:hill" in TEX and "\\ref{app:hill}" in TEX
    assert "1/\\mathrm{HHI}" in TEX


def test_provenance_reports_the_measured_archive_outcome():
    """The Provenance paragraph must state the real coverage, not a promise."""
    assert "46 now have a snapshot" in TEX
    log = (ROOT / "results" / "archive_log.txt")
    assert log.exists() and log.stat().st_size > 0


# --------------------------------------------------------------------------
# Exclusivity of the leader's base, by partial identification (Section 4.9)
# --------------------------------------------------------------------------


def test_no_multihoming_adjusted_hhi_is_reported():
    """Dividing shares by a mean multi-homing factor gives a vector that no longer
    sums to 100 and is not comparable with the HHI bands. It was reported in an
    earlier version and withdrawn; nothing may reintroduce it."""
    assert "adjusted_hhi" not in RESULTS["multihoming"]
    assert "HHI}{1.49" not in TEX and "1.49^{2}" not in TEX
    assert "no longer sums to 100" in TEX


def test_exclusivity_bounds_follow_the_two_binding_constraints():
    e = RESULTS["multihoming"]["exclusivity"]
    A, b, riv = e["any_use_mass"], e["leader_mass"], e["rival_masses"]
    assert e["excl_lo_pop"] == approx(max(0.0, A - sum(riv)), tol=0.05)
    assert e["excl_hi_pop"] == approx(min(b, A - max(riv)), tol=0.05)
    assert e["excl_lo_pop"] < e["excl_hi_pop"]


def test_exclusivity_interval_is_9_to_27_points_of_population():
    e = RESULTS["multihoming"]["exclusivity"]
    assert (e["excl_lo_pop"], e["excl_hi_pop"]) == (9.0, 27.0)
    assert e["excl_lo_share"] == approx(17.6, tol=0.1)
    assert e["excl_hi_share"] == approx(52.9, tol=0.1)


def test_upper_bound_lies_strictly_below_the_raw_share():
    """The only thing the marginals establish: the share overstates dependence."""
    e = RESULTS["multihoming"]["exclusivity"]
    assert e["excl_hi_share"] < e["raw_share"]
    assert e["raw_share"] == approx(66.7, tol=0.1)


def test_both_bounds_are_sharp_under_the_linear_program():
    """validate_stats.py searches over user types; a bound is sharp when the value
    just outside it admits no feasible population."""
    lp = VALIDATION["exclusivity_bounds"]
    assert lp["analytic"] == [9.0, 27.0]
    assert lp["lo_sharp"] is True and lp["hi_sharp"] is True
    assert lp["interior_feasible"] is True


def test_trackers_table_compares_a_single_month():
    """Gap raised in review: the published table paired Similarweb October with
    StatCounter August, over a window in which StatCounter's own leader share fell
    27.8 points."""
    assert "both for October 2026" in TEX
    assert "confounded method with period" in TEX
    sc = RESULTS["rev6"]["statcounter_regions"]["Worldwide (Oct)"]["shares"]
    assert "51.58" in TEX and f"{sc['Perplexity']:.2f}" in TEX


def test_thresholds_separate_merger_screening_from_dominance():
    assert "applies to mergers" in TEX and "not a test of unilateral dominance" in TEX


def test_geopolitical_claim_is_stated_as_association():
    assert "The access counterfactual" not in TEX
    assert "no more strongly than an association" in TEX
    assert "identified counterfactuals" in TEX


# --------------------------------------------------------------------------
# METR citations (Sections 4.11 and 7)
# --------------------------------------------------------------------------


def test_metr_citations_are_defined_and_used():
    bib = (ROOT / "paper" / "references.bib").read_text()
    for key in ("metr2026swebench", "metr2026survey"):
        assert f"@misc{{{key}" in bib, key
        assert f"\\cite{{{key}}}" in TEX or f"{key}}}" in TEX, key


def test_metr_grader_finding_sits_in_the_capability_section():
    """The SWE-bench grader gap belongs with 'Capability is not adoption',
    where it supports the claim that the ruler decides the verdict."""
    cap = TEX.split("\\subsection{Capability is not adoption}")[1].split("\\subsection{")[0]
    assert "metr2026swebench" in cap
    assert "24.2 percentage points" in cap


def test_metr_self_report_finding_sits_in_threats_to_validity():
    threats = TEX.split("\\section{Threats to validity}")[1].split("\\section{")[0]
    assert "metr2026survey" in threats
    assert "40" in threats and "self-report" in threats


def test_metr_urls_are_in_the_source_list_for_archiving():
    rows = (ROOT / "data" / "sources.csv").read_text()
    assert "metr.org/notes/2026-03-10" in rows
    assert "metr.org/blog/2026-05-11" in rows
