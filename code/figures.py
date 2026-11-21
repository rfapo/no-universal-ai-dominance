"""Figures for 'No Universal AI Dominance'.
One visual system for all figures: Inter typeface, full text width (6.27 in) so that type is
set at its true size, a four-hue palette validated for colour-vision deficiency plus neutrals,
direct labels, and a single dashed rule for the 50% threshold."""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
HALO = [pe.withStroke(linewidth=2.2, foreground='white')]

for f in glob.glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    fm.fontManager.addfont(f)
FONT = "Inter" if any(f.name == "Inter" for f in fm.fontManager.ttflist) else "DejaVu Sans"

from paths import RESULTS_JSON, FIGS_DIR
R = json.load(open(RESULTS_JSON))
W = 6.27                                                   # text width of the paper, inches
INK, SUB, MUTE, RULE, TRACK = "#1c1c1a", "#4a4945", "#8b8982", "#dedcd5", "#efeee9"
BLUE, ORANGE, TEAL, AMBER, GREY = "#2b6cb0", "#d9622b", "#1f9e89", "#dba112", "#b9b7af"
BLUE_T = "#a9c4e3"                                        # tint of BLUE for de-emphasised marks
plt.rcParams.update({
    "font.family": FONT, "font.size": 8, "pdf.fonttype": 3, "axes.linewidth": 0.6,
    "axes.edgecolor": RULE, "axes.labelcolor": SUB, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.color": MUTE, "ytick.color": INK, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "legend.fontsize": 7.8, "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
    "figure.dpi": 100, "savefig.pad_inches": 0.02})

def bare(ax, grid="x", ticks=(0, 25, 50, 75, 100), lim=(0, 100)):
    """Remove chart junk; keep a faint grid on one axis."""
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, pad=4); ax.set_axisbelow(True)
    if grid == "x":
        ax.xaxis.grid(True, color=RULE, lw=0.6)
        if lim: ax.set_xlim(*lim)
        if ticks is not None: ax.set_xticks(ticks)
    elif grid == "y":
        ax.yaxis.grid(True, color=RULE, lw=0.6); ax.spines["bottom"].set_visible(True)

def threshold(ax, x=50, label="50%", y=None, va="bottom"):
    ax.axvline(x, color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
    if label:
        y = ax.get_ylim()[1] if y is None else y
        ax.text(x, y, label, ha="center", va=va, fontsize=7, color=SUB)

def panel(ax, text):
    ax.set_title(text, loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=7)

def save(fig, name):
    fig.savefig(FIGS_DIR / f"{name}.pdf"); fig.savefig(FIGS_DIR / f"{name}.png", dpi=220); plt.close(fig)

# ------------------------------------------------------------------ 1  leader share by metric
rows = R["concentration"]
codes = ["A", "B", "C", "D", "E", "D′", "E′", "F", "G"]
desc = ["Consumer web visits", "Monthly active users", "LLM revenue", "Developer gateway requests", "Enterprise API spend, US",
        "Developer tokens excl. BYOK", "Business token spend, US", "App and web audience", "China AI-native apps, top five"]
ypos = [9.6, 8.6, 7.6, 6.6, 5.6, 3.9, 2.9, 1.9, 0.9]
fig, ax = plt.subplots(figsize=(W, 3.45))
for y, r, c, d in zip(ypos, rows, codes, desc):
    hit = r["s1"] >= 50
    ax.barh(y, 100, height=0.52, color=TRACK, zorder=1)
    ax.barh(y, r["s1"], height=0.52, color=BLUE if hit else BLUE_T, zorder=3)
    ax.text(-60.5, y, c, ha="left", va="center", fontweight="bold", color=INK)
    ax.text(-56, y, d, ha="left", va="center", color=INK)
    ax.text(104, y, r["leader"], ha="left", va="center", color=INK)
    ax.text(133, y, f'{r["s1"]:.1f}', ha="right", va="center", color=INK, fontweight="bold" if hit else "normal")
ax.text(-60.5, 10.35, "PRIMARY METRICS", fontsize=6.6, color=MUTE, va="center")
ax.text(-60.5, 4.65, "SECONDARY AND REGIONAL METRICS", fontsize=6.6, color=MUTE, va="center")
ax.text(104, 10.35, "LEADER", fontsize=6.6, color=MUTE, va="center"); ax.text(133, 10.35, "SHARE, %", fontsize=6.6, color=MUTE, va="center", ha="right")
bare(ax, lim=(-61, 134)); ax.set_ylim(0.3, 10.7); ax.set_yticks([])
ax.xaxis.grid(False)
for x in (25, 75): ax.plot([x, x], [0.55, 9.95], color="white", lw=0.7, zorder=2)
ax.plot([50, 50], [0.5, 10.0], color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=4)
ax.text(50, 10.35, "50% THRESHOLD", fontsize=6.6, color=SUB, va="center", ha="center")
ax.set_xticks([0, 25, 50, 75, 100])
ax.legend(handles=[Patch(fc=BLUE, label="Leader at or above 50%"), Patch(fc=BLUE_T, label="Leader below 50%")],
          loc="upper left", bbox_to_anchor=(0.305, -0.075), ncol=2, frameon=False, handlelength=1.2, handleheight=0.8, columnspacing=1.6)
fig.subplots_adjust(left=0.01, right=0.995, top=0.995, bottom=0.14)
save(fig, "fig_leader")

# ------------------------------------------------------------------ 2  heat map vendor x dimension
M = R["matrix"]; dims = list(M); labs = list(M[dims[0]])
Z = np.array([[M[d][l] for d in dims] for l in labs])
cmap = LinearSegmentedColormap.from_list("seq", ["#f6f8fb", "#cfe0f1", "#8db4dd", "#2b6cb0", "#173f6c"])
fig, ax = plt.subplots(figsize=(W, 2.75))
ax.imshow(Z, cmap=cmap, vmin=0, vmax=60, aspect="auto")
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        lead = Z[i, j] == Z[:, j].max()
        ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
                color="white" if Z[i, j] > 34 else INK, fontweight="bold" if lead else "normal")
heads = [("A", "Web visits"), ("B", "Active users"), ("C", "Revenue"), ("D", "Gateway requests"), ("E", "Enterprise API spend")]
ax.set_xticks(range(5)); ax.set_xticklabels([f"{a}\n{b}" for a, b in heads], color=INK, fontsize=8, linespacing=1.35)
ax.xaxis.tick_top(); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, len(labs)), minor=True)
ax.grid(which="minor", color="white", lw=2.2); ax.tick_params(which="both", length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.subplots_adjust(left=0.115, right=0.995, top=0.84, bottom=0.02)
save(fig, "fig_heatmap")

# ------------------------------------------------------------------ 3  measurement-error simulation
mc = R["mc"]; sg = mc["sigmas"]
fig, axes = plt.subplots(1, 2, figsize=(W, 2.55), sharey=True)
for ax, key, ttl in zip(axes, ("favourable_indep", "favourable_corr"), ("(a) Independent errors", "(b) Errors correlated within vendor")):
    ax.axhspan(0, 5, color=TRACK, zorder=0)
    ends = {}
    for tau, col in (("30", AMBER), ("40", ORANGE), ("50", BLUE)):
        y = [100*v for v in mc[key][tau]]; ax.plot(sg, y, color=col, lw=1.8, solid_capstyle="round", zorder=3); ends[tau] = y[-1]
    off = {"30": 0.0, "40": 1.3, "50": -0.2} if key == "favourable_indep" else {"30": 0, "40": 0, "50": 0}
    for tau in ends: ax.text(1.03, ends[tau] + off[tau], f"τ = {tau}%", color=INK, fontsize=7.5, va="center")
    ax.axvline(0.35, color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
    bare(ax, grid="y", lim=None); ax.set_xlim(0, 1.22); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylim(0, 27)
    ax.set_xlabel("Scale of measurement error, σ (log units)"); panel(ax, ttl)
axes[0].set_ylabel("Draws in which the null holds, %")
axes[1].text(1.21, 4.1, "5% of draws", color=MUTE, fontsize=7, va="center", ha="right")
axes[0].text(0.37, 25.5, "σ = 0.35: 95% of errors\nwithin a factor of two", color=SUB, fontsize=7, va="top", linespacing=1.35)
fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.165, wspace=0.07)
save(fig, "fig_mc")

# ------------------------------------------------------------------ 4  open-weight share by metric
P = R["paradigm"]["metrics"]
groups = [("DEVELOPER GATEWAYS, TOKENS", [0, 1]), ("DEVELOPER GATEWAYS, SPEND", [2, 3]), ("ENTERPRISES", [4, 5, 6]), ("CONSUMERS", [7])]
names = ["OpenRouter", "Vercel AI Gateway", "OpenRouter", "Vercel AI Gateway", "Usage, US survey", "Metered consumption", "Spend, US survey", "Web visits to an open-weight lab"]
fig, ax = plt.subplots(figsize=(W, 3.05)); y = 0; yt = []
for g, idx in groups:
    ax.text(-52, y + 0.05, g, fontsize=6.6, color=MUTE, va="center"); y -= 0.85
    for i in idx:
        v = P[i][1]; hit = v >= 50
        ax.plot([0, 100], [y, y], color=TRACK, lw=5.5, solid_capstyle="butt", zorder=1)
        ax.plot([0, v], [y, y], color=BLUE if hit else BLUE_T, lw=5.5, solid_capstyle="butt", zorder=2)
        ax.text(-52, y, names[i], va="center", color=INK); ax.text(104, y, f"{v:g}", va="center", color=INK, fontweight="bold" if hit else "normal")
        y -= 0.85
    y -= 0.25
bare(ax, lim=(-53, 121)); ax.xaxis.grid(False); ax.set_yticks([]); ax.set_ylim(y + 0.55, 0.55)
ax.plot([50, 50], [y + 0.75, -0.45], color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=4)
ax.text(50, 0.05, "50%", fontsize=7, color=SUB, ha="center", va="center"); ax.text(104, 0.05, "SHARE, %", fontsize=6.6, color=MUTE, va="center")
ax.set_xlabel("Open-weight share, %", x=0.59)
fig.subplots_adjust(left=0.01, right=0.995, top=0.995, bottom=0.135)
save(fig, "fig_open")

# ------------------------------------------------------------------ 5  token scale
sc = R["scale"]
items = [("OpenRouter, all models", "Sep 2026", sc["openrouter_T_per_day"], BLUE), ("Google, first-party API", "Q3 2026", sc["google_api_T_per_day"], GREY),
         ("Google, all surfaces", "May 2026", sc["google_T_per_day"], GREY), ("ByteDance Doubao models", "Jun 2026", sc["doubao_T_per_day"], GREY)]
fig, ax = plt.subplots(figsize=(W, 1.55)); ys = np.arange(len(items))[::-1]
for y, (n, d, v, c) in zip(ys, items):
    ax.barh(y, v, height=0.5, color=c, zorder=3); ax.text(v + 3, y, f"{v:.0f}", va="center", color=INK, fontweight="bold" if c == BLUE else "normal")
    ax.text(-128, y, n, va="center", color=INK); ax.text(-34, y, d, va="center", color=MUTE, fontsize=7.5)
bare(ax, ticks=[0, 50, 100, 150, 200], lim=(-129, 205)); ax.set_yticks([]); ax.set_ylim(-0.55, 3.55)
ax.set_xlabel("Trillion tokens per day", x=0.69)
fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.27)
save(fig, "fig_scale")

# ------------------------------------------------------------------ 6  origin dumbbell
O = R["origin"]
names = ["Consumer web visits", "Enterprise API spend, US", "LLM revenue", "Vendor monthly active users", "Developer tokens excl. BYOK",
         "Developer tokens excl. BYOK", "Open-model downloads", "Gateway tokens"]
dates = ["Oct 2026", "2026", "Q1 2026", "Apr 2026", "Oct 2026", "Jul 2026", "Year to Feb 2026", "Sep 2026"]
fig, ax = plt.subplots(figsize=(W, 2.85)); ys = np.arange(len(O))[::-1]
for y, (_, us, cn), n, d in zip(ys, O, names, dates):
    ax.plot([0, 100], [y, y], color=TRACK, lw=1.2, zorder=1)
    if us is not None and cn is not None: ax.plot([us, cn], [y, y], color=GREY, lw=2.2, zorder=2, solid_capstyle="round")
    for v, col, mk, other in ((us, BLUE, "o", cn), (cn, ORANGE, "D", us)):
        if v is None: continue
        ax.plot(v, y, mk, ms=6.6 if mk == "o" else 5.6, color=col, mec="white", mew=1.1, zorder=3)
        right = other is None or v > other
        ax.text(v + (3 if right else -3), y, f"{v:.1f}", va="center", ha="left" if right else "right", fontsize=7.6, color=INK, path_effects=HALO, zorder=6)
    ax.text(-80, y, n, va="center", color=INK); ax.text(-26, y, d, va="center", color=MUTE, fontsize=7.3)
bare(ax, lim=(-81, 108)); ax.set_yticks([]); ax.set_ylim(-0.7, len(O) - 0.25)
ax.plot([50, 50], [-0.55, len(O) - 0.55], color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=1.5)
ax.set_xlabel("Share held by labs of each origin, %", x=0.7)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, mec="white", ms=6.6, label="US-origin labs"),
                   Line2D([], [], marker="D", ls="", color=ORANGE, mec="white", ms=5.6, label="China-origin labs")],
          loc="lower left", bbox_to_anchor=(0.42, 0.985), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.8)
fig.subplots_adjust(left=0.01, right=0.995, top=0.915, bottom=0.15)
save(fig, "fig_origin")

# ------------------------------------------------------------------ 7  assistants by country
G = R["geo"]; geo = {g["country"]: g for g in G}
C = {"United States":[62.9,16.5,14.8,1.0], "India":[64.1,20.2,11.7,1.1], "Japan":[55.6,35.5,6.3,0.1], "Brazil":[65.1,26.3,4.7,1.6], "Indonesia":[49.4,34.8,11.0,2.2],
     "United Kingdom":[68.2,14.3,13.0,1.0], "South Korea":[56.5,30.6,9.8,0.4], "Germany":[62.6,21.5,9.8,1.9], "France":[62.5,21.4,12.0,0.9], "Canada":[64.3,19.3,11.6,1.1],
     "Vietnam":[46.8,43.5,3.4,2.6], "Mexico":[47.1,39.2,7.3,2.6], "China":[17.7,5.5,1.3,73.2], "Philippines":[62.2,25.7,8.4,1.3], "Spain":[55.0,29.7,10.0,1.7],
     "Italy":[43.5,44.8,7.8,1.2], "Australia":[66.6,12.5,15.5,0.8], "Turkey":[48.4,43.0,5.8,1.0], "Russia":[43.6,13.7,4.5,32.1]}
order = sorted(C, key=lambda c: -C[c][0]); series = [("ChatGPT", BLUE), ("Gemini", ORANGE), ("Claude", TEAL), ("DeepSeek", AMBER), ("Other three", GREY)]
fig, ax = plt.subplots(figsize=(W, 4.25)); ys = np.arange(len(order))[::-1]
for y, c in zip(ys, order):
    parts = C[c] + [100 - sum(C[c])]; left = 0
    for (n, col), p in zip(series, parts):
        ax.barh(y, p, left=left, height=0.64, color=col, edgecolor="white", linewidth=0.9, zorder=3)
        if p >= 12: ax.text(left + p/2, y, f"{p:.0f}", ha="center", va="center", fontsize=7, color="white" if col in (BLUE, ORANGE, TEAL) else INK, zorder=4)
        left += p
    ax.text(-1.5, y, c, ha="right", va="center", color=INK)
bare(ax, lim=(-24, 102)); ax.xaxis.grid(False); ax.set_yticks([]); ax.set_ylim(-0.65, len(order) - 0.35)
ax.plot([50, 50], [-0.55, len(order) - 0.45], color=INK, lw=0.8, ls=(0, (3, 2.2)), zorder=5)
ax.set_xlabel("Share of web visits among seven assistants, %", x=0.6)
ax.legend(handles=[Patch(fc=c, label=n) for n, c in series], loc="lower left", bbox_to_anchor=(0.185, 0.992), ncol=5, frameon=False,
          handlelength=1.1, handleheight=0.8, handletextpad=0.45, columnspacing=1.5)
fig.subplots_adjust(left=0.01, right=0.995, top=0.94, bottom=0.095)
save(fig, "fig_geo")

# ------------------------------------------------------------------ 8  adoption: by economy, by definition
fig, axes = plt.subplots(1, 2, figsize=(W, 3.5), gridspec_kw=dict(width_ratios=[1, 1.32]))
ax = axes[0]; ms = R["microsoft"]; ys = np.arange(len(ms))[::-1]
for y, (n, v, k) in zip(ys, ms):
    agg = k == "a"
    ax.barh(y, v, height=0.56, color=GREY if agg else BLUE, zorder=3)
    ax.text(v + 2, y, f"{v:.1f}", va="center", color=INK, fontsize=7.6)
    ax.text(-2.5, y, n, va="center", ha="right", color=SUB if agg else INK, fontstyle="italic" if agg else "normal")
bare(ax, ticks=[0, 25, 50, 75], lim=(-68, 88)); ax.set_yticks([]); ax.set_ylim(-0.6, len(ms) - 0.4)
panel(ax, "(a) By economy")
ax.set_xlabel("% of working-age population, Q2 2026", x=0.62, fontsize=7.4)
ax = axes[1]; am = R["adoption_measures"]; ys = np.arange(len(am))[::-1]
who = ["Organisations, any AI use", "US businesses paying for AI", "Individuals, weekly use", "Individuals, China", "US workers, use at work",
       "EU individuals, last 3 months", "Brazil, internet users", "EU enterprises using AI", "World working-age population", "US firms, AI in production"]
src = ["global survey, 2025", "card data, Oct 2026", "six countries, 2026", "CNNIC, mid-2026", "survey, Nov 2025", "Eurostat, 2025", "TIC Domicílios, 2025", "Eurostat, 2025",
       "telemetry, Q2 2026", "Census, late 2023"]
for y, (_, v), a, b in zip(ys, am, who, src):
    ax.plot([0, 100], [y, y], color=TRACK, lw=1.2, zorder=1); ax.plot([0, v], [y, y], color=BLUE_T, lw=2.2, zorder=2)
    ax.plot(v, y, "o", ms=6.2, color=BLUE, mec="white", mew=1.1, zorder=3)
    ax.text(v + 3.5, y, f"{v:g}" + ("+" if v == 50.0 else ""), va="center", fontsize=7.6, color=INK)
    ax.text(-3, y + 0.17, a, va="center", ha="right", color=INK, fontsize=7.6); ax.text(-3, y - 0.24, b, va="center", ha="right", color=MUTE, fontsize=6.6)
bare(ax, lim=(-104, 104)); ax.set_yticks([]); ax.set_ylim(-0.6, len(am) - 0.4)
panel(ax, "(b) By definition of adoption"); ax.set_xlabel("%", x=0.75, fontsize=7.4)
fig.subplots_adjust(left=0.01, right=0.995, top=0.925, bottom=0.115, wspace=0.03)
save(fig, "fig_adoption")

# ------------------------------------------------------------------ 9  persistence
S = R["series"]
fig, axes = plt.subplots(1, 2, figsize=(W, 2.6))
ax = axes[0]; months = ["Dec 25", "Jan", "Feb", "Mar", "Apr 26", "May", "Jun", "Jul", "Aug", "Sep 26"]
full = ["Dec 25", "Jan 26", "Feb 26", "Mar 26", "Apr 26", "May 26", "Jun 26", "Jul 26", "Aug 26", "Sep 26"]
xt = [full.index(m) for m, _ in S["vercel_tokens"]]; yt = [v for _, v in S["vercel_tokens"]]
xs = [full.index(m) for m, _ in S["vercel_spend"]]; ysp = [v for _, v in S["vercel_spend"]]
ax.plot(xt, yt, "-o", color=BLUE, lw=1.8, ms=4.6, mec="white", mew=1.0, zorder=3); ax.plot(xs, ysp, "-D", color=ORANGE, lw=1.8, ms=4.0, mec="white", mew=1.0, zorder=3)
ax.text(9.3, yt[-1], "Tokens  71.5", va="center", color=INK, fontsize=7.6); ax.text(9.3, ysp[-1], "Spend  23", va="center", color=INK, fontsize=7.6)
ax.text(0, 11, "7", ha="center", color=INK, fontsize=7.6); ax.text(6, 9, "4", ha="center", color=INK, fontsize=7.6)
ax.axhline(50, color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
bare(ax, grid="y", lim=None); ax.set_xlim(-0.5, 12.3); ax.set_xticks([0, 4, 6, 9]); ax.set_xticklabels(["Dec 2025", "Apr", "Jun", "Sep 2026"]); ax.set_ylim(0, 82)
ax.set_yticks([0, 25, 50, 75]); panel(ax, "(a) Open-weight share on one gateway, %")
ax = axes[1]
for name, col, mk in (("Anthropic", BLUE, "o"), ("OpenAI", ORANGE, "D"), ("Google", TEAL, "s")):
    pts = S["menlo"][name]; x = [p[0] for p in pts]; y = [p[1] for p in pts]
    ax.plot(x, y, "-" + mk, color=col, lw=1.8, ms=4.6 if mk == "o" else 4.0, mec="white", mew=1.0, zorder=3)
    ax.text(2026.13, y[-1], f"{name}  {y[-1]}", va="center", color=INK, fontsize=7.6)
ax.text(2023, 53.5, "50", ha="center", color=INK, fontsize=7.6); ax.text(2023, 6.5, "12", ha="center", color=INK, fontsize=7.6)
ax.axhline(50, color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
bare(ax, grid="y", lim=None); ax.set_xlim(2022.75, 2027.25); ax.set_xticks([2023, 2025, 2026]); ax.set_ylim(0, 62); ax.set_yticks([0, 25, 50])
panel(ax, "(b) Enterprise API spend share, US, %")
fig.subplots_adjust(left=0.045, right=0.995, top=0.895, bottom=0.1, wspace=0.16)
save(fig, "fig_persist")

# ------------------------------------------------------------------ 10  access counterfactual: DeepSeek share
A = R["access"]; sw = A["similarweb"]
rest = [("China", 89, sw["China"]), ("Belarus", 56, None), ("Cuba", 49, None), ("Russia", 43, sw["Russia"])]
opn = sorted([(c, sw[c]) for c in A["south"] + A["north"]], key=lambda t: -t[1])
fig, axes = plt.subplots(1, 2, figsize=(W, 3.35), gridspec_kw=dict(width_ratios=[1, 1.05]))
ax = axes[0]; ys = np.arange(len(rest))[::-1]
for y, (c, m, s) in zip(ys, rest):
    ax.barh(y + 0.17, m, height=0.3, color=ORANGE, zorder=3); ax.text(m + 2, y + 0.17, f"{m}", va="center", fontsize=7.6, color=INK, path_effects=HALO, zorder=6)
    if s is not None:
        ax.barh(y - 0.17, s, height=0.3, color="#efb497", zorder=3); ax.text(s + 2, y - 0.17, f"{s:.1f}", va="center", fontsize=7.6, color=INK)
    ax.text(-2.5, y, c, va="center", ha="right", color=INK)
bare(ax, lim=(-27, 104)); ax.set_yticks([]); ax.set_ylim(-0.75, len(rest) - 0.3)
ax.plot([50, 50], [-0.6, len(rest) - 0.45], color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
panel(ax, "(a) Markets where US services are restricted"); ax.set_xlabel("DeepSeek share, %", x=0.6)
fig.legend(handles=[Patch(fc=ORANGE, label="Microsoft telemetry, H2 2025"), Patch(fc="#efb497", label="Similarweb visits, Sep 2026")], loc="lower left",
           bbox_to_anchor=(0.085, 0.0), ncol=2, frameon=False, handlelength=1.1, handleheight=0.8, columnspacing=1.4)
ax = axes[1]; ys = np.arange(len(opn))[::-1]
for y, (c, v) in zip(ys, opn):
    south = c in A["south"]
    ax.barh(y, v, height=0.6, color=AMBER if south else GREY, zorder=3); ax.text(v + 1.6, y, f"{v:.1f}", va="center", fontsize=7.2, color=INK)
    ax.text(-2.5, y, c, va="center", ha="right", color=INK, fontsize=7.4)
bare(ax, lim=(-36, 104)); ax.set_yticks([]); ax.set_ylim(-0.75, len(opn) - 0.3)
ax.plot([50, 50], [-0.6, len(opn) - 0.45], color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
panel(ax, "(b) Markets where US services are available"); ax.set_xlabel("DeepSeek share of web visits, Sep 2026, %", x=0.62)
ax.legend(handles=[Patch(fc=AMBER, label="Global South"), Patch(fc=GREY, label="Global North")], loc="lower right", bbox_to_anchor=(1.0, 0.03),
          ncol=1, frameon=False, handlelength=1.1, handleheight=0.8, labelspacing=0.35)
fig.subplots_adjust(left=0.01, right=0.995, top=0.925, bottom=0.2, wspace=0.04)
save(fig, "fig_access")
print("figures written with font:", FONT)

# ====================================================================================== revision 5
# Inside the open-weight segment. Hue encodes origin as in the origin figure (orange: China, blue: US);
# other origins are neutral grey; teal marks the open-weight frontier in the capability figure.
OS = R["open_segment"]
ORANGE_T, GREY_T, TEAL_T = "#efb497", "#dcdad3", "#a5d8cf"
OCOL = {"China": (ORANGE, ORANGE_T), "US": (BLUE, BLUE_T), "Other": ("#8b8982", GREY_T)}

# ------------------------------------------------------------------ 11  open families by layer
fams = [("Qwen", "Alibaba", "China"), ("DeepSeek", "DeepSeek", "China"), ("GLM", "Z.ai", "China"), ("Kimi", "Moonshot", "China"),
        ("Gemma", "Google", "US"), ("Llama", "Meta", "US"), ("Nemotron", "NVIDIA", "US"), ("Mistral", "Mistral AI", "Other")]
mx = OS["matrix"]
layers = [("Hub downloads", "% of top 300, 30 days", mx["downloads"], 72, [0, 25, 50]),
          ("Local inference", "GGUF, million a month", mx["gguf"], 72, [0, 25, 50]),
          ("Gateway tokens", "% of open-weight tokens", mx["gateway"], 72, [0, 25, 50]),
          ("Enterprise use", "% of production users", mx["enterprise"], 72, [0, 25, 50])]
ypos = [7.6, 6.6, 5.6, 4.6, 2.9, 1.9, 0.9, -0.1]
fig, axes = plt.subplots(1, 4, figsize=(W, 3.3), sharey=True)
for k, (ax, (title, sub, data, xmax, ticks)) in enumerate(zip(axes, layers)):
    top = max(v for f, v in data.items() if f in [x[0] for x in fams])
    for y, (f, lab, org) in zip(ypos, fams):
        v = data.get(f)
        ax.barh(y, xmax, height=0.56, color=TRACK, zorder=1)
        if v is None:
            ax.text(1.5, y, "not reported", va="center", fontsize=6.8, color=MUTE, style="italic", zorder=4)
            continue
        lead = v == top
        ax.barh(y, v, height=0.56, color=OCOL[org][0 if lead else 1], zorder=3)
        txt = "<0.1" if v < 0.1 else (f"{v:.1f}" if v != int(v) else f"{int(v)}")
        ax.text(v + 1.8, y, txt, va="center", fontsize=7.6, color=INK, fontweight="bold" if lead else "normal", path_effects=HALO, zorder=6)
    bare(ax, ticks=ticks, lim=(0, xmax)); ax.set_yticks([]); ax.set_ylim(-0.75, 9.9)
    ax.xaxis.grid(False)
    for x in ticks[1:]: ax.plot([x, x], [-0.45, 7.95], color="white", lw=0.7, zorder=2)
    ax.text(0, 9.55, f"({'abcd'[k]}) {title}", fontsize=8.3, fontweight="bold", color=INK, va="center")
    ax.text(0, 8.85, sub, fontsize=6.7, color=SUB, va="center")
ax0 = axes[0]; tr = ax0.get_yaxis_transform()
for y, (f, lab, org) in zip(ypos, fams):
    ax0.text(-1.03, y + 0.02, f, transform=tr, va="center", color=INK, fontweight="bold", fontsize=8)
    ax0.text(-0.47, y, lab, transform=tr, va="center", color=MUTE, fontsize=7.2)
ax0.text(-1.03, 8.3, "CHINA-ORIGIN FAMILIES", transform=tr, fontsize=6.4, color=MUTE, va="center")
ax0.text(-1.03, 3.6, "US AND EU FAMILIES", transform=tr, fontsize=6.4, color=MUTE, va="center")
fig.legend(handles=[Patch(fc=ORANGE, label="China-origin, leader of the layer"), Patch(fc=ORANGE_T, label="China-origin"),
                    Patch(fc=BLUE, label="US-origin, leader of the layer"), Patch(fc=BLUE_T, label="US-origin"), Patch(fc=GREY_T, label="EU-origin")],
           loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=5, frameon=False, handlelength=1.1, handleheight=0.8, columnspacing=1.2, fontsize=7.2)
fig.subplots_adjust(left=0.215, right=0.995, top=0.995, bottom=0.155, wspace=0.08)
save(fig, "fig_openfam")

# ------------------------------------------------------------------ 12  how far behind, by ruler
lag = OS["lag_months"]
order = [0, 1, 2, 3, 4, 5, 6, 7]
names = ["Capabilities index, 2023 to Oct 2025", "Capabilities index, since Jan 2026", "Public benchmarks", "Cyber capabilities index",
         "Furniture-assembly benchmark", "Private benchmarks", "Training compute, 2024 estimate", "Benchmarks, 2024 estimate"]
fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw=dict(width_ratios=[1.42, 1]))
ax = axes[0]; ys = np.arange(len(lag))[::-1]
for y, i, n in zip(ys, order, names):
    _, lo, hi, src = lag[i]; old = src == "epoch2024"; col = GREY if old else BLUE
    ax.plot([0, 24], [y, y], color=TRACK, lw=1.2, zorder=1)
    if i == 6: ax.plot([6, 22], [y, y], color=GREY, lw=1.0, zorder=2)
    if hi > lo: ax.plot([lo, hi], [y, y], color=col, lw=5.2, solid_capstyle="round", zorder=3)
    else: ax.plot(lo, y, "o", ms=6.8, color=col, mec="white", mew=1.1, zorder=4)
    lab = f"{lo}" if hi == lo else f"{lo}–{hi}"
    ax.text(hi + 1.1, y + (0.3 if i == 6 else 0), lab, va="center", fontsize=7.7, color=INK, path_effects=HALO, zorder=6)
    ax.text(-27.5, y, n, va="center", color=INK, fontsize=7.6)
bare(ax, ticks=[0, 6, 12, 18, 24], lim=(-28, 27)); ax.set_yticks([]); ax.set_ylim(-0.75, len(lag) - 0.3)
panel(ax, "(a) Lag behind the closed frontier, months"); ax.set_xlabel("Months", x=0.75)
ax = axes[1]; aa = OS["aa_index"]; ys = np.arange(len(aa))[::-1]
for y, r in zip(ys, aa):
    ax.plot([0, 70], [y, y], color=TRACK, lw=1.2, zorder=1)
    ax.plot([r["open"], r["closed"]], [y, y], color=GREY, lw=2.2, zorder=2, solid_capstyle="round")
    ax.plot(r["closed"], y, "o", ms=6.8, color=SUB, mec="white", mew=1.1, zorder=4)
    ax.plot(r["open"], y, "D", ms=5.8, color=TEAL, mec="white", mew=1.1, zorder=4)
    ax.text(r["closed"] + 3, y, f'{r["closed"]}', va="center", fontsize=7.7, color=INK, path_effects=HALO, zorder=6)
    ax.text(r["open"] - 3, y, f'{r["open"]}', va="center", ha="right", fontsize=7.7, color=INK, path_effects=HALO, zorder=6)
    if "unreleased" in r:
        ax.plot(r["unreleased"], y, "D", ms=5.4, mfc="white", mec=TEAL, mew=1.2, zorder=5)
        ax.text(r["unreleased"], y - 0.3, f'{r["unreleased"]}', va="center", ha="center", fontsize=7.0, color=SUB, zorder=6)
    ax.text(0, y + 0.3, r["date"], va="center", color=INK, fontsize=7.6, fontweight="bold")
    ax.text(70, y + 0.3, f'gap {r["gap"]}', va="center", ha="right", color=SUB, fontsize=7.4)
bare(ax, ticks=[0, 20, 40, 60], lim=(-1, 71)); ax.set_yticks([]); ax.set_ylim(-0.75, len(aa) - 0.3)
panel(ax, "(b) Composite index score"); ax.set_xlabel("Index points")
fig.legend(handles=[Patch(fc=BLUE, label="Estimates of 2025–26"), Patch(fc=GREY, label="Estimates of 2024"),
                    Line2D([], [], marker="o", ls="", color=SUB, mec="white", ms=6.8, label="Best closed model"),
                    Line2D([], [], marker="D", ls="", color=TEAL, mec="white", ms=5.8, label="Best open-weight model"),
                    Line2D([], [], marker="D", ls="", mfc="white", mec=TEAL, mew=1.2, ms=5.4, label="Weights not released")], loc="lower left",
           bbox_to_anchor=(0.0, 0.0), ncol=5, frameon=False, handletextpad=0.35, columnspacing=1.15, handlelength=1.1, handleheight=0.8, fontsize=7.3)
fig.subplots_adjust(left=0.01, right=0.995, top=0.925, bottom=0.215, wspace=0.07)
save(fig, "fig_gap")

# ------------------------------------------------------------------ 13  who makes the weights, who sells the tokens
S = OS["seller"]; G = OS["gateway_family"]
hqa = S["hq_all_tokens"]; tp = S["third_party_hq"]
stacks = [("By origin of the model family", [("China", G["china_origin"]), ("US", G["us_origin"]), ("Other", G["eu_origin"]), ("NL", round(100 - G["china_origin"] - G["us_origin"] - G["eu_origin"], 1))]),
          ("By headquarters of the seller", [("China", hqa["China"]), ("US", hqa["US"]), ("Other", round(hqa["Singapore"] + hqa["Other"] + hqa["EU"], 1)), ("NL", round(100 - hqa["China"] - hqa["US"] - hqa["Singapore"] - hqa["Other"] - hqa["EU"], 1))]),
          ("Third-party hosts only", [("China", tp["China"]), ("US", tp["US"]), ("Other", round(sum(v for k, v in tp.items() if k not in ("US", "China", "n/l")), 1)), ("NL", round(100 - sum(v for k, v in tp.items() if k != "n/l"), 1))])]
SCOL = {"China": ORANGE, "US": BLUE, "Other": "#8b8982", "NL": GREY_T}
fig = plt.figure(figsize=(W, 3.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[1.5, 1], wspace=0.1, hspace=0.78)
ax = fig.add_subplot(gs[0, 0]); ys = [2, 1, 0]
for y, (n, segs) in zip(ys, stacks):
    x = 0
    for kname, v in segs:
        ax.barh(y, v, left=x, height=0.46, color=SCOL[kname], zorder=3, edgecolor="white", linewidth=1.2)
        if v >= 6: ax.text(x + v/2, y, f"{v:.1f}", ha="center", va="center", fontsize=7.6, color="white" if kname != "NL" else INK, fontweight="bold", zorder=5)
        x += v
    ax.text(0, y + 0.42, n, va="center", color=INK, fontsize=7.7)
bare(ax, lim=(0, 100)); ax.xaxis.grid(False); ax.set_yticks([]); ax.set_ylim(-0.42, 2.72)
panel(ax, "(a) The same open-weight tokens, counted twice"); ax.set_xlabel("Share of tokens, %")
fig.legend(handles=[Patch(fc=ORANGE, label="China"), Patch(fc=BLUE, label="United States"), Patch(fc="#8b8982", label="Other countries"), Patch(fc=GREY_T, label="Not listed")],
           loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=4, frameon=False, handlelength=1.1, handleheight=0.8, columnspacing=1.3)
ax = fig.add_subplot(gs[1, 0])
hh = [("By model family", S["hhi_family"], SUB), ("By seller", S["hhi_seller"], SUB)]
for y, (n, v, c) in zip([1, 0], hh):
    ax.barh(y, v, height=0.5, color=c, zorder=3); ax.text(v + 70, y, f"{v:,}", va="center", fontsize=7.8, color=INK, fontweight="bold", path_effects=HALO, zorder=6)
    ax.text(-1500, y, n, va="center", color=INK, fontsize=7.7)
bare(ax, ticks=[0, 1000, 1800, 3000], lim=(-1520, 4000)); ax.set_yticks([]); ax.set_ylim(-0.6, 1.6); ax.xaxis.grid(False)
for x in (1000, 1800): ax.plot([x, x], [-0.5, 1.5], color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
ax.set_xticklabels(["0", "1,000", "1,800", "3,000"])
panel(ax, "(c) Concentration of those tokens, HHI")
ax = fig.add_subplot(gs[:, 1])
ls = [(f, v) for f, v in S["lab_served_by_family"].items() if v > 0]
forg = {"Hunyuan": "China", "Nemotron": "US", "MiniMax": "China", "MiMo": "China", "Qwen": "China", "Kimi": "China", "Mistral": "Other", "DeepSeek": "China", "GLM": "China"}
gwT = {"DeepSeek": 81.1, "Hunyuan": 18.8, "GLM": 17.9, "Nemotron": 8.16, "Kimi": 6.47, "MiniMax": 5.70, "MiMo": 3.44, "Qwen": 2.27, "Mistral": 0.519}
ys = np.arange(len(ls))[::-1]
for y, (f, v) in zip(ys, ls):
    ax.barh(y, 100, height=0.56, color=TRACK, zorder=1)
    ax.barh(y, v, height=0.56, color=OCOL[forg[f]][0], zorder=3)
    ax.text(v + 2.5, y, f"{v}", va="center", fontsize=7.6, color=INK, path_effects=HALO, zorder=6)
    ax.text(-3, y, f, va="center", ha="right", color=INK, fontsize=7.7)
    ax.text(137, y, f"{gwT[f]:.1f}", va="center", ha="right", color=SUB, fontsize=7.4)
ax.text(137, len(ls) - 0.28, "TOKENS, T", fontsize=6.4, color=MUTE, va="center", ha="right")
bare(ax, lim=(-33, 138)); ax.set_yticks([]); ax.set_ylim(-0.75, len(ls) - 0.05); ax.xaxis.grid(False)
for x in (25, 50, 75): ax.plot([x, x], [-0.45, len(ls) - 0.55], color="white", lw=0.7, zorder=2)
panel(ax, "(b) Tokens served by the lab itself, %"); ax.set_xlabel("Share of the family's tokens, %", x=0.5)
fig.subplots_adjust(left=0.01, right=0.995, top=0.935, bottom=0.2)
save(fig, "fig_serving")

# ------------------------------------------------------------------ 14  thirteen-month panel (revision 6)
PN = R["rev6"]["panel"]; xs = np.arange(13)
LABC = {"OpenAI": (ORANGE, "D"), "Google": (TEAL, "s"), "Anthropic": (BLUE, "o"), "Xiaomi": (AMBER, "^"), "Qwen": ("#8b8982", "v")}
fig, axes = plt.subplots(1, 2, figsize=(W, 2.9))
ax = axes[0]
for name in ("OpenAI", "Google", "Anthropic"):
    y = PN["web_series"][name]; col, mk = LABC[name]
    ax.plot(xs, y, "-" + mk, color=col, lw=1.8, ms=4.2, mec="white", mew=0.9, zorder=3)
    ax.text(12.45, y[-1], f"{name}  {y[-1]:.1f}", va="center", color=INK, fontsize=7.6)
ax.text(0, 79.5, "75.1", ha="center", color=INK, fontsize=7.6)
ax.axhline(50, color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=2); ax.text(-0.45, 51.8, "50%", color=SUB, fontsize=7, va="bottom")
bare(ax, grid="y", lim=None); ax.set_xlim(-0.6, 17.6); ax.set_ylim(0, 84); ax.set_yticks([0, 25, 50, 75])
ax.set_xticks([0, 3, 6, 9, 12]); ax.set_xticklabels(["Oct 2025", "Jan", "Apr", "Jul", "Oct 2026"])
panel(ax, "(a) Consumer web visits by lab, %")
ax = axes[1]
for name in ("Qwen", "Google", "OpenAI", "Anthropic", "Xiaomi"):
    y = PN["dev_series"][name]; col, mk = LABC[name]
    ax.plot(xs, y, "-" + mk, color=col, lw=1.6, ms=4.0, mec="white", mew=0.9, zorder=3 if name != "Xiaomi" else 4)
ax.text(9, 74.5, "Xiaomi  70.3", ha="center", color=INK, fontsize=7.6, path_effects=HALO)
ax.text(0.35, 45.5, "Qwen  42.6", ha="left", color=INK, fontsize=7.6, path_effects=HALO)
ax.text(12.45, 30.39, "Google  30.4", va="center", color=INK, fontsize=7.6)
ax.axhline(50, color=SUB, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
bare(ax, grid="y", lim=None); ax.set_xlim(-0.6, 17.2); ax.set_ylim(0, 84); ax.set_yticks([0, 25, 50, 75]); ax.set_yticklabels([])
ax.set_xticks([0, 3, 6, 9, 12]); ax.set_xticklabels(["Oct 2025", "Jan", "Apr", "Jul", "Oct 2026"])
panel(ax, "(b) Developer tokens on one gateway by lab, %")
fig.legend(handles=[Line2D([], [], marker=LABC[n][1], ls="-", lw=1.6, color=LABC[n][0], mec="white", ms=5, label=n) for n in ("OpenAI", "Google", "Anthropic", "Xiaomi", "Qwen")],
           loc="lower left", bbox_to_anchor=(0.03, 0.0), ncol=5, frameon=False, handlelength=1.8, columnspacing=1.6)
fig.subplots_adjust(left=0.045, right=0.995, top=0.905, bottom=0.2, wspace=0.06)
save(fig, "fig_panel")
