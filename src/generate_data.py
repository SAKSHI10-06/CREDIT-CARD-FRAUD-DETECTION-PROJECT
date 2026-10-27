"""
Extend dataset.csv (500 rows) to 5,000 rows with SYNTHETIC transactions.

Run from the project root:
    python src/generate_data.py

Design (document this in your report):
- Original 500 rows are kept unchanged (trans_id 1-500).
- Rows 501-5000 are synthetic and learn their distributions from the original
  NORMAL rows (category mix, card-type mix, amount scale, hour-of-day pattern).
- Synthetic fraud is generated at ~3% of the new rows so the whole file ends
  near 3% fraud (inside the 2-5% range in the brief). The fraud amount scale
  comes from the 6 real fraud rows; the other fraud patterns (rapid repeat
  transactions on one card, occasional very large amounts) are ASSUMPTIONS
  and are labelled as such. Real-world fraud is far rarer (~0.17%).
- Timestamps are sequential, format YYYY-MM-DD HH:MM:SS. Card tokens are masked.
- Fixed random seed, so the file is reproducible.
"""
import numpy as np
import pandas as pd

SEED = 42
IN_PATH = "data/raw/dataset.csv"
OUT_PATH = "data/processed/dataset_extended.csv"
TARGET_ROWS = 5000
SYNTH_FRAUD_RATE = 0.032          # of the synthetic rows

rng = np.random.default_rng(SEED)
orig = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
normal = orig[orig.is_fraud == 0]
fraud = orig[orig.is_fraud == 1]

n_new = TARGET_ROWS - len(orig)
n_fraud = int(round(n_new * SYNTH_FRAUD_RATE))
n_norm = n_new - n_fraud

# ---- distributions learned from the original data -------------------------
cat_p = normal.merchant_category.value_counts(normalize=True)
card_p = normal.card_type.value_counts(normalize=True)
hour_p = (normal.timestamp.dt.hour.value_counts(normalize=True)
          .reindex(range(24), fill_value=0) + 0.005)
hour_p = hour_p / hour_p.sum()

# amount scale per category (lognormal fit on normal rows)
la = np.log(normal.amount)
cat_stats = la.groupby(normal.merchant_category).agg(["mean", "std"]).fillna(la.std())
fraud_mu, fraud_sd = np.log(fraud.amount).mean(), np.log(fraud.amount).std()

# merchant -> category map (a merchant belongs to one category)
mer_map = dict(zip(orig.merchant_id, orig.merchant_category))
all_mer = [f"MER{i:04d}" for i in range(1, 501)]
cats = list(cat_p.index)
for m in all_mer:
    mer_map.setdefault(m, rng.choice(cats, p=cat_p.values))
mer_by_cat = {c: [m for m in all_mer if mer_map[m] == c] for c in cats}

# card pool (token has last 4 digits only = masked)
pool = [f"CARD_XXXX{rng.integers(0, 10000):04d}" for _ in range(1500)]
card_type_of = {t: rng.choice(card_p.index, p=card_p.values) for t in pool}
card_type_of.update(dict(zip(orig.card_token, orig.card_type)))

def make_row(is_fraud, card=None, cat=None, amount=None):
    cat = cat or rng.choice(cats, p=cat_p.values)
    if amount is None:
        if is_fraud:
            amount = np.exp(rng.normal(fraud_mu, fraud_sd))
        else:
            m, s = cat_stats.loc[cat]
            amount = np.exp(rng.normal(m, s))
    card = card or rng.choice(pool)
    return dict(card_token=card, amount=round(float(np.clip(amount, 1, 8000)), 2),
                merchant_category=cat, is_fraud=int(is_fraud),
                merchant_id=rng.choice(mer_by_cat[cat]),
                card_type=card_type_of[card])

rows = [make_row(0) for _ in range(n_norm)]

# fraud pattern A (60%): single suspicious transaction, amount from real fraud rows,
#                        a bit more likely in Electronics / Online (assumption)
# fraud pattern B (25%): burst - one card, 3 quick transactions (assumption)
# fraud pattern C (15%): very large amount outlier (assumption)
nA, nB, nC = int(n_fraud * .60), int(n_fraud * .25) // 3, n_fraud - int(n_fraud * .60) - (int(n_fraud * .25) // 3) * 3
fav = [c for c in ["Electronics", "Online"] if c in cats]
for _ in range(nA):
    c = rng.choice(fav) if rng.random() < .4 else None
    rows.append(make_row(1, cat=c))
for _ in range(nC):
    rows.append(make_row(1, amount=rng.uniform(1500, 5000)))
bursts = []
for _ in range(nB):
    card = rng.choice(pool)
    for _ in range(3):
        bursts.append(make_row(1, card=card, amount=rng.uniform(150, 700)))

new = pd.DataFrame(rows)
new["burst"] = 0
new = pd.concat([new, pd.DataFrame(bursts).assign(burst=1)], ignore_index=True)
new = new.sample(frac=1, random_state=SEED).reset_index(drop=True)
# trim/pad to exactly n_new rows
new = new.iloc[:n_new].copy()

# ---- sequential timestamps: continue after the last original timestamp -----
start = orig.timestamp.max().normalize() + pd.Timedelta(days=1)   # synthetic block starts the next day
span_days = (orig.timestamp.max() - orig.timestamp.min()).total_seconds() / 86400
rate_per_day = len(orig) / span_days
gaps = rng.exponential(86400 / rate_per_day, size=len(new))
ts = start + pd.to_timedelta(np.cumsum(gaps), unit="s")
# nudge towards the learned hour-of-day pattern by resampling the hour
ts = pd.Series(ts).dt.floor("D") + pd.to_timedelta(
    rng.choice(24, size=len(new), p=hour_p.values), unit="h") + pd.to_timedelta(
    rng.integers(0, 3600, size=len(new)), unit="s")
new["timestamp"] = np.sort(ts.values)
# bursts: make the 3 rows of a card happen within minutes of each other
new = new.sort_values("timestamp").reset_index(drop=True)
for card, g in new[new.burst == 1].groupby("card_token"):
    t0 = g.timestamp.iloc[0]
    for k, idx in enumerate(g.index):
        new.loc[idx, "timestamp"] = t0 + pd.Timedelta(minutes=int(rng.integers(1, 8)) * k)
new = new.sort_values("timestamp").reset_index(drop=True)
assert (new.timestamp >= start).all() and len(new) == n_new

new["trans_id"] = np.arange(len(orig) + 1, len(orig) + len(new) + 1)
cols = ["trans_id", "timestamp", "card_token", "amount",
        "merchant_category", "is_fraud", "merchant_id", "card_type"]
out = pd.concat([orig[cols], new[cols]], ignore_index=True)
out["timestamp"] = pd.to_datetime(out.timestamp).dt.strftime("%Y-%m-%d %H:%M:%S")

out.to_csv(OUT_PATH, index=False)
print("Rows:", len(out), "| frauds:", out.is_fraud.sum(),
      f"({out.is_fraud.mean():.2%})")
print("Original rows kept:", len(orig), "| synthetic rows:", len(out) - len(orig))
print("Time range:", out.timestamp.iloc[0], "->", out.timestamp.iloc[-1])
