# gold_scar's direction model

`autotrader/options/builtin/gold_scar.py` (v2) trades the predictions of `autotrader/options/ml/`: a ridge model
plus three small causal transformers. Together they read SPY, QQQ, IWM and seven cross-asset ETFs (DIA, TLT, HYG,
SMH, XLF, XLK, GLD) in 5-minute bars and predict each target's return to the 15:20 flatten. Inference is plain
numpy; the weights ship as `autotrader/options/ml/gold_scar.npz`.

Retrain (needs a CUDA GPU for the transformers, about 25 minutes):

    uv run python research/gold_scar/download.py          # Alpaca 5m bars since 2016 + CBOE VIX history
    uv run --with torch --with scikit-learn python research/gold_scar/train.py 3

Changing the weights changes the strategy file's behaviour without changing its source, so bump
`GoldScar.version` when you ship new weights.

## What the research found (Oct 2026)

- Out-of-sample IC of the ensemble at 9:45 ET, by year 2019-2026: +0.09, -0.09, +0.02, +0.07, +0.09, +0.00,
  -0.01, +0.12. Pooled over every decision time: +0.02.
- The model was chosen on 2019-2023 IC alone, before looking at option results. Of three candidates it had the
  best IC there.
- Replay through `autotrader.options.sim` on real Alpaca 0DTE prints, Feb 2024 - Oct 2026 (142 weeks). Each year
  was traded by a fold trained only on earlier years:

  | variant | mean week | median | weeks up | worst |
  |---|---|---|---|---|
  | default | +1.8% | 0 | 37% (48% of traded weeks) | -39% |
  | 1.5x spread | +1.3% | 0 | 37% | -40% |
  | budget_pct 0.35 | +2.6% | 0 | 37% | -67% |

- Best documented intraday index strategies run about 1-2% a month. Sizing this signal at full Kelly gives
  about +4% in 20 days, and the chance of 200x in 20 days is zero at any leverage tested.
