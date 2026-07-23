# Streaming Measurement of the Microphone-Domain False-Alarm Penalty in a Deployed Keyword-Spotting Sensor

Code, trained models, measured transducer response, and per-result data accompanying the
IEEE Transactions on Instrumentation and Measurement manuscript
*"Streaming Measurement of the Microphone-Domain False-Alarm Penalty in a Deployed
Keyword-Spotting Sensor: Per-Clip Accuracy Is the Wrong Instrument."*

The paper treats a deployed 75k-parameter INT8 wake-word sensor (InvenSense INMP441 MEMS
microphone on a Raspberry Pi) as a **measurement problem**: per-clip AUC saturates at
~0.9999 and is blind to a ~3.3x microphone-domain false-alarm penalty that only a
**streaming measurement through the sensor's own decision logic** reveals. A single
swept-sine (Farina) measurement characterizes the deployment transducer and yields a
magnitude-only response-equalization filter. The detector reuses established building
blocks (depthwise-separable convolutions, squeeze-and-excitation, coordinate attention,
a streaming PCEN front-end, INT8 QAT); no architectural novelty is claimed — the
contributions are the measurement protocol and the measured filter.

## Layout

### `measurement/` — transducer characterization
- `twin_fir.npy` — the measured conventional→INMP441 magnitude-response FIR *H* (16 kHz, minimum-phase).
- `reference.wav` — the 9 s log-sine sweep stimulus (50 Hz–7.5 kHz) used for the swept-sine measurement.
- `inmp_fanon.wav`, `inmp_fanoff.wav` — fan on/off INMP441 recordings behind the Limitations' self-noise note (a ~18.5 dB tonal peak near 1.7 kHz with band totals and broadband RMS essentially unchanged).

### `models/`
- `3class_qat_final_int8.tflite` — the deployed INT8 detector (~182 KB, 75,075 params) → **Table I (`tab:bench`)**, **Table `tab:fa`**.
- `fulltrained/A{0_conv,1_generic,2_twin,3_randtwin}_s{99,100,101}_int8.tflite` — the four fully-trained arms × three seeds → **Table `tab:fulltrained`** (three-seed mean). Seed **s99** additionally drives the 35.7 h real-device McNemar.
- `sweep_arms/{conventional,eq_agnostic,measured_filter,micaugment}.tflite` — the four speaker-disjoint arms → **Fig. 2** threshold sweep and **Table `tab:speakerdisjoint`**.

### `src/` — front-end, training, and evaluation
Front-end / core: `common.py` (32-band mel + streaming PCEN front-end, bit-identical to the C++ deployment to 1e-6; and the model), `tflite_utils.py`, `macs.py`.
Filter + corpus synthesis: `build_twin_filter.py`, `build_twin_filter_v2.py` (build *H*), `build_twin_cache.py` (synthesize the INMP441-domain corpus), `twomic_ratio.py`, `analyze_domainshift.py` (**Fig. 1** two-mic response).
Training: `train_ablation.py`, `run_twin_ladder.py` (cross-domain train×test matrix → `tab:fulltrained` penalty), `train_baselines.py` + `models_baseline.py` (**Table I**), `spk_disjoint.py` + `build_spkdisjoint_cache.py` (speaker-disjoint rebuild), `exp4_export_and_far.py` (export ladder → INT8).
Streaming evaluation: `eval_far.py`, `eval_stream_arms.py`, `eval_stream_roc.py`, `eval_canonical.py`, `eval_pooled_device16.py`, `eval_pooled_significance_canonical.py`, `eval_extended_device16.py`, `eval_extended_session.py`, `eval_room3_matched_v2.py`.
Fully-trained real-device tests (**§ Results, 35.7 h McNemar**): `exp3_ft_streaming.py`, `exp5_ft_matched.py`, `exp6_ft_paired.py` (paired McNemar + episode clustering), `exp7_session2_score.py`, `exp7_session3_score.py`, `exp8_ft_stress.py`.
Threshold sweep (**Fig. 2**): `run_far_sim_sweep.py` (inference once, replays the Schmitt/cooldown trigger across a threshold grid; self-check reproduces the 0.60 counts), `run_far_sim_all_arms.py` (fixed-threshold sibling → `tab:speakerdisjoint`).
Stats / other: `stats_review.py` (Clopper–Pearson, Fisher, McNemar — `tab:fa`, § where-FA), `exp_power.py` (4363-clip powered pool, § twin), `exp2_mcnemar_curves.py` (10 h episode McNemar / operating curves → `tab:live`).

### `results/` — the numbers behind each table/figure
- `baseline_*.json`, `macs.json`, `ablation_3class_qat_final.json` → **Table I**.
- `fulltrained/A*_s*.json` (12) → **Table `tab:fulltrained`** three-seed mean.
- `fulltrained/paired_10h.json`, `matched_10h.json`, `session2_scored.json`, `session3_scored.json` → the **35.7 h real-device McNemar** (conv 15 vs measured 1; s99 fire times = 5 + 2 + 8 across the three sessions).
- `fulltrained/pooled_7pos_fulltrained.json`, `realmic_P{1..4}_*.json` → **Table `tab:twin`** (seven positions).
- `far_sweep_{conventional,eq_agnostic,measured_filter,micaugment}.json` → **Fig. 2** curve data (413.7 h). `far_sim_result_*_20260711_205248.json` → fixed-0.60 counts for **Table `tab:speakerdisjoint`**.
- `exp2/mcnemar_episodes.json`, `exp2/operating_curves.json` → **Table `tab:live`** (10 h episode McNemar, p < 1e-4).

## Reproducing

Front-end/model constants (16 kHz, 512-pt FFT, 320 hop, 32 mel 60–6000 Hz, PCEN α=0.90 δ=2.0 r=0.5 τ=0.10 s; streaming EMA α=0.3, Schmitt TH_UP/TH_DOWN=0.60/0.30, margin 0.20, cooldown 7 strides) are defined in `common.py` and must not be changed. The Fig. 2 sweep runs from the committed `models/sweep_arms/` models; set `STRESS_CACHE_DIR` to a directory of per-file `*.npz` PCEN features to supply the 413.7 h negative corpus (see Data availability).

## Data availability

- The **keyword recordings** and all **live/replayed session audio** are identifiable
  human-voice data recorded and submitted voluntarily for this work, without permission for public redistribution,
  and are **withheld to protect speaker privacy**.
- The **413.7 h stress corpus**, cached feature `*.npz` files (8.7 GB), and the raw
  measurement/session WAVs are **too large to distribute**; the non-keyword classes derive
  from publicly available corpora (Speech Commands, ESC-50, VocalSound) plus recorded
  non-keyword speech.
- Evaluation scripts other than the Fig. 2 sweep expect local paths to this withheld
  audio; they document the exact protocol and can be applied to any keyword/microphone.
