# Repository structure

## What is pirate's and what is ours

**pirate** (Kendrick Smith, CHORD) is the folder above `charts/` (this repository is the CHARTS fork) and **was not
modified**. It does two things: it receives the data over the network (and keeps it in a ring buffer in RAM) and
dedisperses it on the GPU with the tree algorithm, producing an S/N map over DM and time. **Everything in `charts/` is
ours:** feeding pirate, interpreting its output, deciding the alerts and measuring.

## The chain and which file does each part

```
 1. SIMULATOR      2. EMITTER              3. PIRATE           4. GROUPER            5. SIFTER          6. EVALUATION
 simulator/     ─► emitter/emitter.py ──► (this fork)     ──► server/grouper/   ──► server/sifter/ ──► evaluation/
 generates the     mimics the X-engine     GPU                 grouper_v1.py         alert_sifter.py    evaluate_test_64b.py
 .fil files        + RFI + mask            dedispersion        events in the map     FRB ALERTS         verify_alerts.py
                          └──────────── everything is started by server/launcher.py ────────────────┘
```

## Folders

| Folder | Purpose | Needed to run the test? |
|---|---|---|
| `simulator/` | Generates the 4000 test `.fil` files (FRBs simulated with shapes from the CHIME catalog): `12_generate_comparison_set_21504.py` and its dependencies, catalogs and draws | yes, once (the data are not in git) |
| `quantization/` | 8 to 4-bit scaling: theory (`quantization_theory.py`), S/N measured on the set (`quantization_21504.py`), range loss (`fig_quantization_range.py`), file validation and figures. Backs `docs/ANALYSIS_pirate_quantization_…` | no |
| `configs/` | `test_64b.yml`: **the only config** of the test (beams, channels, network, memory, dedispersion, grouper, alerts, emitter). The others are the pirate configs of the reported runs | yes |
| `emitter/` | The standalone emitter (no pirate): `protocol.py` (bytes of the network protocol), `quantization.py` (8 → 4 bits, mask), `sources.py` (where the data come from), `emitter.py` (processes + connections) | yes |
| `server/` | `launcher.py` (builds pirate's configs from the YAML and starts sifter, grouper and pirate; modes `server`, `emitter`, `local`), `grouper/` (grouper v1 and v0) and `sifter/alert_sifter.py` (decides what is an FRB and emits the alerts; pirate's FrbSifter gRPC service) | yes |
| `evaluation/` | Analysis of the runs (see the table below) | yes (`evaluate_test_64b.py`) |
| `site_rfi/` | RFI measured at the site: ×32 channelization, RFI model, masks m1/m2, injection (`site_rfi.py`, used by the emitter with `--rfi`), .fil header reading (`sigproc.py`) | only with RFI |
| `offline_search/` | `pirate_fil_search.py`: search a single .fil with pirate, without the network | no |
| `rfi_analysis/` | Analysis of the runs with RFI (counting without caps, chance coincidences, full runs with mask) and per-process resource monitor | no |
| `scaling/` | pirate's memory and capacity versus maximum DM, block size and `max_gpu_clag` | no |
| `docs/` | Results, protocol and Kotekan adaptation, pirate installation and internals, quantization | to read |
| `figures/` | Examples: FRB with and without RFI, channelization, counts | to read |

## Adapting to another architecture

- **Another data source** (live, another format): a class with `block(t0, T, out)` in `emitter/sources.py`.
- **Another emitter** (Kotekan, FPGA): implement `emitter/protocol.py` in that language; see
  `docs/PROTOCOL_pirate_and_Kotekan_adaptation.md`.
- **Another search machine:** change `network.receivers` (IP of the search machine) and the memory in the YAML; pirate is autodetected (active environment, `CHARTS_PIRATE_ENV` or `environments.pirate`).
- **Another number of beams or channels:** `test.beams` / `test.channels` in the YAML; the launcher builds pirate's configs.
- **Another alert rule or downstream action:** `server.alerts` in the YAML (`threshold`, `dm_min`, `max_beams`, `action`), or replace `server/sifter/alert_sifter.py` with any FrbSifter gRPC service.

## Analysis of the runs (`evaluation/` and `rfi_analysis/`)

All of them read a run folder (the one the launcher leaves) and write their tables in it or in `<run>/analysis/`.

| Script | What it answers |
|---|---|
| `evaluation/evaluate_test_64b.py` | **The main one:** completeness per set, false alarms per beam·hour and FP in negatives, at fixed thresholds and at equal FA rate → `results.md` |
| `evaluation/verify_alerts.py` | Each alert in `alerts.jsonl` against the injected FRBs: true, non-FRB, false alarms, FRBs without alert and latency → `alerts.md` |
| `evaluation/standard_rates.py` | Completeness at 10 FA per beam·hour and at 10 FA/h of the system (`--log all_events.log` for the full log) |
| `evaluation/grouper_profile.py` | Detection time per block: grouper working vs waiting for pirate, per stage (requires `profile: true`) |
| `evaluation/resource_summary.py` | GPU, RAM and CPU per process (from `rfi_analysis/process_monitor.py`) |
| `evaluation/summarize.sh` | Quick summary of `server.log`: blocks, equivalent beams, Gbit/s |
| `evaluation/evaluate_when_done.sh` | Waits for a run to finish and evaluates it |
| `evaluation/loss_analysis_64b.py`, `nearby_events_64b.py` | Missed FRBs and events near each file |
| `evaluation/show_windows.py` | Shows windows of the DM-time map saved by the grouper |
| `evaluation/map_statistics/` | Map statistics: noise vs signal, measured vs expected S/N (`map_statistics.py`) and why each FRB is missed (`missed.py`) |
| `evaluation/subbands/` | Full band vs subbands `[0,0,1,2,1]` file by file (`paired_comparison.py`; config `configs/charts_64b_dm1000_sb1.yml`) |
| `rfi_analysis/analyze_all.py` | Count of all candidates with and without RFI (without the grouper caps) |
| `rfi_analysis/compare_diag.py` | Grouper and mask variants, with correction for chance hits |
| `rfi_analysis/analyze_full_runs.py` | Full runs with mask: recall, FA and chance hits |
| `rfi_analysis/process_monitor.py` | Logs per-process resources every 5 s during a run |

Variables: `CHARTS_RUN` (default run of `map_statistics/`), `CHARTS_ANALYSIS_OUT` (where they write),
`CHARTS_SUBBAND_RESULTS` and `CHARTS_RFI_RESULTS` (folders with several runs to compare). The `rfi_analysis/` scripts
are run with the Python of pirate's environment. If a variable is missing, the script says so and exits.
