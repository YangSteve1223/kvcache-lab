# SpectrumKV Project Context

> **Purpose**: This file enables any AI agent to clone this repo and immediately understand the full project context, decisions, and current state — as if it had been working on the project from the start.

## Project Phase (Updated 2026-06-07)

**Current**: Paper completed (28 pages), arXiv submission in progress, targeting MLSys 2027 (deadline ~2026-10-30).

- ✅ GPU experiments: 3 models (Qwen2.5-7B, Mistral-7B, Gemma-2-9B) complete
- ✅ Paper: `paper/main_v2_final.tex` (28p, natbib super+square, zero overfull)
- ✅ Three rounds of review + polishing (4 versions: polish_a/b/c/d)
- ✅ Figure/Table fixes (arrows, subscripts, on-demand direction)
- ✅ OS narrative added (Intro/Discussion/Related Work)
- ✅ Appendix with raw numerical data (6 tables)
- ⏳ arXiv submission (metadata page — use pure ASCII for title/abstract)
- ⏳ OSF preprint → advisor endorsement → arXiv
- 🔜 After enrollment: vLLM plugin → e2e serving → top conference resubmission

## What is SpectrumKV

SpectrumKV is a **per-token mixed-precision KV cache transfer policy** for PD (prefill-decode) disaggregated LLM serving.

### Three Components
1. **SWS (Semantic Working Set)**: Importance scoring for each token's KV cache entry. SWS is a **component** of SpectrumKV, not a separate method.
2. **QCBM (Quality-Constrained Bandwidth Minimization)**: Greedy precision allocation under a transfer budget constraint. Assigns FP16/INT8/INT4 per token.
3. **Probe**: Lightweight deployment-time detection of INT4 tolerance. Runs 3 aggressive NIAH trials under 3-tier policy. Models that pass → FP16+INT8+INT4; models that fail → FP16+INT8 (2-tier).

### Key Distinction from Prior Work
- **Prior work** (PDTrim, H2O, SWS as eviction): Binary keep/drop decision. Dropped tokens are gone forever.
- **SpectrumKV**: Precision spectrum. All tokens are retained, just at different precision. Low-importance tokens can be upgraded on-demand.

⚠️ **Critical naming rule**: SpectrumKV = SWS + QCBM + Probe. These are NOT "v2", "improvements", or "extensions" of SWS. SpectrumKV IS the core algorithm. Never use "v2/改进/沿用" to describe SpectrumKV.

## Key Experimental Results

### GPU Verification (WikiText-2 PPL, seq=2048)

| Model | Budget | SpectrumKV | PDTrim | SWS Original |
|-------|--------|-----------|--------|-------------|
| Qwen2.5-7B | b=0.5 | +1.97% | +25.85% | +30.9% |
| Mistral-7B | b=0.5 | -0.06% | +22.07% | +33.2% |
| Gemma-2-9B | b=0.5 | -0.44% | +35.63% | +43.8% |

### NIAH Retrieval (b=0.3)
- SpectrumKV 2-tier on Qwen: 52.6% vs PDTrim 26.3% (+26pp)
- Mistral/Gemma 3-tier: 100% at b=0.3-0.7
- ⚠️ 52.6% is the heuristic upper bound for 2-tier, not the theoretical limit (oracle → 100%)

### Three Attention Locality Patterns
| Pattern | Model | Key trait |
|---------|-------|-----------|
| Local-dominant | Qwen2.5-7B/14B | Recent tokens dominate |
| Sink-dominant | Mistral-7B | First few tokens (attention sinks) dominate |
| Hybrid | Gemma-2-9B | Both sink and recent matter; sink protection essential |

### Quantization Safety
- **INT8**: Safe across all models (cosine > 0.9999)
- **INT4**: Model-dependent. Qwen catastrophically fails; Mistral/Gemma stable.
- **b=0.5 INT8 convergence**: At 50% budget, Uniform_INT8 ≈ SpectrumKV_Greedy because INT8 is lossless. Importance-aware advantage only shows at low budgets requiring INT4.
- **Per-layer variation**: INT4 cosine varies ~2.5x across layers in Qwen → supports per-layer budget allocation.
- **Key > Value sensitivity**: Key tensors are more sensitive to quantization than Value tensors.

### FP16 Baseline PPL (GPU, seq=2048)
- Qwen2.5-7B: 7.0415
- Mistral-7B: 7.9568
- Gemma-2-9B: 11.2042

### TTFT/TPS
- 50% budget → TTFT halved; TPS impact < 11%

## Academic Writing Rules (MANDATORY)

### Red Lines ❌
1. **Never fabricate data.** Paper figures must use only real experimental data.
2. **Never use "首次/redefine/从根本上规避/打破认知"** — let data speak.
3. **SWS_Original is actually WORSE than PDTrim** (PPL +30.9% vs +25.8%). The differentiation is mixed precision vs. binary selection.
4. **On-demand pull is NOT implemented.** Do not write it as an implemented feature.
5. **INT4 tolerance is an evaluation finding, not a characterization conclusion.** Do not label it in Table 2 (circular reasoning).
6. **Never use "v2/改进/沿用" for SpectrumKV** — it IS the core algorithm.

### Style Preferences
- Precise data, no ranges (weakened methods)
- Don't mention low-level bugs in the paper
- Use data when available, don't summarize vaguely
- Reduce AI traces: no triple parallelism, clichés, mechanical sublimation
- ~1/3 passive → active voice
- "Surprisingly" for counterintuitive findings
- Figure/table references integrated into argument flow, not standalone

## GPU Operations Rules

- ⚠️ **NEVER delete models** (user explicitly requested 2026-05-28)
- Credentials in SECRET.md; data pushed to GitHub
- Scripts must be pre-reviewed before running
- Better to stop machine than corrupt data
- Code changes require process restart
- 📌 **Data discrepancy → check script first (not parameters)**: Multiple NIAH script errors found this way
- 📌 **NIAH 8 bugs fixed**: chat_template + Mistral system_message; see experiment_log_v2.md
- 📌 **PD separation**: Single-card hook equivalent; e2e needs 2 cards (future work)
- ⚠️ **Avoid result-oriented programming**: Building narrative first then verifying is risky

## Competitive Landscape

### Direct Competitors
- **PDTrim** (2025): First/last token truncation. Our baseline.
- **OrbitFlow** (VLDB'26): ILP + real vLLM. **Most dangerous competitor.**

### Orthogonal/Complementary
- KVServe (SIGCOMM'26), SplitZip, LMCache, CacheBlend

### Industrial
- NVIDIA Dynamo, vLLM NIXL/MORI-IO

### Honest Assessment
- No real e2e serving system → not enough for top conference
- OrbitFlow has real system implementation → biggest risk
- Recommended path: arXiv first, then e2e implementation after enrollment

## Key Decisions Log

### OS Narrative Positioning (2026-05-29)
- OS is inspiration and narrative entry, NOT the main title or method name
- Three natural embedding points: Introduction (motivation analogy), Discussion (concept mapping), Related Work (Denning/PagedAttention)
- Mapping: Working Set → SWS, Virtual Memory Pages → Precision Tiers, Page Fault → On-demand Upgrade, Swap Space → INT4 Cold Tier

### SpectrumKV Naming (2026-05-31)
- SWS = Semantic Working Set (component)
- SpectrumKV = core algorithm = SWS + QCBM + Probe
- Never call it "SWS v2" or "improved SWS"

### Adaptive Precision Tier (2026-06-04)
- Original: hard-coded 2-tier
- Probe: automatic 2/3-tier selection based on INT4 tolerance detection
- This IS SpectrumKV's adaptive mechanism, not a separate version

### Paper Self-Assessment (2026-05)
- Evidence chain is weak (pure simulation / no e2e) → not enough for top conference
- Recommended: arXiv first, solid version after enrollment
- Gap analysis: P0 lacks e2e + strong baselines; P1 lacks ablation + 14B/long context + deeper theory

## Repository Structure Guide

```
paper/main_v2_final.tex     → ★ LATEST paper (28p, compile with pdflatex)
paper/spectrumkv_v2.bib     → BibTeX source (22 references)
paper/main_v2_final.bbl     → Compiled refs (needed for arXiv)
paper/arxiv_metadata.txt    → Pure ASCII title/abstract for arXiv submission
paper/REVIEW_CRITERIA.md    → Review audit criteria used in 3 review rounds

gpu-experiments/exp_spectrumkv_gpu.py  → ★ Main SpectrumKV GPU experiment
gpu-experiments/exp_qcbm_quantization.py → QCBM quantization experiment
gpu-experiments/results/     → GPU experiment result JSONs
gpu-experiments/EXPERIMENT_PLAN.md → Experiment planning doc

spectrumkv_data/ALL_EXPERIMENT_DATA.md → ★ Comprehensive 317KB data document
spectrumkv_data/exp1-4_*.json         → Per-model PPL/NIAH/cosine data
spectrumkv_data/raw/                  → Unprocessed experiment logs
spectrumkv_data/experiment_log_v2.md  → Experiment log organized by RQ

docs/learning-guide/         → Learning guide v3 (Word)
```

## Compilation Instructions

```bash
cd paper
pdflatex main_v2_final.tex
bibtex main_v2_final
pdflatex main_v2_final.tex
pdflatex main_v2_final.tex
# Result: main_v2_final.pdf (28 pages, zero errors, zero overfull)
```

- Use **pdflatex** (NOT xelatex). No Chinese packages needed.
- natbib config: `\usepackage[super,square,sort&compress]{natbib}` → citations as ^[1]
- All figures are TikZ/pgfplots code — no external image files needed.

## arXiv Submission Notes

- Only need `.tex` + `.bbl` (no external images)
- Title and Abstract fields: **pure ASCII only** (no LaTeX commands, no Unicode dashes)
- Replace: `\method{}`→SpectrumKV, `\fp`→FP16, `\intviii`→INT8, `\intiv`→INT4, `\niah{}`→NIAH, `\budget`→b, `\pct{X}`→X%
- Use `-` not `–` (en-dash) in metadata fields
- arXiv categories: cs.DC + cs.CL

## Next Steps

1. Complete arXiv submission
2. OSF preprint placeholder → advisor endorsement → arXiv publish
3. After enrollment (2026-09): vLLM plugin → e2e serving → top conference
4. Potential improvements: per-head scoring, 70B+ models, OrbitFlow/KVQuant combination

## User Profile

- 杨鹏举 | JLU 2022 Communication Engineering | 北航电子信息 2026硕士
- GitHub: github.com/YangSteve1223
- Advisor: 王勇 (JLU), dual-affiliation with BUAA
- 优毕设 (outstanding thesis), 软著(student first author)
- Research direction: PD separation KV memory management
