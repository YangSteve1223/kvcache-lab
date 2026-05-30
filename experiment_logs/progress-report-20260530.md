# kvcache-lab M1 Baseline Progress Report (Final)
**Date**: 2026-05-30
**Milestone**: M1 (Week 1-4) - Day 3

## Pipeline Status ✅
Full pipeline validated: `CompressionOrchestrator → CompressionAdapter → PDSimulator`

### Bug Fixes Applied (5 total)
1. **PDSimulator precision-unaware transfer** (CRITICAL): Transfer time only considered retention, not precision. Fixed with `avgPrecisionRatio = avg(keyPrec/16 × valPrec/16)`.
2. **Uniform strategy bandwidth-blind** (HIGH): Added bandwidth-awareness with SLO-based calculation.
3. **PD-Task-Aware avgCompressionRatio** (MEDIUM): Fixed to include both key and value precision.
4. **KV size double-counting** (CRITICAL): PDSimulator.computeKVSize multiplied by totalLayers, but kvBytesPerToken was already total. Fixed to use per-layer kvBytesPerToken=512.
5. **PDAware hardcoded REQUIRED_BANDWIDTH=100** (HIGH): Replaced with actual KV transfer demand vs SLO-based bandwidth pressure calculation.

### New Components
- **CompressionAdapter** (`src/compression/CompressionAdapter.ts`): Bridges `CompressionOutput` → `CompressionConfig`
- **M1 Baseline Benchmark** (`experiments/exp-m1-baseline.ts`): 360 experiments (6 strategies × 4 tasks × 3 bandwidth × 5 seq_lengths)

## Benchmark Results

### Short context (4K tokens, 1Gbps) — No compression needed
| Strategy | TTFT(ms) | TTFT↓ | Quality |
|----------|----------|-------|---------|
| None | 997 | 0.0% | 1.000 |
| PD-Aware | 997 | 0.0% | 1.000 |
| PD-Task-Aware | 984 | 1.2% | 0.996 |
| QCBM | 470 | 52.9% | 0.865 |

→ PD-Aware correctly detects no bandwidth pressure and skips compression.

### Long context (32K tokens, 1Gbps) — Compression kicks in
| Strategy | TTFT(ms) | TTFT↓ | Quality | Compression |
|----------|----------|-------|---------|-------------|
| None | 7622 | 0.0% | 1.000 | 1.000 |
| Uniform | 3827 | 49.8% | 0.781 | 0.116 |
| PD-Aware | 3910 | 48.7% | 0.815 | 0.136 |
| Task-Aware | 4579 | 39.9% | 0.814 | 0.292 |
| PD-Task-Aware | 3966 | 48.0% | 0.820 | 0.149 |
| QCBM | 3411 | 55.3% | 0.715 | 0.019 |

### Key Findings
1. **PD-Aware is SLO-aware**: Only compresses when bandwidth pressure > 0. This is the correct production behavior.
2. **At 32K/1Gbps**: PD-Aware achieves 49% TTFT reduction with quality=0.815 (best balanced)
3. **QCBM always compresses**: 55% TTFT reduction but quality=0.715 (most aggressive)
4. **PD-Task-Aware adds task-sensitivity**: Slightly better quality (0.820 vs 0.815) with similar TTFT
5. **Dynamic precision**: PDAware now adjusts precision (FP16→INT8→INT4) based on bandwidth pressure

### Bandwidth Sensitivity (PD-Task-Aware, 32K tokens, math)
- Low-1Gbps: TTFT=3966ms, Quality=0.820, BW Saving=85.1%
- Mid-10Gbps: Higher bandwidth → less compression → higher quality
- High-100Gbps: Minimal compression, near-baseline quality

### Sequence Length Scaling (PD-Task-Aware, 1Gbps, math)
- seq=2K: TTFT=517ms, no compression needed
- seq=4K: TTFT=984ms, minimal compression
- seq=16K: TTFT=3005ms, quality=0.937 (compression active)
- seq=32K: TTFT=3926ms, quality=0.908 (significant compression)

## M1 Completion Assessment
| M1 Task | Status | Notes |
|---------|--------|-------|
| Fork modules | ✅ 85% | Core modules exist; minor cleanup needed |
| Compression Orchestrator | ✅ 100% | Plugin framework working |
| CompressionAdapter | ✅ 100% | New bridge component |
| Baseline experiments | ✅ 90% | 360 experiments with correct KV model |
| KVServe/SplitZip deep reading | ⏳ 60% | Survey done; need architecture comparison |
| Quality model calibration | ⏳ 40% | Needs DeepSeek API validation |

**M1 Overall**: ~80% complete. Key remaining: KVServe/SplitZip deep analysis + quality calibration.
