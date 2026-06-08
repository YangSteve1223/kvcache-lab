# SpectrumKV vLLM Plugin Integration Architecture

> **Version**: 0.1.0-draft  
> **Date**: 2026-06-04  
> **Status**: Architecture Blueprint  
> **Target**: vLLM V1 Disaggregated Serving (≥ v0.8.0) + NVIDIA Dynamo + NIXL  

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Core Interface Design](#3-core-interface-design)
4. [Data Flow Design](#4-data-flow-design)
5. [NIXL Transfer Layer Integration](#5-nixl-transfer-layer-integration)
6. [Configuration & API Design](#6-configuration--api-design)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Appendix](#8-appendix)

---

## 1. Introduction & Motivation

### 1.1 Problem Statement

SpectrumKV is a **Per-Token Mixed-Precision KV Cache Transmission Algorithm** designed for Prefill-Decode (PD) disaggregated serving scenarios. Its core components are:

| Component | Full Name | Responsibility |
|-----------|-----------|----------------|
| **SWS** | Semantic Importance Weighted Sorting | Rank tokens by semantic importance using attention pattern analysis |
| **QCBM** | Quantization Configuration Block Manager | Allocate per-token precision (FP16 / FP8 / INT4 / drop) based on SWS ranking |
| **Probe** | Adaptive Precision Probe | Runtime detection of precision degradation, trigger re-transmission at higher precision |

The paper's key shortcoming is **lack of real system integration**. This document provides the architecture blueprint for integrating SpectrumKV into vLLM's V1 disaggregated serving pipeline as a first-class KV Transfer Backend connector.

### 1.2 Design Goals

1. **Minimally Invasive**: Implement as a `KVConnectorBase_V1` subclass, requiring zero changes to vLLM core scheduler logic
2. **Transparent Precision**: Mixed-precision encoding/decoding is transparent to the attention backend; decode worker sees uniform-precision KV cache
3. **NIXL-Native Transport**: Leverage NIXL's async RDMA transfer as the data plane; SpectrumKV handles precision compression at the application layer
4. **Probe-Driven Adaptation**: Runtime quality monitoring with automatic precision escalation without disrupting ongoing decode
5. **Dynamo-Compatible**: Expose KV cache metadata that Dynamo's KV-aware router can consume for optimal prefill-to-decode scheduling

### 1.3 Key Reference Systems

| System | Integration Pattern | Relevance to SpectrumKV |
|--------|--------------------|------------------------|
| **NixlConnector** | Direct P2P KV transfer via NIXL read/write API | Transport layer basis |
| **LMCacheConnectorV1** | External cache engine with STORE/RETRIEVE/LOOKUP protocol | Multi-tier caching + CacheBlend selective recompute pattern |
| **P2pNcclConnector** | Scheduler-side metadata + Worker-side layer-by-layer save/load | KVConnectorBase_V1 lifecycle reference |
| **OffloadingConnector** | GPU→CPU offload with watermark eviction | Block-level precision mapping reference |
| **FlexKVConnectorV1** | Distributed RadixTree + multi-tier offload | Cross-node KV sharing pattern |
| **NVIDIA Dynamo KVBM** | KV-aware routing + NIXL-backed multi-tier storage | Router integration + NIXL transfer coordination |

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          NVIDIA Dynamo / External Router                        │
│   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐   │
│   │  Smart Router    │  │  KV Event Bus    │  │  Planner (Auto-scaling)     │   │
│   │ (KV-aware Route) │  │  (NATS/etcd)     │  │                             │   │
│   └────────┬─────────┘  └────────┬─────────┘  └─────────────────────────────┘   │
└────────────┼──────────────────────┼──────────────────────────────────────────────┘
             │ KV hit events        │ KV metadata
             ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SpectrumKV Plugin Layer                                   │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                     SpectrumKVConnector (KVConnectorBase_V1)              │  │
│  │                                                                           │  │
│  │  ┌─────────────────────────┐  ┌──────────────────────────────────────┐   │  │
│  │  │  Scheduler-Side         │  │  Worker-Side                         │   │  │
│  │  │  ┌───────────────────┐  │  │  ┌────────────────────────────────┐ │   │  │
│  │  │  │SWS Ranker         │  │  │  │ SpectrumKVAttentionBackend     │ │   │  │
│  │  │  │(importance scoring)│  │  │  │ (precision decode + load/save) │ │   │  │
│  │  │  └───────────────────┘  │  │  └────────────────────────────────┘ │   │  │
│  │  │  ┌───────────────────┐  │  │  ┌────────────────────────────────┐ │   │  │
│  │  │  │QCBM Allocator     │  │  │  │ QCBM Codec Engine             │ │   │  │
│  │  │  │(precision mapping)│  │  │  │ (encode: quant+compress        │ │   │  │
│  │  │  └───────────────────┘  │  │  │  decode: dequant+decompress)   │ │   │  │
│  │  │  ┌───────────────────┐  │  │  └────────────────────────────────┘ │   │  │
│  │  │  │Probe Controller   │  │  │  ┌────────────────────────────────┐ │   │  │
│  │  │  │(quality monitor)  │  │  │  │ NIXL Transport Agent           │ │   │  │
│  │  │  └───────────────────┘  │  │  │ (async send/recv via NIXL)     │ │   │  │
│  │  │  ┌───────────────────┐  │  │  └────────────────────────────────┘ │   │  │
│  │  │  │Metadata Builder   │  │  │  ┌────────────────────────────────┐ │   │  │
│  │  │  │(connector_meta)   │  │  │  │ Probe Evaluator                │ │   │  │
│  │  │  └───────────────────┘  │  │  │ (local quality check + report) │ │   │  │
│  │  └─────────────────────────┘  │  └────────────────────────────────┘ │   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                SpectrumKVBlockManager (extends BlockSpaceManager)          │  │
│  │  ┌──────────────────────┐  ┌──────────────────────────────────────────┐  │  │
│  │  │ Precision Block Pool │  │ Precision-Aware Allocation               │  │  │
│  │  │ (FP16/FP8/INT4/DROP) │  │ (variable-size virtual blocks per token) │  │  │
│  │  └──────────────────────┘  └──────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└────────────────────────────────┬────────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼                         ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│    Prefill Instance      │   │    Decode Instance       │
│  ┌────────────────────┐  │   │  ┌────────────────────┐  │
│  │ vLLM Scheduler     │  │   │  │ vLLM Scheduler     │  │
│  │ + SpectrumKV       │  │   │  │ + SpectrumKV       │  │
│  │   SchedulerMgr     │  │   │  │   SchedulerMgr     │  │
│  └────────────────────┘  │   │  └────────────────────┘  │
│  ┌────────────────────┐  │   │  ┌────────────────────┐  │
│  │ vLLM Worker        │  │   │  │ vLLM Worker        │  │
│  │ + SpectrumKV       │  │   │  │ + SpectrumKV       │  │
│  │   WorkerHandler    │  │   │  │   WorkerHandler    │  │
│  │   (ENCODER role)   │  │   │  │   (DECODER role)   │  │
│  └────────────────────┘  │   │  └────────────────────┘  │
│  ┌────────────────────┐  │   │  ┌────────────────────┐  │
│  │ NIXL Sender        │  │   │  │ NIXL Receiver      │  │
│  │ (RDMA Write)       │──┼───┼─▶│ (RDMA Read)        │  │
│  └────────────────────┘  │   │  └────────────────────┘  │
│         GPU HBM          │   │         GPU HBM          │
└──────────────────────────┘   └──────────────────────────┘
```

### 2.2 Component Dependency Map

```
SpectrumKVConnector
 ├── SpectrumKVSchedulerManager    (scheduler process)
 │    ├── SWSRanker                (attention-weight based importance)
 │    ├── QCBMAllocator            (precision assignment policy)
 │    ├── ProbeController          (degradation detection + escalation)
 │    └── MetadataBuilder          (build KVConnectorMetadata)
 │
 ├── SpectrumKVWorkerHandler       (worker process)
 │    ├── QCBMCodecEngine          (encode/decode per-token precision)
 │    ├── SpectrumKVAttentionHook  (integrate with attention layer)
 │    ├── NIXLTransportAgent       (async send/recv)
 │    └── ProbeEvaluator           (local quality verification)
 │
 └── SpectrumKVBlockManager        (block allocation with precision awareness)
      ├── PrecisionBlockPool       (manage blocks across precision tiers)
      └── PrecisionAwareAllocator  (variable-size allocation)
```

---

## 3. Core Interface Design

### 3.1 SpectrumKVConnector

The top-level connector class, inheriting from `KVConnectorBase_V1`. It delegates to `SpectrumKVSchedulerManager` (scheduler process) or `SpectrumKVWorkerHandler` (worker process) based on the `KVConnectorRole`.

```python
# File: spectrumkv/vllm_connector/connector.py

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorOutput,
    KVConnectorRole,
)
from vllm.config import VllmConfig, KVTransferConfig
from vllm.v1.core.kv_cache_manager import KVCacheBlocks

class SpectrumKVConnector(KVConnectorBase_V1):
    """SpectrumKV: Per-Token Mixed-Precision KV Cache Transfer Connector.
    
    Implements KVConnectorBase_V1 for PD disaggregated serving with
    semantic-aware mixed-precision KV cache compression and transfer.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig",
        role: KVConnectorRole,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, role)

        # Parse SpectrumKV-specific config from kv_connector_extra_config
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.spectrum_config = SpectrumKVConfig.from_extra_config(extra_config)

        # Delegate to role-specific handler
        self.scheduler_manager: SpectrumKVSchedulerManager | None = None
        self.worker_handler: SpectrumKVWorkerHandler | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.scheduler_manager = SpectrumKVSchedulerManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                spectrum_config=self.spectrum_config,
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = SpectrumKVWorkerHandler(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                spectrum_config=self.spectrum_config,
            )

    # ─── Scheduler-Side Methods ───────────────────────────────────────────

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Return (num_matched_tokens, is_async).

        SpectrumKV extends the base semantics: matched tokens account for
        tokens that were previously transferred at sufficient precision
        (checked via ProbeController). Tokens previously dropped or
        low-precision may be re-matched for precision upgrade.
        """
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        """After block allocation, update QCBM precision mapping for
        the allocated block range.
        
        This is where QCBMAllocator assigns per-token precision levels
        based on SWS ranking of the request's tokens.
        """
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_state_after_alloc(
                request, blocks, num_external_tokens
            )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        """Build SpectrumKVMetadata containing:
        - Per-token precision assignments (from QCBMAllocator)
        - SWS importance scores (for worker-side optimization)
        - Probe escalation flags (if re-transmission needed)
        - NIXL transfer descriptors
        """
        if self.scheduler_manager is not None:
            return self.scheduler_manager.build_connector_meta(scheduler_output)
        return SpectrumKVMetadata()

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        """Process worker-side feedback:
        - Probe evaluation results (quality scores)
        - Transfer completion status
        - Precision upgrade requests
        """
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request finishes. Returns:
        - (should_free_now: bool, kv_transfer_params: dict | None)
        
        SpectrumKV may defer block freeing if Probe indicates
        the KV cache should be retained for potential re-transmission.
        """
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished(request, block_ids)
        return True, None

    def take_events(self) -> list["KVCacheEvent"]:
        """Emit KV cache events for Dynamo's KV-aware router.
        
        Events include:
        - KV_CACHE_STORED: new KV blocks available with precision info
        - KV_CACHE_REMOVED: blocks freed after transfer
        - KV_CACHE_UPGRADED: precision upgrade completed
        """
        if self.scheduler_manager is not None:
            return self.scheduler_manager.take_events()
        return []

    # ─── Worker-Side Methods ───────────────────────────────────────────────

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register GPU KV cache tensors with the worker handler.
        
        SpectrumKV stores references for in-place precision encoding/decoding.
        """
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(kv_caches)

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        """Bind scheduler-emitted metadata (precision assignments,
        SWS scores, transfer descriptors) to the worker handler."""
        super().bind_connector_metadata(connector_metadata)
        if self.worker_handler is not None:
            assert isinstance(connector_metadata, SpectrumKVMetadata)
            self.worker_handler.bind_connector_metadata(connector_metadata)

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        """Begin async KV cache loading from remote prefill worker.
        
        SpectrumKV workflow:
        1. Issue NIXL recv for precision-encoded KV data
        2. QCBMCodecEngine prepares decode buffers
        3. Async copy+dequant pipeline starts
        """
        if self.worker_handler is not None:
            self.worker_handler.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until layer KV data is fully loaded and decoded
        to full precision in GPU KV cache.
        
        Decoding pipeline:
        NIXL recv → dequant (INT4/FP8→FP16) → write to kv_cache
        """
        if self.worker_handler is not None:
            self.worker_handler.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Save KV cache for a single layer with precision encoding.
        
        Prefill worker pipeline:
        1. Extract per-token data from kv_layer
        2. QCBMCodecEngine: quantize per SWS-assigned precision
        3. Pack precision-encoded data + metadata
        4. Issue NIXL async send
        """
        if self.worker_handler is not None:
            self.worker_handler.save_kv_layer(
                layer_name, kv_layer, attn_metadata, **kwargs
            )

    def wait_for_save(self) -> None:
        """Block until all async NIXL sends complete."""
        if self.worker_handler is not None:
            self.worker_handler.wait_for_save()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Return (finished_sending_ids, finished_receiving_ids).
        
        Also triggers ProbeEvaluator for completed receives.
        """
        if self.worker_handler is not None:
            return self.worker_handler.get_finished(finished_req_ids)
        return None, None

    def get_kv_connector_stats(self) -> dict[str, Any]:
        """Return SpectrumKV-specific stats for observability:
        - tokens_sent_per_precision: {FP16: N, FP8: M, INT4: K, DROP: D}
        - avg_compression_ratio: float
        - probe_escalation_count: int
        - transfer_bandwidth_utilization: float
        """
        if self.worker_handler is not None:
            return self.worker_handler.get_stats()
        return {}
```

### 3.2 SpectrumKVBlockManager

Extends vLLM's block allocation logic to account for **precision-dependent virtual block sizes**. A token stored at FP16 occupies the standard block slot, while FP8/INT4 tokens can be packed more densely in the transfer buffer.

```python
# File: spectrumkv/vllm_connector/block_manager.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class PrecisionLevel(Enum):
    """Per-token precision levels, ordered by compression ratio."""
    FP16 = 0    # Full precision (baseline, 16-bit per element)
    FP8  = 1    # 8-bit floating point (2x compression)
    INT4 = 2    # 4-bit integer (4x compression, with scale factors)
    DROP = 3    # Token dropped entirely (infinite compression)

    @property
    def bytes_per_element(self) -> int:
        return {0: 2, 1: 1, 2: 1, 3: 0}[self.value]  # INT4 packs 2 per byte

    @property
    def compression_ratio(self) -> float:
        return {0: 1.0, 1: 2.0, 2: 4.0, 3: float('inf')}[self.value]


@dataclass
class TokenPrecisionMap:
    """Maps token positions to their assigned precision levels.
    
    This is the output of QCBMAllocator and is carried through
    the metadata pipeline to the worker-side codec.
    """
    request_id: str
    # token_index → PrecisionLevel, length = num_tokens
    precision_vector: list[PrecisionLevel] = field(default_factory=list)
    # Per-token scale factors for INT4 quantization
    # shape: [num_tokens, num_kv_heads, 2] (k_scale, v_scale)
    scale_factors: Optional[torch.Tensor] = None
    # SWS importance scores that drove the precision assignment
    importance_scores: Optional[list[float]] = None

    def get_transfer_bytes(
        self,
        num_kv_heads: int,
        head_size: int,
        start_token: int = 0,
        end_token: int | None = None,
    ) -> int:
        """Calculate total bytes needed for transfer of the token range,
        accounting for mixed-precision encoding."""
        if end_token is None:
            end_token = len(self.precision_vector)
        total = 0
        for i in range(start_token, end_token):
            prec = self.precision_vector[i]
            if prec == PrecisionLevel.DROP:
                continue
            # KV = 2 (key + value), each with num_kv_heads * head_size elements
            elements_per_token = 2 * num_kv_heads * head_size
            if prec == PrecisionLevel.INT4:
                # INT4: 2 elements per byte + scale factors (2 fp16 per head)
                total += (elements_per_token + 1) // 2  # packed INT4 data
                total += num_kv_heads * 2 * 2  # k_scale + v_scale per head
            else:
                total += elements_per_token * prec.bytes_per_element
        return total


@dataclass
class PrecisionBlock:
    """A block in the precision-aware block pool.
    
    Unlike standard KVCacheBlock which has fixed size (block_size tokens),
    PrecisionBlock tracks the actual transfer size for the block
    based on the precision assignment of its tokens.
    """
    block_id: int
    # Number of tokens in this block (same as vLLM block_size)
    num_tokens: int
    # Precision levels for tokens in this block
    token_precisions: list[PrecisionLevel]
    # Compressed transfer size in bytes
    transfer_size_bytes: int
    # Reference count for lifecycle management
    ref_cnt: int = 0


class SpectrumKVBlockManager:
    """Precision-aware block manager for SpectrumKV.
    
    This class wraps vLLM's standard BlockSpaceManager and adds
    precision-dependent transfer buffer allocation on top.
    
    Standard block allocation (GPU HBM) remains unchanged — decode worker
    always receives full-precision KV cache. The precision optimization
    only applies to the **transfer** path (network I/O reduction).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig",
        spectrum_config: "SpectrumKVConfig",
    ) -> None:
        self.block_size = vllm_config.cache_config.block_size
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = vllm_config.model_config.get_head_size()
        self.spectrum_config = spectrum_config

        # Transfer buffer pool — separate from GPU block pool
        # This is where precision-encoded data is staged before NIXL send
        self._transfer_buffer_pool: dict[int, PrecisionBlock] = {}
        self._next_block_id = 0

    def allocate_precision_blocks(
        self,
        token_precision_map: TokenPrecisionMap,
        gpu_block_ids: list[int],
    ) -> list[PrecisionBlock]:
        """Create PrecisionBlocks from a TokenPrecisionMap.
        
        One PrecisionBlock per vLLM GPU block (same block_size).
        The transfer_size_bytes is computed from the mixed-precision encoding.
        
        Args:
            token_precision_map: Per-token precision assignment
            gpu_block_ids: Allocated GPU block IDs from BlockSpaceManager
            
        Returns:
            List of PrecisionBlocks with computed transfer sizes
        """
        precision_blocks = []
        num_blocks = len(gpu_block_ids)

        for block_idx in range(num_blocks):
            token_start = block_idx * self.block_size
            token_end = min(token_start + self.block_size,
                          len(token_precision_map.precision_vector))
            block_precisions = token_precision_map.precision_vector[token_start:token_end]

            transfer_bytes = token_precision_map.get_transfer_bytes(
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                start_token=token_start,
                end_token=token_end,
            )

            pb = PrecisionBlock(
                block_id=gpu_block_ids[block_idx],
                num_tokens=len(block_precisions),
                token_precisions=block_precisions,
                transfer_size_bytes=transfer_bytes,
            )
            self._transfer_buffer_pool[pb.block_id] = pb
            precision_blocks.append(pb)

        return precision_blocks

    def free_precision_blocks(self, block_ids: list[int]) -> None:
        """Release precision blocks back to the pool."""
        for bid in block_ids:
            self._transfer_buffer_pool.pop(bid, None)

    def get_total_transfer_bytes(
        self, block_ids: list[int]
    ) -> int:
        """Sum transfer sizes for a list of blocks — used for
        NIXL buffer reservation."""
        return sum(
            self._transfer_buffer_pool.get(bid, PrecisionBlock(
                block_id=bid, num_tokens=0, token_precisions=[],
                transfer_size_bytes=0
            )).transfer_size_bytes
            for bid in block_ids
        )

    def get_precision_block(self, block_id: int) -> PrecisionBlock | None:
        return self._transfer_buffer_pool.get(block_id)
```

### 3.3 SpectrumKVAttentionBackend

Hooks into vLLM's attention layer to perform precision encoding (prefill worker) and decoding (decode worker). This is not a full `AttentionBackend` replacement — it operates as a **connector hook** invoked via the `@maybe_transfer_kv_layer` decorator pattern already used by vLLM connectors.

```python
# File: spectrumkv/vllm_connector/attention_hook.py

from typing import Optional

class SpectrumKVAttentionHook:
    """Precision-aware KV cache encoding/decoding hook.
    
    This class is called from within the attention layer execution
    (via vLLM's kv_transfer_state mechanism) to:
    
    Prefill worker:
      - After attention computation, encode KV per QCBM precision
      - Hand off encoded data to NIXLTransportAgent for async send
    
    Decode worker:
      - Before attention computation, decode received KV to full precision
      - Write decoded KV into the standard kv_cache tensor
      - Run ProbeEvaluator on decoded KV for quality verification
    """

    def __init__(
        self,
        spectrum_config: "SpectrumKVConfig",
        qcbm_codec: "QCBMCodecEngine",
        nixl_agent: "NIXLTransportAgent",
        probe_evaluator: "ProbeEvaluator",
    ) -> None:
        self.config = spectrum_config
        self.codec = qcbm_codec
        self.transport = nixl_agent
        self.probe = probe_evaluator

    # ─── Prefill Worker: Encode + Send ─────────────────────────────────────

    def encode_and_send_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        token_precision_map: TokenPrecisionMap,
        request_id: str,
        block_ids: list[int],
    ) -> None:
        """Encode a single layer's KV cache with mixed precision and
        initiate async NIXL transfer.
        
        Pipeline (overlapped with next-layer attention compute):
        1. Extract KV for this layer from kv_layer using block_ids
        2. Per-token quantize: FP16→FP8/INT4 per QCBM assignment
        3. Pack encoded data + metadata into transfer buffer
        4. Issue NIXL async write (non-blocking)
        
        Args:
            layer_name: e.g. "model.layers.0.self_attn"
            kv_layer: GPU tensor [2, num_blocks, block_size, num_kv_heads, head_size]
            token_precision_map: Per-token precision assignment
            request_id: Request identifier for NIXL tensor naming
            block_ids: GPU block IDs containing this request's KV data
        """
        # Step 1: Extract relevant KV data
        kv_data = self._extract_kv_for_blocks(kv_layer, block_ids)

        # Step 2: Encode with mixed precision
        encoded_data, encode_metadata = self.codec.encode_layer(
            kv_data=kv_data,
            precision_map=token_precision_map,
            layer_name=layer_name,
        )

        # Step 3: Issue async NIXL send
        tensor_id = f"{request_id}#{layer_name}"
        self.transport.async_send(
            tensor_id=tensor_id,
            data=encoded_data,
            metadata=encode_metadata,
        )

    # ─── Decode Worker: Receive + Decode ───────────────────────────────────

    def recv_and_decode_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        token_precision_map: TokenPrecisionMap,
        request_id: str,
        block_ids: list[int],
    ) -> None:
        """Receive precision-encoded KV data from prefill worker,
        decode to full precision, and write into GPU kv_cache.
        
        Pipeline (overlapped with previous-layer attention compute):
        1. Wait for NIXL async recv completion for this layer
        2. Per-token dequantize: FP8/INT4→FP16 per QCBM metadata
        3. Write decoded full-precision KV into kv_layer at block_ids
        4. Run ProbeEvaluator on decoded KV (if probe is enabled)
        
        Args:
            layer_name: Layer identifier
            kv_layer: Target GPU KV cache tensor to write into
            token_precision_map: Precision metadata for decoding
            request_id: Request identifier
            block_ids: Target GPU block IDs to write decoded KV
        """
        # Step 1: Wait for NIXL recv
        tensor_id = f"{request_id}#{layer_name}"
        encoded_data, encode_metadata = self.transport.wait_recv(tensor_id)

        # Step 2: Decode to full precision
        decoded_kv = self.codec.decode_layer(
            encoded_data=encoded_data,
            encode_metadata=encode_metadata,
            precision_map=token_precision_map,
            layer_name=layer_name,
        )

        # Step 3: Write into kv_cache
        self._write_kv_to_blocks(kv_layer, decoded_kv, block_ids)

        # Step 4: Probe evaluation (optional, configurable)
        if self.config.enable_probe:
            quality_score = self.probe.evaluate_layer(
                decoded_kv=decoded_kv,
                layer_name=layer_name,
                request_id=request_id,
            )
            if quality_score < self.config.probe_quality_threshold:
                self.probe.request_escalation(
                    request_id=request_id,
                    layer_name=layer_name,
                    current_precision=token_precision_map,
                    quality_score=quality_score,
                )

    # ─── Internal Helpers ──────────────────────────────────────────────────

    def _extract_kv_for_blocks(
        self,
        kv_layer: torch.Tensor,
        block_ids: list[int],
    ) -> torch.Tensor:
        """Extract KV data for specific blocks from the layer tensor.
        
        Args:
            kv_layer: [2, num_blocks, block_size, num_kv_heads, head_size]
            block_ids: Block indices to extract
            
        Returns:
            Extracted KV: [2, len(block_ids), block_size, num_kv_heads, head_size]
        """
        return kv_layer[:, block_ids, ...]

    def _write_kv_to_blocks(
        self,
        kv_layer: torch.Tensor,
        decoded_kv: torch.Tensor,
        block_ids: list[int],
    ) -> None:
        """Write decoded full-precision KV back into kv_cache.
        
        In-place operation on the GPU kv_cache tensor.
        """
        kv_layer[:, block_ids, ...] = decoded_kv
```

### 3.4 QCBMCodecEngine

The core encoding/decoding engine that implements per-token mixed-precision quantization.

```python
# File: spectrumkv/vllm_connector/qcbm_codec.py

from dataclasses import dataclass

@dataclass
class EncodedLayerData:
    """Container for precision-encoded KV data of a single layer."""
    # Packed KV data: [total_encoded_bytes] on GPU
    data_tensor: torch.Tensor
    # Metadata for decoding
    metadata: "EncodeLayerMetadata"

@dataclass
class EncodeLayerMetadata:
    """Metadata needed to decode a precision-encoded layer."""
    layer_name: str
    # Number of tokens at each precision level
    precision_counts: dict[PrecisionLevel, int]
    # Per-token precision vector (same length as block_size * num_blocks)
    precision_vector: list[PrecisionLevel]
    # INT4 scale factors: [num_int4_tokens, num_kv_heads, 2] (k_scale, v_scale)
    int4_scales: torch.Tensor | None
    # FP8 scale factors: [num_fp8_tokens, num_kv_heads, 2]
    fp8_scales: torch.Tensor | None
    # Total tokens encoded (excluding DROP)
    num_encoded_tokens: int
    # Original shape for reconstruction
    original_shape: tuple[int, ...]


class QCBMCodecEngine:
    """Quantization Configuration Block Manager — Codec Engine.
    
    Handles per-token mixed-precision encoding (prefill side) and
    decoding (decode side) of KV cache data.
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        device: str = "cuda",
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.device = device

    def encode_layer(
        self,
        kv_data: torch.Tensor,
        precision_map: TokenPrecisionMap,
        layer_name: str,
    ) -> tuple[EncodedLayerData, EncodeLayerMetadata]:
        """Encode a full layer's KV data with per-token mixed precision.
        
        Args:
            kv_data: [2, num_blocks, block_size, num_kv_heads, head_size] FP16
            precision_map: TokenPrecisionMap with per-token assignments
            layer_name: Layer identifier
            
        Returns:
            Tuple of (encoded data container, metadata for decoding)
            
        Encoding Strategy:
            FP16 tokens: stored as-is (no compression)
            FP8 tokens:  cast to FP8_E4M3 with per-head scale factors
            INT4 tokens: quantized to INT4 with per-head scale factors
            DROP tokens: omitted entirely (will be re-computed by attention)
        """
        # Flatten token dimension for per-token processing
        # kv_data: [2, num_blocks, block_size, num_kv_heads, head_size]
        num_blocks = kv_data.shape[1]
        total_tokens = num_blocks * self.block_size

        precision_counts = {p: 0 for p in PrecisionLevel}
        fp8_scales_list = []
        int4_scales_list = []

        encoded_parts = []  # Collect encoded tensors per precision tier

        # Process by precision tier for vectorized encoding
        for prec in [PrecisionLevel.FP16, PrecisionLevel.FP8,
                     PrecisionLevel.INT4]:
            # Build mask for tokens at this precision
            mask = self._build_precision_mask(
                precision_map.precision_vector, prec, total_tokens
            )
            count = mask.sum().item()
            precision_counts[prec] = int(count)

            if count == 0:
                continue

            # Extract tokens at this precision
            # kv_data reshaped: [2, total_tokens, num_kv_heads, head_size]
            kv_flat = kv_data.reshape(2, total_tokens,
                                      self.num_kv_heads, self.head_size)
            selected_kv = kv_flat[:, mask, ...]

            if prec == PrecisionLevel.FP16:
                encoded_parts.append(selected_kv.flatten())

            elif prec == PrecisionLevel.FP8:
                # Per-head FP8 quantization
                quantized, scales = self._quantize_fp8_per_head(selected_kv)
                encoded_parts.append(quantized.flatten())
                fp8_scales_list.append(scales)

            elif prec == PrecisionLevel.INT4:
                # Per-head INT4 quantization
                quantized, scales = self._quantize_int4_per_head(selected_kv)
                encoded_parts.append(quantized.flatten())
                int4_scales_list.append(scales)

        precision_counts[PrecisionLevel.DROP] = sum(
            1 for p in precision_map.precision_vector
            if p == PrecisionLevel.DROP
        )

        # Concatenate all encoded parts into a single GPU tensor
        if encoded_parts:
            data_tensor = torch.cat(encoded_parts)
        else:
            data_tensor = torch.tensor([], device=self.device)

        metadata = EncodeLayerMetadata(
            layer_name=layer_name,
            precision_counts=precision_counts,
            precision_vector=precision_map.precision_vector[:total_tokens],
            int4_scales=torch.cat(int4_scales_list) if int4_scales_list
                         else None,
            fp8_scales=torch.cat(fp8_scales_list) if fp8_scales_list
                        else None,
            num_encoded_tokens=total_tokens - precision_counts[PrecisionLevel.DROP],
            original_shape=kv_data.shape,
        )

        return EncodedLayerData(data_tensor=data_tensor, metadata=metadata), metadata

    def decode_layer(
        self,
        encoded_data: torch.Tensor,
        encode_metadata: EncodeLayerMetadata,
        precision_map: TokenPrecisionMap,
        layer_name: str,
    ) -> torch.Tensor:
        """Decode precision-encoded KV data back to full FP16.
        
        Reconstructs the original [2, num_blocks, block_size, num_kv_heads,
        head_size] tensor from the mixed-precision encoded data.
        
        DROP tokens are zero-filled — the attention layer will re-compute
        them during the next forward pass (similar to CacheBlend's
        selective recompute pattern).
        """
        original_shape = encode_metadata.original_shape
        total_tokens = original_shape[1] * original_shape[2]  # num_blocks * block_size

        # Pre-allocate output tensor (zeros for DROP tokens)
        output = torch.zeros(
            2, total_tokens, self.num_kv_heads, self.head_size,
            dtype=torch.float16, device=self.device,
        )

        offset = 0
        fp8_scale_offset = 0
        int4_scale_offset = 0

        # Decode by precision tier (must match encode order)
        for prec in [PrecisionLevel.FP16, PrecisionLevel.FP8,
                     PrecisionLevel.INT4]:
            count = encode_metadata.precision_counts[prec]
            if count == 0:
                continue

            mask = self._build_precision_mask(
                encode_metadata.precision_vector, prec, total_tokens
            )

            if prec == PrecisionLevel.FP16:
                num_elements = count * 2 * self.num_kv_heads * self.head_size
                decoded = encoded_data[offset:offset + num_elements].reshape(
                    2, count, self.num_kv_heads, self.head_size
                )
                output[:, mask, ...] = decoded
                offset += num_elements

            elif prec == PrecisionLevel.FP8:
                num_elements = count * 2 * self.num_kv_heads * self.head_size
                scales = encode_metadata.fp8_scales[fp8_scale_offset:
                    fp8_scale_offset + count]
                decoded = self._dequantize_fp8_per_head(
                    encoded_data[offset:offset + num_elements].reshape(
                        2, count, self.num_kv_heads, self.head_size
                    ),
                    scales,
                )
                output[:, mask, ...] = decoded
                offset += num_elements
                fp8_scale_offset += count

            elif prec == PrecisionLevel.INT4:
                # INT4 is packed: 2 elements per byte
                num_packed = (count * 2 * self.num_kv_heads * self.head_size + 1) // 2
                scales = encode_metadata.int4_scales[int4_scale_offset:
                    int4_scale_offset + count]
                decoded = self._dequantize_int4_per_head(
                    encoded_data[offset:offset + num_packed],
                    scales,
                    count,
                )
                output[:, mask, ...] = decoded
                offset += num_packed
                int4_scale_offset += count

        # Reshape to original layout
        return output.reshape(original_shape)

    # ─── Quantization Primitives ───────────────────────────────────────────

    def _quantize_fp8_per_head(
        self,
        kv: torch.Tensor,  # [2, N, num_kv_heads, head_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head FP8 E4M3 quantization.
        
        Returns:
            quantized: [2, N, num_kv_heads, head_size] in FP8
            scales: [N, num_kv_heads, 2] (k_scale, v_scale)
        """
        # Compute per-head max for scale
        k_max = kv[0].abs().amax(dim=-1)  # [N, num_kv_heads]
        v_max = kv[1].abs().amax(dim=-1)  # [N, num_kv_heads]

        k_scale = k_max / 448.0  # FP8 E4M3 max representable
        v_scale = v_max / 448.0

        scales = torch.stack([k_scale, v_scale], dim=-1)  # [N, num_kv_heads, 2]

        kv_k = (kv[0] / k_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        kv_v = (kv[1] / v_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

        return torch.stack([kv_k, kv_v]), scales

    def _quantize_int4_per_head(
        self,
        kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head INT4 quantization with scale factors.
        
        Returns:
            quantized: packed INT4 tensor
            scales: [N, num_kv_heads, 2] (k_scale, v_scale)
        """
        k_max = kv[0].abs().amax(dim=-1)
        v_max = kv[1].abs().amax(dim=-1)

        k_scale = k_max / 7.0  # INT4 max = 7 (signed: -8 to 7)
        v_scale = v_max / 7.0

        scales = torch.stack([k_scale, v_scale], dim=-1)

        # Quantize and pack (2 values per byte)
        kv_k_int = ((kv[0] / k_scale.unsqueeze(-1)).round()).clamp(-8, 7).to(torch.int8)
        kv_v_int = ((kv[1] / v_scale.unsqueeze(-1)).round()).clamp(-8, 7).to(torch.int8)

        # Pack: interleave k and v, pack pairs of int4 into int8
        # (Implementation uses CUDA kernel for throughput)
        packed = self._pack_int4(kv_k_int, kv_v_int)

        return packed, scales

    def _dequantize_fp8_per_head(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize FP8 back to FP16 using per-head scales."""
        k_scale = scales[:, :, 0].unsqueeze(-1)  # [N, num_kv_heads, 1]
        v_scale = scales[:, :, 1].unsqueeze(-1)

        kv_k = quantized[0].to(torch.float16) * k_scale
        kv_v = quantized[1].to(torch.float16) * v_scale

        return torch.stack([kv_k, kv_v])

    def _dequantize_int4_per_head(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Dequantize INT4 back to FP16 using per-head scales."""
        kv_k_int, kv_v_int = self._unpack_int4(packed, num_tokens)

        k_scale = scales[:, :, 0].unsqueeze(-1)
        v_scale = scales[:, :, 1].unsqueeze(-1)

        kv_k = kv_k_int.to(torch.float16) * k_scale
        kv_v = kv_v_int.to(torch.float16) * v_scale

        return torch.stack([kv_k, kv_v])

    # ─── Utility Methods ───────────────────────────────────────────────────

    def _build_precision_mask(
        self,
        precision_vector: list[PrecisionLevel],
        target_prec: PrecisionLevel,
        total_tokens: int,
    ) -> torch.Tensor:
        """Build boolean mask for tokens at a given precision level."""
        mask = torch.tensor(
            [p == target_prec for p in precision_vector[:total_tokens]],
            device=self.device,
        )
        return mask

    def _pack_int4(self, k_int: torch.Tensor, v_int: torch.Tensor) -> torch.Tensor:
        """Pack INT4 values into bytes (2 per byte). CUDA kernel."""
        # Fallback: CPU implementation for correctness
        # Production: custom CUDA kernel for throughput
        raise NotImplementedError("INT4 packing requires CUDA kernel")

    def _unpack_int4(self, packed: torch.Tensor, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack INT4 bytes back to individual values. CUDA kernel."""
        raise NotImplementedError("INT4 unpacking requires CUDA kernel")
```

### 3.5 SWS Ranker (Scheduler-Side)

```python
# File: spectrumkv/vllm_connector/sws_ranker.py

from dataclasses import dataclass

@dataclass
class SWSConfig:
    """Configuration for Semantic Importance Weighted Sorting."""
    # Scoring method: "attention_norm" | "gradient_proxy" | "entropy"
    scoring_method: str = "attention_norm"
    # Number of recent layers to aggregate for importance score
    aggregation_layers: int = 4
    # Top-k ratio of tokens to keep at FP16 (0.0-1.0)
    fp16_ratio: float = 0.15
    # Ratio of tokens to keep at FP8
    fp8_ratio: float = 0.35
    # Ratio of tokens to compress to INT4
    int4_ratio: float = 0.35
    # Remaining tokens are DROP
    # drop_ratio = 1.0 - fp16_ratio - fp8_ratio - int4_ratio
    # Minimum importance score threshold for DROP
    min_importance_for_keep: float = 0.01


@dataclass
class SWSResult:
    """Output of SWS ranking for a request."""
    request_id: str
    # Per-token importance score [0.0, 1.0]
    importance_scores: list[float]
    # Resulting precision assignment
    precision_map: TokenPrecisionMap


class SWSRanker:
    """Semantic Importance Weighted Sorting.
    
    Ranks tokens by their semantic importance for attention computation
    and assigns precision levels accordingly.
    
    Importance scoring strategies:
    - "attention_norm": Use attention weight L2 norm from recent layers
    - "gradient_proxy": Use gradient-like proxy (output magnitude)
    - "entropy": Use attention entropy as inverse importance signal
    """

    def __init__(self, config: SWSConfig) -> None:
        self.config = config

    def rank_and_assign(
        self,
        request: "Request",
        num_tokens: int,
        # Optional: cached importance from previous prefill passes
        cached_scores: list[float] | None = None,
    ) -> SWSResult:
        """Compute importance scores and assign precision levels.
        
        This runs in the scheduler process, BEFORE block allocation.
        The resulting TokenPrecisionMap is stored and used by
        QCBMAllocator during update_state_after_alloc.
        
        Args:
            request: The incoming request
            num_tokens: Total tokens in the request
            cached_scores: Previously computed importance scores
            
        Returns:
            SWSResult with importance scores and precision assignment
        """
        # Step 1: Get or compute importance scores
        if cached_scores is not None:
            scores = cached_scores
        else:
            # For first pass, use heuristic scoring:
            # - System prompt tokens: high importance (FP16)
            # - Recent tokens: high importance (FP16)
            # - Middle document tokens: medium importance (FP8/INT4)
            # - Low-entropy filler tokens: low importance (DROP)
            scores = self._heuristic_scoring(request, num_tokens)

        # Step 2: Sort and assign precision levels by ratio
        precision_vector = self._assign_by_ratio(scores, num_tokens)

        # Step 3: Build TokenPrecisionMap
        precision_map = TokenPrecisionMap(
            request_id=request.request_id,
            precision_vector=precision_vector,
            importance_scores=scores,
        )

        return SWSResult(
            request_id=request.request_id,
            importance_scores=scores,
            precision_map=precision_map,
        )

    def _heuristic_scoring(
        self,
        request: "Request",
        num_tokens: int,
    ) -> list[float]:
        """Heuristic importance scoring for first-pass prefill.
        
        Strategy:
        - First N tokens (system prompt): score = 1.0
        - Last M tokens (user query): score = 1.0
        - Middle tokens: score decreases with distance from boundaries
        """
        scores = [0.5] * num_tokens  # default: medium importance

        # System prompt tokens (first ~10%): high importance
        system_boundary = min(int(num_tokens * 0.1), 256)
        for i in range(system_boundary):
            scores[i] = 1.0

        # Recent tokens (last ~5%): high importance
        recent_boundary = max(num_tokens - int(num_tokens * 0.05), system_boundary)
        for i in range(recent_boundary, num_tokens):
            scores[i] = 1.0

        # Gradual decay in middle
        for i in range(system_boundary, recent_boundary):
            progress = (i - system_boundary) / max(recent_boundary - system_boundary, 1)
            # V-shape: higher near boundaries, lower in middle
            scores[i] = 0.3 + 0.4 * (1.0 - abs(2.0 * progress - 1.0))

        return scores

    def _assign_by_ratio(
        self,
        scores: list[float],
        num_tokens: int,
    ) -> list[PrecisionLevel]:
        """Assign precision levels based on importance score ranking
        and configured ratio thresholds.
        """
        # Sort token indices by score (descending)
        indexed_scores = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        # Compute cutoff counts
        n_fp16 = int(num_tokens * self.config.fp16_ratio)
        n_fp8 = int(num_tokens * self.config.fp8_ratio)
        n_int4 = int(num_tokens * self.config.int4_ratio)

        # Assign
        precision_vector = [PrecisionLevel.DROP] * num_tokens
        for rank, (idx, score) in enumerate(indexed_scores):
            if score < self.config.min_importance_for_keep:
                precision_vector[idx] = PrecisionLevel.DROP
            elif rank < n_fp16:
                precision_vector[idx] = PrecisionLevel.FP16
            elif rank < n_fp16 + n_fp8:
                precision_vector[idx] = PrecisionLevel.FP8
            elif rank < n_fp16 + n_fp8 + n_int4:
                precision_vector[idx] = PrecisionLevel.INT4
            else:
                precision_vector[idx] = PrecisionLevel.DROP

        return precision_vector

    def update_scores_from_probe(
        self,
        request_id: str,
        current_scores: list[float],
        probe_feedback: "ProbeFeedback",
    ) -> list[float]:
        """Update importance scores based on Probe evaluator feedback.
        
        When the decode-side Probe detects quality degradation for
        specific tokens, this method boosts their importance scores
        to trigger precision escalation on the next transfer.
        """
        updated = list(current_scores)
        for token_idx, quality_delta in probe_feedback.token_quality_deltas:
            if token_idx < len(updated):
                # Boost importance for degraded tokens
                updated[token_idx] = min(1.0, updated[token_idx] + quality_delta)
        return updated
```

### 3.6 Probe Controller & Evaluator

```python
# File: spectrumkv/vllm_connector/probe.py

from dataclasses import dataclass, field

@dataclass
class ProbeConfig:
    """Configuration for adaptive precision probing."""
    # Enable/disable probe
    enable_probe: bool = True
    # Quality threshold below which escalation is triggered
    quality_threshold: float = 0.85
    # Probe evaluation layers (which layers to check)
    # Typically: first layer, last layer, and a few middle layers
    probe_layers: list[int] = field(default_factory=lambda: [0, 15, 31])
    # Maximum escalation retries per request
    max_escalation_retries: int = 2
    # Probe method: "cosine_similarity" | "relative_error" | "attention_shift"
    probe_method: str = "cosine_similarity"
    # Sampling rate for probe (check every N-th token)
    probe_sample_rate: int = 1  # check all tokens by default

@dataclass
class ProbeFeedback:
    """Feedback from decode-side probe evaluation."""
    request_id: str
    # Average quality score across probed layers
    overall_quality: float
    # Per-token quality deltas (token_idx, delta_to_boost_importance)
    token_quality_deltas: list[tuple[int, float]]
    # Layers that triggered escalation
    escalated_layers: list[str]
    # Current escalation retry count
    retry_count: int

@dataclass
class ProbeResult:
    """Result of a single probe evaluation."""
    layer_name: str
    request_id: str
    # Per-token cosine similarity: [num_tokens]
    token_similarities: torch.Tensor
    # Average quality for this layer
    avg_quality: float
    # Whether escalation is recommended
    needs_escalation: bool


class ProbeEvaluator:
    """Decode-side quality evaluator.
    
    Runs after KV cache decoding to verify that the mixed-precision
    transfer did not introduce unacceptable quality degradation.
    
    Evaluation methods:
    - "cosine_similarity": Compare decoded KV vectors against
      a small set of reference tokens kept at FP16
    - "relative_error": Measure relative error of decoded vs. reference
    - "attention_shift": Compute attention output shift from expected
    """

    def __init__(
        self,
        config: ProbeConfig,
        num_kv_heads: int,
        head_size: int,
    ) -> None:
        self.config = config
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        # Store reference tokens (FP16 ground truth) for comparison
        self._reference_cache: dict[str, dict[str, torch.Tensor]] = {}

    def evaluate_layer(
        self,
        decoded_kv: torch.Tensor,
        layer_name: str,
        request_id: str,
    ) -> float:
        """Evaluate the quality of decoded KV for a single layer.
        
        Returns:
            quality_score: float in [0.0, 1.0], where 1.0 = perfect
        """
        # Check if this layer should be probed
        layer_idx = self._parse_layer_index(layer_name)
        if layer_idx not in self.config.probe_layers:
            return 1.0  # Skip probing for this layer

        # Get reference KV (if available — stored from FP16 tokens)
        reference = self._reference_cache.get(request_id, {}).get(layer_name)
        if reference is None:
            # No reference available; use statistical probe
            return self._statistical_probe(decoded_kv, layer_name)

        if self.config.probe_method == "cosine_similarity":
            return self._cosine_similarity_probe(decoded_kv, reference)
        elif self.config.probe_method == "relative_error":
            return self._relative_error_probe(decoded_kv, reference)
        else:
            return 1.0

    def request_escalation(
        self,
        request_id: str,
        layer_name: str,
        current_precision: TokenPrecisionMap,
        quality_score: float,
    ) -> None:
        """Request precision escalation for degraded tokens.
        
        This sends a feedback signal back to the scheduler-side
        ProbeController via the connector output pipeline.
        """
        # Build escalation request — this will be carried back
        # via KVConnectorOutput and processed by ProbeController
        pass

    def _cosine_similarity_probe(
        self,
        decoded: torch.Tensor,
        reference: torch.Tensor,
    ) -> float:
        """Compute average cosine similarity between decoded and reference KV."""
        # decoded, reference: [2, num_blocks, block_size, num_kv_heads, head_size]
        decoded_flat = decoded.flatten()
        reference_flat = reference.flatten()

        cos_sim = torch.nn.functional.cosine_similarity(
            decoded_flat.unsqueeze(0),
            reference_flat.unsqueeze(0),
            dim=1,
        )
        return cos_sim.item()

    def _relative_error_probe(
        self,
        decoded: torch.Tensor,
        reference: torch.Tensor,
    ) -> float:
        """Compute 1 - relative_error as quality score."""
        rel_err = (decoded - reference).norm() / (reference.norm() + 1e-8)
        return max(0.0, 1.0 - rel_err.item())

    def _statistical_probe(
        self,
        decoded: torch.Tensor,
        layer_name: str,
    ) -> float:
        """Statistical probe when no reference is available.
        Checks for anomalies in the decoded KV distribution."""
        # Heuristic: check if KV values have reasonable distribution
        std = decoded.std().item()
        mean = decoded.mean().item()
        # Very low std or extreme mean suggests corrupted decode
        if std < 1e-6 or abs(mean) > 100:
            return 0.0
        return 0.9  # Conservative default

    def _parse_layer_index(self, layer_name: str) -> int:
        """Extract layer index from layer name string."""
        # e.g., "model.layers.15.self_attn" → 15
        import re
        match = re.search(r"layers\.(\d+)", layer_name)
        return int(match.group(1)) if match else -1


class ProbeController:
    """Scheduler-side controller that manages probe escalation logic.
    
    Receives ProbeFeedback from decode-side evaluator and triggers
    precision escalation (re-transfer at higher precision).
    """

    def __init__(self, config: ProbeConfig, sws_ranker: SWSRanker) -> None:
        self.config = config
        self.sws_ranker = sws_ranker
        # Track escalation state per request
        self._escalation_state: dict[str, int] = {}  # request_id → retry_count

    def process_feedback(self, feedback: ProbeFeedback) -> bool:
        """Process probe feedback from decode worker.
        
        Returns:
            True if precision escalation should be triggered
        """
        if feedback.retry_count >= self.config.max_escalation_retries:
            return False  # Max retries reached

        if feedback.overall_quality < self.config.quality_threshold:
            # Update importance scores to boost degraded tokens
            self.sws_ranker.update_scores_from_probe(
                request_id=feedback.request_id,
                current_scores=[],  # Will be fetched from stored state
                probe_feedback=feedback,
            )
            self._escalation_state[feedback.request_id] = feedback.retry_count + 1
            return True

        return False

    def get_escalated_precision_map(
        self,
        request_id: str,
        original_map: TokenPrecisionMap,
    ) -> TokenPrecisionMap:
        """Generate an escalated precision map for re-transfer.
        
        Strategy: Upgrade all tokens that were below FP16 by one tier:
        - DROP → INT4
        - INT4 → FP8
        - FP8 → FP16
        - FP16 → FP16 (no change)
        """
        retry_count = self._escalation_state.get(request_id, 0)
        upgrade_levels = min(retry_count, 3)

        new_vector = []
        for prec in original_map.precision_vector:
            new_level = max(PrecisionLevel.FP16,
                          PrecisionLevel(prec.value - upgrade_levels))
            new_vector.append(new_level)

        return TokenPrecisionMap(
            request_id=request_id,
            precision_vector=new_vector,
            importance_scores=original_map.importance_scores,
        )
```

### 3.7 Core Data Structures

```python
# File: spectrumkv/vllm_connector/metadata.py

from dataclasses import dataclass, field
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class SpectrumKVMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler connector to worker connector.
    
    Carries per-request precision assignments, SWS scores,
    and NIXL transfer descriptors.
    """
    # Per-request metadata
    requests: dict[str, "SpectrumKVRequestMeta"] = field(default_factory=dict)

    # Probe escalation requests (from decode worker → scheduler)
    escalation_requests: list[str] = field(default_factory=list)

    # KV cache events for Dynamo router
    kv_cache_events: list[dict] = field(default_factory=list)


@dataclass
class SpectrumKVRequestMeta:
    """Per-request metadata within SpectrumKVMetadata."""
    request_id: str
    # Number of tokens to transfer
    num_tokens: int
    # GPU block IDs for this request
    block_ids: list[int]
    # Token precision assignment
    token_precision_map: TokenPrecisionMap
    # NIXL transfer descriptor (opaque, created by NIXL agent)
    nixl_transfer_desc: dict | None = None
    # Whether this is an escalation re-transfer
    is_escalation: bool = False
    # Escalation retry count
    escalation_retry: int = 0


@dataclass
class SpectrumKVConfig:
    """SpectrumKV configuration, parsed from kv_connector_extra_config."""
    # ─── SWS Configuration ─────────────────────────────────────────────
    sws_scoring_method: str = "attention_norm"
    sws_fp16_ratio: float = 0.15
    sws_fp8_ratio: float = 0.35
    sws_int4_ratio: float = 0.35
    # drop_ratio = 1.0 - fp16_ratio - fp8_ratio - int4_ratio

    # ─── Probe Configuration ───────────────────────────────────────────
    enable_probe: bool = True
    probe_quality_threshold: float = 0.85
    probe_method: str = "cosine_similarity"
    probe_layers: list[int] = field(default_factory=lambda: [0, 15, 31])
    max_escalation_retries: int = 2

    # ─── NIXL Transport Configuration ──────────────────────────────────
    nixl_backends: list[str] = field(default_factory=lambda: ["UCX"])
    nixl_buffer_device: str = "cuda"
    nixl_buffer_size: int = 1073741824  # 1GB default
    nixl_side_channel_port: int = 55555

    # ─── Codec Configuration ───────────────────────────────────────────
    enable_fp8: bool = True
    enable_int4: bool = True
    # INT4 quantization scale mode: "per_head" | "per_token"
    int4_scale_mode: str = "per_head"

    @classmethod
    def from_extra_config(cls, extra_config: dict) -> "SpectrumKVConfig":
        """Parse from vLLM's kv_connector_extra_config dict."""
        config = cls()
        for key, value in extra_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
```

---

## 4. Data Flow Design

### 4.1 Prefill Path (Encoder → NIXL Send)

```
                    Prefill Instance
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. REQUEST ARRIVAL                                              │
│     ┌─────────────┐                                              │
│     │  Scheduler   │ get_num_new_matched_tokens()                │
│     │  receives    │──┐                                          │
│     │  request     │  │ 2. SWS RANKING                          │
│     └─────────────┘  │    ┌────────────────────────────────┐    │
│                       └──▶│ SWSRanker.rank_and_assign()     │    │
│                            │                                  │    │
│                            │ Input: request tokens            │    │
│                            │ Output: TokenPrecisionMap        │    │
│                            │   FP16: 15% | FP8: 35%          │    │
│                            │   INT4: 35% | DROP: 15%         │    │
│                            └──────────────┬─────────────────────┘
│                                           │                     │
│  3. BLOCK ALLOCATION                      ▼                     │
│     ┌──────────────────────────────────────────────────┐        │
│     │ update_state_after_alloc()                        │        │
│     │   → QCBMAllocator maps precision to block_ids    │        │
│     │   → SpectrumKVBlockManager.allocate_precision_blocks()    │
│     │   → Computes transfer_size_bytes per block       │        │
│     └──────────────────────────┬───────────────────────┘        │
│                                │                                 │
│  4. METADATA BUILD             ▼                                 │
│     ┌──────────────────────────────────────────────────┐        │
│     │ build_connector_meta()                            │        │
│     │   → SpectrumKVMetadata {                          │        │
│     │       requests: {req_id: SpectrumKVRequestMeta},  │        │
│     │       escalation_requests: [],                    │        │
│     │     }                                             │        │
│     └──────────────────────────┬───────────────────────┘        │
│                                │                                 │
│  5. WORKER EXECUTION           ▼                                 │
│     ┌──────────────────────────────────────────────────┐        │
│     │ GPUModelRunner.execute_model()                    │        │
│     │   ┌─────────────────────────────────────────┐     │        │
│     │   │  For each attention layer (L=0..N-1):   │     │        │
│     │   │                                          │     │        │
│     │   │  a. Compute attention (standard vLLM)    │     │        │
│     │   │     → KV written to kv_cache tensor      │     │        │
│     │   │                                          │     │        │
│     │   │  b. save_kv_layer(L) triggered           │     │        │
│     │   │     → SpectrumKVAttentionHook             │     │        │
│     │   │       .encode_and_send_layer()            │     │        │
│     │   │                                          │     │        │
│     │   │  ┌─────────────────────────────────┐     │     │        │
│     │   │  │ QCBMCodecEngine.encode_layer()  │     │     │        │
│     │   │  │  1. Extract KV from kv_cache    │     │     │        │
│     │   │  │  2. Per-token quantize:          │     │     │        │
│     │   │  │     FP16→copy  FP8→E4M3+scale   │     │     │        │
│     │   │  │     INT4→pack+scale  DROP→skip   │     │     │        │
│     │   │  │  3. Concatenate encoded parts    │     │     │        │
│     │   │  │  4. Return EncodedLayerData      │     │     │        │
│     │   │  └────────────┬────────────────────┘     │     │        │
│     │   │               │                           │     │        │
│     │   │  ┌────────────▼────────────────────┐     │     │        │
│     │   │  │ NIXLTransportAgent.async_send() │     │     │        │
│     │   │  │  tensor_id = req_id#layer_name  │     │     │        │
│     │   │  │  NIXL write (RDMA, non-blocking)│     │     │        │
│     │   │  └─────────────────────────────────┘     │     │        │
│     │   └─────────────────────────────────────────┘     │        │
│     │                                                   │        │
│     │  6. wait_for_save() → ensure all NIXL sends done  │        │
│     └──────────────────────────────────────────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Decode Path (NIXL Recv → Decode → Attention)

```
                    Decode Instance
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. SCHEDULER RECEIVES PREFILL RESULT                            │
│     ┌─────────────┐                                              │
│     │  Scheduler   │ get_num_new_matched_tokens()                │
│     │  schedules   │──┐ returns (num_tokens, is_async=True)      │
│     │  decode      │  │                                          │
│     └─────────────┘  │                                           │
│                       │                                           │
│  2. METADATA BIND    ▼                                           │
│     ┌──────────────────────────────────────────────────┐        │
│     │ build_connector_meta()                            │        │
│     │   → SpectrumKVMetadata carries:                   │        │
│     │     - Precision map from prefill SWS              │        │
│     │     - Block IDs for decode allocation             │        │
│     │     - NIXL transfer descriptors                   │        │
│     └──────────────────────────┬───────────────────────┘        │
│                                │                                 │
│  3. WORKER RECEIVES METADATA   ▼                                 │
│     ┌──────────────────────────────────────────────────┐        │
│     │ bind_connector_metadata()                         │        │
│     │   → Worker handler receives SpectrumKVMetadata    │        │
│     │   → NIXLTransportAgent prepares recv buffers      │        │
│     └──────────────────────────┬───────────────────────┘        │
│                                │                                 │
│  4. ASYNC KV LOAD              ▼                                 │
│     ┌──────────────────────────────────────────────────┐        │
│     │ start_load_kv()                                   │        │
│     │   → For each layer, issue NIXL async recv         │        │
│     │   → NIXLTransportAgent.async_recv(tensor_id)      │        │
│     └──────────────────────────┬───────────────────────┘        │
│                                │                                 │
│  5. LAYER-BY-LAYER DECODE     ▼                                 │
│     ┌──────────────────────────────────────────────────┐        │
│     │ GPUModelRunner.execute_model()                    │        │
│     │   ┌─────────────────────────────────────────┐     │        │
│     │   │  For each attention layer (L=0..N-1):   │     │        │
│     │   │                                          │     │        │
│     │   │  a. wait_for_layer_load(L)               │     │        │
│     │   │     → Wait for NIXL recv completion      │     │        │
│     │   │                                          │     │        │
│     │   │  ┌─────────────────────────────────┐     │     │        │
│     │   │  │ QCBMCodecEngine.decode_layer()  │     │     │        │
│     │   │  │  1. Dequantize FP8/INT4 → FP16  │     │     │        │
│     │   │  │  2. Zero-fill DROP tokens        │     │     │        │
│     │   │  │  3. Write full-precision KV      │     │     │        │
│     │   │  │     into kv_cache at block_ids   │     │     │        │
│     │   │  └────────────┬────────────────────┘     │     │        │
│     │   │               │                           │     │        │
│     │   │  ┌────────────▼────────────────────┐     │     │        │
│     │   │  │ ProbeEvaluator.evaluate_layer() │     │     │        │
│     │   │  │  (if probe_layers contains L)   │     │     │        │
│     │   │  │  quality_score = cos_sim(       │     │     │        │
│     │   │  │    decoded_kv, reference_kv)    │     │     │        │
│     │   │  │  if quality < threshold:        │     │     │        │
│     │   │  │    → request_escalation()       │     │     │        │
│     │   │  └─────────────────────────────────┘     │     │        │
│     │   │                                          │     │        │
│     │   │  b. Compute attention with full KV       │     │        │
│     │   │     → DROP tokens re-computed naturally  │     │        │
│     │   │       (zero KV → attention output=0,     │     │        │
│     │   │        equivalent to CacheBlend pattern)  │     │        │
│     │   └─────────────────────────────────────────┘     │        │
│     │                                                   │        │
│     │  6. get_finished() → report completed requests    │        │
│     │     + probe quality feedback                      │        │
│     └──────────────────────────────────────────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.3 Probe Path (Quality Degradation → Precision Escalation)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PROBE ESCALATION FLOW                         │
│                                                                      │
│  Decode Worker                    Scheduler (Prefill Instance)       │
│  ┌─────────────────┐              ┌──────────────────────┐          │
│  │                  │              │                       │          │
│  │ 1. ProbeEvaluator│              │                       │          │
│  │    detects       │              │                       │          │
│  │    quality <     │              │                       │          │
│  │    threshold     │              │                       │          │
│  │        │         │              │                       │          │
│  │        ▼         │              │                       │          │
│  │ 2. Build         │              │                       │          │
│  │    ProbeFeedback │              │                       │          │
│  │    {req_id,      │              │                       │          │
│  │     quality,     │              │                       │          │
│  │     token_deltas}│              │                       │          │
│  │        │         │              │                       │          │
│  │        ▼         │              │                       │          │
│  │ 3. get_finished()│──KVConnector──▶ update_connector_   │          │
│  │    carries       │  Output      │  output()             │          │
│  │    feedback      │  (async)     │      │                │          │
│  │                  │              │      ▼                │          │
│  │                  │              │ 4. ProbeController    │          │
│  │                  │              │    .process_feedback()│          │
│  │                  │              │      │                │          │
│  │                  │              │      ▼                │          │
│  │                  │              │ 5. SWSRanker          │          │
│  │                  │              │    .update_scores_    │          │
│  │                  │              │     from_probe()      │          │
│  │                  │              │      │                │          │
│  │                  │              │      ▼                │          │
│  │                  │              │ 6. Escalated          │          │
│  │                  │              │    TokenPrecisionMap  │          │
│  │                  │              │    (all tiers ↑1)     │          │
│  │                  │              │    FP16: 15% → 50%    │          │
│  │                  │              │    FP8:  35% → 35%    │          │
│  │                  │              │    INT4: 35% → 15%    │          │
│  │                  │              │    DROP: 15% → 0%     │          │
│  │                  │              │      │                │          │
│  │                  │              │      ▼                │          │
│  │ 8. recv_and_     │◀──NIXL──────│ 7. encode_and_send_   │          │
│  │    decode_layer()│  re-transfer │    layer()             │          │
│  │    (higher prec) │  (escalated) │    (escalated prec)   │          │
│  │        │         │              │                       │          │
│  │        ▼         │              │                       │          │
│  │ 9. ProbeEvaluator│              │                       │          │
│  │    re-evaluates  │              │                       │          │
│  │    quality ≥     │              │                       │          │
│  │    threshold ✓   │              │                       │          │
│  │                  │              │                       │          │
│  └─────────────────┘              └──────────────────────┘          │
│                                                                      │
│  NOTE: If quality still < threshold after max_escalation_retries,    │
│  the system falls back to full FP16 transfer for that request.       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. NIXL Transfer Layer Integration

### 5.1 Integration Architecture

SpectrumKV uses NIXL as the transport layer, wrapping it in a `NIXLTransportAgent` that handles the async send/recv lifecycle. The key design decision is that **NIXL transfers precision-encoded (compressed) data**, not raw KV cache tensors. This means:

- **Prefill worker** (sender): Encodes KV with mixed precision → packs into a contiguous GPU buffer → issues NIXL async write
- **Decode worker** (receiver): Issues NIXL async read → receives packed encoded data → decodes to full precision

```
┌────────────────────────────────────────────────────────────────────┐
│                    NIXL Integration Layer                          │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                NIXLTransportAgent                            │  │
│  │                                                              │  │
│  │  ┌──────────────────────┐    ┌────────────────────────────┐  │  │
│  │  │  Sender Side          │    │  Receiver Side             │  │  │
│  │  │  (Prefill Worker)     │    │  (Decode Worker)           │  │  │
│  │  │                       │    │                             │  │  │
│  │  │  1. Register encoded  │    │  1. Register decode        │  │  │
│  │  │     buffer as NIXL    │    │     buffer with NIXL       │  │  │
│  │  │     export region     │    │     import region           │  │  │
│  │  │                       │    │                             │  │  │
│  │  │  2. async_send()      │    │  2. async_recv()           │  │  │
│  │  │     → nixl_agent.     │    │     → nixl_agent.          │  │  │
│  │  │       write_async(    │    │       read_async(           │  │  │
│  │  │         src_buf,      │    │         dst_buf,            │  │  │
│  │  │         remote_desc)  │    │         remote_desc)        │  │  │
│  │  │                       │    │                             │  │  │
│  │  │  3. wait_send()      │    │  3. wait_recv(tensor_id)   │  │  │
│  │  │     → nixl_agent.    │    │     → nixl_agent.          │  │  │
│  │  │       wait()          │    │       wait()                │  │  │
│  │  └──────────────────────┘    └────────────────────────────┘  │  │
│  │                                                              │  │
│  │  ┌──────────────────────────────────────────────────────────┐│  │
│  │  │  Side Channel (ZMQ/TCP for metadata exchange)            ││  │
│  │  │  - Transfer NIXL descriptors (export/import regions)     ││  │
│  │  │  - Exchange EncodeLayerMetadata (precision vector, etc.) ││  │
│  │  │  - Coordinate probe escalation requests                  ││  │
│  │  └──────────────────────────────────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  NIXL Backend Selection                                      │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │  │
│  │  │  UCX     │ │  GDS     │ │  TCP     │ │  NVLink      │    │  │
│  │  │(RDMA/    │ │(GPU Dir  │ │(fallback)│ │(intra-node)  │    │  │
│  │  │ RoCE)    │ │ Storage) │ │          │ │              │    │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 NIXLTransportAgent Implementation

```python
# File: spectrumkv/vllm_connector/nixl_transport.py

import nixl.api as nixl_api
from typing import Optional

@dataclass
class NIXLTransferDesc:
    """NIXL transfer descriptor exchanged via side channel."""
    # NIXL export region handle (opaque)
    export_handle: bytes
    # Remote agent name
    remote_agent_name: str
    # Buffer address and size
    buffer_addr: int
    buffer_size: int
    # Tensor ID for matching
    tensor_id: str


class NIXLTransportAgent:
    """NIXL-based transport agent for SpectrumKV.
    
    Manages async KV data transfer between prefill and decode workers
    using NVIDIA's NIXL library. Operates on precision-encoded
    (compressed) data buffers rather than raw KV cache tensors.
    """

    def __init__(
        self,
        spectrum_config: SpectrumKVConfig,
        rank: int,
        world_size: int,
    ) -> None:
        self.config = spectrum_config

        # Initialize NIXL agent
        self.nixl_agent = nixl_api.nixl_agent(
            name=f"spectrumkv_rank{rank}",
            backends=spectrum_config.nixl_backends,
        )

        # Transfer buffer for encoded data
        self.transfer_buffer = torch.empty(
            spectrum_config.nixl_buffer_size,
            dtype=torch.uint8,
            device=spectrum_config.nixl_buffer_device,
        )

        # Register buffer with NIXL
        self._registered_buffer = self.nixl_agent.register_memory(
            self.transfer_buffer
        )

        # Pending transfers tracking
        self._pending_sends: dict[str, nixl_api.nixl_xfer] = {}
        self._pending_recvs: dict[str, nixl_api.nixl_xfer] = {}

        # Side channel for metadata exchange
        self._side_channel: Optional["SideChannel"] = None

    def setup_side_channel(
        self,
        remote_host: str,
        remote_port: int,
        is_sender: bool,
    ) -> None:
        """Establish side channel for NIXL descriptor exchange.
        
        Uses ZMQ DEALER/ROUTER pattern (same as P2pNcclConnector).
        """
        self._side_channel = SideChannel(
            remote_host=remote_host,
            remote_port=remote_port,
            is_sender=is_sender,
        )

    def async_send(
        self,
        tensor_id: str,
        data: torch.Tensor,
        metadata: EncodeLayerMetadata,
    ) -> None:
        """Initiate async NIXL write for encoded KV data.
        
        Pipeline:
        1. Copy encoded data to transfer buffer
        2. Create NIXL export region descriptor
        3. Exchange descriptor via side channel
        4. Issue NIXL write_async
        
        Args:
            tensor_id: "request_id#layer_name"
            data: Precision-encoded GPU tensor
            metadata: Encoding metadata for decode side
        """
        # 1. Copy to transfer buffer (overlap with next layer compute)
        offset = self._alloc_buffer_space(tensor_id, data.numel())
        self.transfer_buffer[offset:offset + data.numel()] = data.flatten().to(torch.uint8)

        # 2. Create export region
        export_desc = self.nixl_agent.export_memory(
            self._registered_buffer,
            offset=offset,
            length=data.numel(),
        )

        # 3. Send descriptor + metadata via side channel
        transfer_desc = NIXLTransferDesc(
            export_handle=export_desc.serialize(),
            remote_agent_name=self.nixl_agent.name,
            buffer_addr=offset,
            buffer_size=data.numel(),
            tensor_id=tensor_id,
        )
        self._side_channel.send_metadata(tensor_id, transfer_desc, metadata)

        # 4. Issue async write
        remote_desc = self._side_channel.get_remote_descriptor(tensor_id)
        xfer = self.nixl_agent.write_async(
            src=self._registered_buffer,
            dst_desc=remote_desc,
        )
        self._pending_sends[tensor_id] = xfer

    def async_recv(
        self,
        tensor_id: str,
    ) -> None:
        """Initiate async NIXL read for incoming encoded KV data.
        
        Pipeline:
        1. Receive descriptor via side channel
        2. Register import region
        3. Issue NIXL read_async
        """
        # 1. Wait for descriptor from sender
        transfer_desc, encode_metadata = self._side_channel.recv_metadata(tensor_id)

        # Store metadata for later decode
        self._recv_metadata[tensor_id] = encode_metadata

        # 2. Register import region
        import_desc = self.nixl_agent.import_memory(
            transfer_desc.export_handle,
            self._registered_buffer,
        )

        # 3. Issue async read
        offset = self._alloc_buffer_space(tensor_id, transfer_desc.buffer_size)
        xfer = self.nixl_agent.read_async(
            src_desc=import_desc,
            dst=self._registered_buffer,
            dst_offset=offset,
        )
        self._pending_recvs[tensor_id] = xfer

    def wait_send(self, tensor_id: str) -> None:
        """Wait for a specific async send to complete."""
        if tensor_id in self._pending_sends:
            self._pending_sends[tensor_id].wait()
            del self._pending_sends[tensor_id]

    def wait_recv(self, tensor_id: str) -> tuple[torch.Tensor, EncodeLayerMetadata]:
        """Wait for a specific async recv to complete.
        
        Returns:
            (received_encoded_data, encode_metadata)
        """
        if tensor_id in self._pending_recvs:
            self._pending_recvs[tensor_id].wait()
            del self._pending_recvs[tensor_id]

        # Extract received data from transfer buffer
        offset, size = self._buffer_allocations[tensor_id]
        data = self.transfer_buffer[offset:offset + size]
        metadata = self._recv_metadata[tensor_id]

        return data, metadata

    def wait_all_sends(self) -> None:
        """Wait for all pending sends to complete."""
        for xfer in self._pending_sends.values():
            xfer.wait()
        self._pending_sends.clear()

    # ─── Buffer Management ────────────────────────────────────────────

    def _alloc_buffer_space(self, tensor_id: str, size: int) -> int:
        """Allocate space in the transfer buffer using bump allocator."""
        offset = self._buffer_offset
        self._buffer_offset += size
        self._buffer_allocations[tensor_id] = (offset, size)
        return offset

    def _reset_buffer(self) -> None:
        """Reset transfer buffer for next scheduling cycle."""
        self._buffer_offset = 0
        self._buffer_allocations.clear()
```

### 5.3 Bandwidth Savings Analysis

For a typical LLM with `num_layers=32`, `num_kv_heads=32`, `head_size=128`, `block_size=16`:

| Metric | Full FP16 | SpectrumKV (default ratio) | Savings |
|--------|-----------|---------------------------|---------|
| Bytes per token (K+V) | 2 × 32 × 128 × 2 = 16,384 B | Mixed: see below | — |
| FP16 tokens (15%) | 16,384 × 0.15 = 2,458 B | — | — |
| FP8 tokens (35%) | 16,384 × 0.35 × 0.5 = 2,867 B | — | — |
| INT4 tokens (35%) | 16,384 × 0.35 × 0.25 + scales ≈ 1,536 B | — | — |
| DROP tokens (15%) | 0 B | — | — |
| **Total per token (avg)** | **16,384 B** | **≈ 6,861 B** | **58.1%** |

With NIXL RDMA transfer at 50 GB/s (InfiniBand), a 4096-token prompt KV transfer:

| | Full FP16 | SpectrumKV |
|---|-----------|-----------|
| Transfer size | 64 MB | 26.8 MB |
| Transfer time @ 50 GB/s | 1.28 ms | 0.54 ms |
| TTFT improvement | — | **~0.74 ms saved** |

---

## 6. Configuration & API Design

### 6.1 vLLM KV Transfer Config

SpectrumKV is configured through vLLM's `--kv-transfer-config` flag:

```bash
# Prefill instance
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8010 \
    --kv-transfer-config '{
        "kv_connector": "SpectrumKVConnector",
        "kv_role": "kv_producer",
        "kv_connector_module_path": "spectrumkv.vllm_connector.connector",
        "kv_connector_extra_config": {
            "sws_scoring_method": "attention_norm",
            "sws_fp16_ratio": 0.15,
            "sws_fp8_ratio": 0.35,
            "sws_int4_ratio": 0.35,
            "enable_probe": true,
            "probe_quality_threshold": 0.85,
            "probe_method": "cosine_similarity",
            "probe_layers": [0, 15, 31],
            "max_escalation_retries": 2,
            "nixl_backends": ["UCX"],
            "nixl_buffer_device": "cuda",
            "nixl_buffer_size": 2147483648,
            "nixl_side_channel_port": 55555,
            "enable_fp8": true,
            "enable_int4": true,
            "int4_scale_mode": "per_head"
        }
    }'

# Decode instance
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8020 \
    --kv-transfer-config '{
        "kv_connector": "SpectrumKVConnector",
        "kv_role": "kv_consumer",
        "kv_connector_module_path": "spectrumkv.vllm_connector.connector",
        "kv_connector_extra_config": {
            "enable_probe": true,
            "probe_quality_threshold": 0.85,
            "probe_method": "cosine_similarity",
            "probe_layers": [0, 15, 31],
            "nixl_backends": ["UCX"],
            "nixl_buffer_device": "cuda",
            "nixl_buffer_size": 2147483648,
            "nixl_side_channel_port": 55556
        }
    }'
```

### 6.2 SpectrumKV Python API

```python
# File: spectrumkv/__init__.py

from spectrumkv.vllm_connector.connector import SpectrumKVConnector
from spectrumkv.vllm_connector.config import SpectrumKVConfig
from spectrumkv.vllm_connector.metadata import (
    SpectrumKVMetadata,
    SpectrumKVRequestMeta,
    TokenPrecisionMap,
    PrecisionLevel,
)

__all__ = [
    "SpectrumKVConnector",
    "SpectrumKVConfig",
    "SpectrumKVMetadata",
    "SpectrumKVRequestMeta",
    "TokenPrecisionMap",
    "PrecisionLevel",
]
```

### 6.3 Programmatic API

```python
# Offline inference with SpectrumKV
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from spectrumkv import SpectrumKVConfig

# Build SpectrumKV config
spectrum_config = SpectrumKVConfig(
    sws_scoring_method="attention_norm",
    sws_fp16_ratio=0.15,
    sws_fp8_ratio=0.35,
    sws_int4_ratio=0.35,
    enable_probe=True,
    probe_quality_threshold=0.85,
)

# Configure KV transfer
ktc = KVTransferConfig(
    kv_connector="SpectrumKVConnector",
    kv_role="kv_both",
    kv_connector_module_path="spectrumkv.vllm_connector.connector",
    kv_connector_extra_config=spectrum_config.to_dict(),
)

# Initialize LLM
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    kv_transfer_config=ktc,
    max_model_len=32768,
    gpu_memory_utilization=0.85,
)

# Run inference
prompts = ["Explain quantum computing in detail." * 100]
sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)
```

### 6.4 Observability API

```python
# SpectrumKV exposes metrics through vLLM's observability framework

# Prometheus metrics (auto-registered when connector is active)
spectrumkv_tokens_sent_total = Counter(
    "spectrumkv_tokens_sent_total",
    "Total tokens sent by precision level",
    ["precision_level"],
)

spectrumkv_compression_ratio = Histogram(
    "spectrumkv_compression_ratio",
    "Effective compression ratio per transfer",
    buckets=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
)

spectrumkv_probe_quality = Histogram(
    "spectrumkv_probe_quality_score",
    "Probe quality score per layer evaluation",
    ["layer"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
)

spectrumkv_escalation_total = Counter(
    "spectrumkv_escalation_total",
    "Total precision escalation events",
)

spectrumkv_transfer_bandwidth_bytes = Counter(
    "spectrumkv_transfer_bandwidth_bytes",
    "Total bytes transferred over NIXL",
    ["direction"],  # "send" or "recv"
)
```

### 6.5 Dynamo Integration Config

For integration with NVIDIA Dynamo's KV-aware router:

```yaml
# Dynamo configuration for SpectrumKV
# File: dynamo_spectrumkv_config.yaml

api_server:
  model: meta-llama/Llama-3.1-70B-Instruct
  port: 8000

prefill:
  kv_connector: SpectrumKVConnector
  kv_connector_module_path: spectrumkv.vllm_connector.connector
  kv_role: kv_producer
  spectrumkv:
    sws_fp16_ratio: 0.15
    sws_fp8_ratio: 0.35
    sws_int4_ratio: 0.35
    enable_probe: true

decode:
  kv_connector: SpectrumKVConnector
  kv_connector_module_path: spectrumkv.vllm_connector.connector
  kv_role: kv_consumer
  spectrumkv:
    enable_probe: true
    probe_quality_threshold: 0.85

router:
  kv_aware: true
  # SpectrumKV emits KV_CACHE_STORED events with precision metadata
  # Router uses these events for cache hit scoring
  # Higher-precision blocks score higher for routing decisions
```

---

## 7. Implementation Roadmap

### Milestone 1: Core Connector + FP8 Compression (Week 1-4)

**Goal**: Basic end-to-end PD disaggregated serving with FP8-only compression, no Probe.

| Task | Description | Est. Effort |
|------|-------------|-------------|
| **M1.1** | Implement `SpectrumKVConnector` skeleton with `KVConnectorBase_V1` interface | 3 days |
| **M1.2** | Implement `SpectrumKVSchedulerManager` with basic `get_num_new_matched_tokens()`, `build_connector_meta()` | 3 days |
| **M1.3** | Implement `SpectrumKVWorkerHandler` with `start_load_kv()`, `save_kv_layer()`, `wait_for_layer_load()` | 4 days |
| **M1.4** | Implement `QCBMCodecEngine` with FP8 E4M3 per-head quantization/dequantization | 5 days |
| **M1.5** | Implement `NIXLTransportAgent` wrapping NIXL read/write API with async pipeline | 5 days |
| **M1.6** | Implement `SpectrumKVBlockManager` with basic precision-aware buffer allocation | 3 days |
| **M1.7** | Integration test: single-request PD transfer with FP8 compression | 3 days |
| **M1.8** | Baseline benchmark: measure transfer time and TTFT vs. raw NixlConnector | 2 days |

**Deliverable**: Working `SpectrumKVConnector` that performs FP8-only KV compression with ~2x bandwidth reduction.

**Acceptance Criteria**:
- [ ] End-to-end PD serving with SpectrumKV produces identical outputs to raw NixlConnector (within FP8 quantization error)
- [ ] TTFT improvement of ≥15% over raw NixlConnector for 4K-token prompts
- [ ] No correctness regressions in vLLM scheduler behavior

### Milestone 2: SWS + INT4 + Probe (Week 5-8)

**Goal**: Add per-token mixed precision with SWS ranking, INT4 quantization, and Probe-based quality monitoring.

| Task | Description | Est. Effort |
|------|-------------|-------------|
| **M2.1** | Implement `SWSRanker` with heuristic scoring (position-based) | 3 days |
| **M2.2** | Implement `QCBMCodecEngine` INT4 per-head quantization with CUDA kernel | 7 days |
| **M2.3** | Implement per-token mixed-precision encode/decode pipeline | 5 days |
| **M2.4** | Implement `ProbeEvaluator` with cosine_similarity method | 4 days |
| **M2.5** | Implement `ProbeController` with escalation logic | 3 days |
| **M2.6** | Implement probe feedback loop via `update_connector_output()` | 3 days |
| **M2.7** | Implement SWS attention-based scoring (using cached attention weights) | 5 days |
| **M2.8** | Multi-request concurrent PD serving test | 3 days |
| **M2.9** | Benchmark: mixed-precision vs. FP8-only vs. raw transfer | 3 days |

**Deliverable**: Full SpectrumKV with SWS + QCBM + Probe, achieving ~3-4x bandwidth reduction.

**Acceptance Criteria**:
- [ ] Mixed-precision transfer achieves ≥3x compression ratio at default ratios
- [ ] Probe escalation successfully recovers quality for degraded requests
- [ ] Multi-request concurrent serving stable under load (no deadlocks, no memory leaks)
- [ ] Quality parity with FP16 transfer for >95% of test prompts (within probe threshold)

### Milestone 3: Dynamo Integration + Production Hardening (Week 9-12)

**Goal**: NVIDIA Dynamo KV-aware router integration, production observability, and performance optimization.

| Task | Description | Est. Effort |
|------|-------------|-------------|
| **M3.1** | Implement `take_events()` for Dynamo KV cache event emission | 3 days |
| **M3.2** | Implement precision-aware KV cache hit scoring for Dynamo router | 4 days |
| **M3.3** | Add Prometheus metrics export (compression ratio, probe quality, etc.) | 3 days |
| **M3.4** | Implement `MultiConnector` wrapper: SpectrumKVConnector + OffloadingConnector | 4 days |
| **M3.5** | Optimize CUDA kernels for INT4 pack/unpack (replace CPU fallback) | 5 days |
| **M3.6** | Optimize encode/decode pipeline: overlap NIXL transfer with compute | 5 days |
| **M3.7** | GDS backend support for KV cache offloading to SSD | 4 days |
| **M3.8** | Stress test: 100+ concurrent requests, varying prompt lengths | 3 days |
| **M3.9** | End-to-end benchmark with Dynamo Smart Router | 4 days |
| **M3.10** | Documentation, examples, and quickstart guide | 3 days |

**Deliverable**: Production-ready SpectrumKV plugin with Dynamo integration and comprehensive observability.

**Acceptance Criteria**:
- [ ] Dynamo KV-aware router correctly routes to SpectrumKV-enabled workers
- [ ] Prometheus metrics export working with Grafana dashboard
- [ ] Stress test passes: 100+ concurrent requests with 99th percentile TTFT < SLA
- [ ] ≥55% bandwidth reduction on average with quality within probe threshold
- [ ] Clean integration with MultiConnector pattern for offloading + SpectrumKV

---

## 8. Appendix

### 8.1 Comparison with Existing Connectors

| Feature | NixlConnector | LMCacheConnectorV1 | MooncakeConnector | **SpectrumKVConnector** |
|---------|---------------|---------------------|-------------------|------------------------|
| Transfer precision | FP16 (raw) | FP16 / configurable | FP16 (raw) | **Per-token mixed (FP16/FP8/INT4/DROP)** |
| Compression | None | CacheGen bitstream | None | **SWS-guided QCBM** |
| Quality monitoring | None | None | None | **Probe with escalation** |
| Selective recompute | No | CacheBlend | No | **DROP token recompute** |
| NIXL backend | Yes | Yes (via LMCache) | No (own engine) | **Yes (native)** |
| KV-aware routing | Via Dynamo | Via LMCache controller | Via proxy | **Via Dynamo events** |
| Bandwidth savings | 0% | ~50% (CacheGen) | 0% | **55-75%** |

### 8.2 Relationship to CacheBlend

SpectrumKV's DROP token handling is architecturally similar to LMCache's CacheBlend selective recompute:

| Aspect | CacheBlend | SpectrumKV DROP |
|--------|-----------|-----------------|
| Trigger | Non-prefix segment boundary | Low importance score from SWS |
| Mechanism | Recompute select boundary tokens | Zero-fill + attention naturally ignores |
| Scope | RAG scenarios with cached document KV | PD transfer bandwidth optimization |
| Integration point | `start_load_kv` in LMCache adapter | `wait_for_layer_load` in SpectrumKV hook |

Key difference: CacheBlend recomputes KV at segment boundaries to correct cross-attention; SpectrumKV DROP tokens are simply zeroed in the KV cache, and the attention layer's softmax naturally suppresses their influence. If quality degrades, the Probe mechanism escalates precision rather than recomputing.

### 8.3 Relationship to NVIDIA Dynamo KVBM

Dynamo's KV Block Manager (KVBM) provides multi-tier KV cache offloading with NIXL transport. SpectrumKV complements KVBM:

- **KVBM**: Manages KV cache lifecycle across GPU/CPU/SSD tiers (where to store)
- **SpectrumKV**: Manages KV cache precision during transfer (how efficiently to transmit)

Integration pattern: Use `MultiConnector` to compose `SpectrumKVConnector` + `DynamoConnector`:

```bash
# Dynamo with SpectrumKV compression
vllm serve Qwen/Qwen3-70B \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "SpectrumKVConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "sws_fp16_ratio": 0.15,
                        "sws_fp8_ratio": 0.35,
                        "sws_int4_ratio": 0.35,
                        "enable_probe": true
                    }
                },
                {
                    "kv_connector": "DynamoConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "kvbm.vllm_integration.connector"
                }
            ]
        }
    }'
```

### 8.4 File Structure

```
spectrumkv/
├── __init__.py                          # Public API exports
├── vllm_connector/
│   ├── __init__.py
│   ├── connector.py                     # SpectrumKVConnector (KVConnectorBase_V1)
│   ├── scheduler_manager.py             # SpectrumKVSchedulerManager
│   ├── worker_handler.py                # SpectrumKVWorkerHandler
│   ├── block_manager.py                 # SpectrumKVBlockManager + PrecisionBlock
│   ├── metadata.py                      # SpectrumKVMetadata, SpectrumKVRequestMeta
│   ├── config.py                        # SpectrumKVConfig, SWSConfig, ProbeConfig
│   ├── attention_hook.py                # SpectrumKVAttentionHook
│   ├── nixl_transport.py               # NIXLTransportAgent + SideChannel
│   ├── sws_ranker.py                    # SWSRanker
│   ├── qcbm_codec.py                    # QCBMCodecEngine + EncodedLayerData
│   ├── probe.py                         # ProbeEvaluator + ProbeController
│   └── cuda_kernels/
│       ├── __init__.py
│       ├── int4_pack.cu                 # INT4 pack/unpack CUDA kernels
│       ├── fp8_quant.cu                 # FP8 quantize/dequantize CUDA kernels
│       └── mixed_precision_encode.cu    # Fused mixed-precision encode kernel
├── examples/
│   ├── disagg_prefill_spectrumkv.sh     # Basic PD disaggregated serving
│   ├── disagg_with_dynamo.sh            # PD with Dynamo KV-aware routing
│   └── offline_inference.py             # Offline inference example
├── tests/
│   ├── test_connector.py                # Connector lifecycle tests
│   ├── test_qcbm_codec.py              # Codec correctness tests
│   ├── test_sws_ranker.py              # SWS ranking tests
│   ├── test_probe.py                    # Probe evaluation tests
│   ├── test_nixl_transport.py          # NIXL transport tests
│   └── test_e2e_pd.py                  # End-to-end PD serving tests
└── docs/
    ├── architecture.md                  # This document
    ├── quickstart.md                    # Getting started guide
    └── configuration.md                 # Configuration reference
```

### 8.5 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Precision only affects transfer, not GPU storage | Decode worker always uses full-precision KV cache for attention; avoids any attention kernel modifications |
| SWS runs on scheduler, not worker | Avoids GPU compute overhead during prefill; scheduler has request-level context |
| Probe uses cosine similarity (not attention shift) | Can be evaluated without running the full attention forward pass; cheaper |
| NIXL as sole transport (no NCCL fallback) | NIXL is SM-free (RDMA offload), which is critical when GPU is compute-bound during prefill |
| DROP tokens zero-filled, not re-computed | Cheaper than selective recompute; Probe escalates if quality drops |
| Per-head quantization (not per-tensor) | Better accuracy than per-tensor; cheaper than per-token; aligns with vLLM's FP8 E4M3 kv_cache practice |

### 8.6 References

- [vLLM Disaggregated Prefilling Documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [KVConnectorBase_V1 API Reference](https://docs.vllm.ai/en/v0.10.1/api/vllm/distributed/kv_transfer/kv_connector/v1/base.html)
- [NVIDIA Dynamo High Level Architecture](https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/overall-architecture)
- [NVIDIA NIXL and Disaggregated Inference Guide](https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/)
- [LMCache Architecture & Developer Guide](https://docs.lmcache.ai/mp/architecture.html)
- [CacheBlend: Fast LLM Serving for RAG (EuroSys'25)](https://dl.acm.org/doi/pdf/10.1145/3790254)
- [vLLM PD Separation KV Transfer Analysis](https://cloud.tencent.cn/developer/article/2647154)
- [KV Transfer Engine Benchmarks (UCCL)](https://uccl-project.github.io/posts/kv-transfer-engine/)
