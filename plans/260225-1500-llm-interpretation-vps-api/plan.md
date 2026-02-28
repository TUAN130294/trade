---
title: "LLM Interpretation & VPS API Integration"
description: "Add AI-powered Vietnamese insights to 18 endpoints and replace CafeF with VPS as primary price source"
status: pending
priority: P1
effort: 12h
branch: feat/llm-interpretation-vps-api
tags: [llm, vps-api, interpretation, vietnamese, market-data]
created: 2026-02-25
---

# LLM Interpretation & VPS API Integration

## Problem Statement

1. **18/21 endpoints return raw numbers** without human-readable insight. Users see RSI=35 but not "Co phieu dang o vung qua ban, co the la co hoi mua".
2. **CafeF price data deviates ~2M VND** from actual. VPS API (`bgapidatafeed.vps.com.vn`) is accurate and includes foreign flow data (`fBVol`/`fSVolume`).

## Solution Overview

- Create shared `InterpretationService` that wraps `LLMClient` with caching, model routing, and prompt templates
- Create `VPSDataConnector` as primary price source, keep CafeF as fallback
- Add `?interpret=true` query param to 18 endpoints (opt-in, no perf hit by default)
- Cache LLM responses 5 min via existing `MemoryCache`

## Architecture

```
[Router Endpoint]
  |-- ?interpret=true --> InterpretationService.interpret(context, template_key)
  |                            |-- check cache (5min TTL)
  |                            |-- select model (haiku=fast, sonnet=deep)
  |                            |-- call LLMClient.chat()
  |                            |-- cache response
  |                            |-- return Vietnamese text
  |
  |-- Price data --> VPSDataConnector (primary)
                        |-- fallback --> CafeF (existing)
```

## Phases

| # | Phase | Status | Effort |
|---|-------|--------|--------|
| 1 | [VPS Data Connector](phase-01-vps-data-connector.md) | pending | 2h |
| 2 | [Interpretation Service](phase-02-interpretation-service.md) | pending | 3h |
| 3 | [Endpoint Integration - Market & Analysis](phase-03-endpoint-integration-market.md) | pending | 3h |
| 4 | [Endpoint Integration - News & Trading](phase-04-endpoint-integration-news-trading.md) | pending | 2h |
| 5 | [Testing & Validation](phase-05-testing-validation.md) | pending | 2h |

## Key Decisions

- **Opt-in interpretation**: `?interpret=true` keeps existing API contract intact
- **Model routing**: haiku for fast endpoints (status, signals), sonnet for deep analysis (technical, backtest)
- **No Redis dependency**: Use existing `MemoryCache` from `quantum_stock/utils/cache.py`
- **VPS as primary**: Better accuracy + foreign flow data; CafeF retained as fallback
- **Vietnamese only**: All interpretations in Vietnamese (target audience)

## Dependencies

- LLM proxy at `http://localhost:8317` with key `sk-***REDACTED***`
- VPS API at `https://bgapidatafeed.vps.com.vn` (public, no auth)
- Existing `LLMClient` in `quantum_stock/agents/llm_agents.py`
- Existing `MemoryCache` in `quantum_stock/utils/cache.py`
