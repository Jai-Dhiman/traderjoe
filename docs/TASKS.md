# TraderJoe Phase 3: ACE E2E, Evening Review, and Signals

Status: Phase 2 Core Complete → Phase 3 Execution & Learning

Goal: Finish ACE end-to-end (Generator–Reflector–Curator with deltas + playbook), complete missing data integrations, add baseline ML signals and backtesting, and deliver evening learning loop with observability.

---

## What’s Already Done (removed from task list)

- Local embeddings (EmbeddingGemma mock, 768-dim) — `src/embeddings/mod.rs`
- Vector store + similarity search + HNSW — `src/vector/mod.rs`
- ACE Context DAO — `src/ace/context.rs`
- Ollama client with explicit errors — `src/llm/mod.rs`
- Prompt templates — `src/ace/prompts.rs`
- Exa API integration with persistence + fallbacks — `src/data/research.rs`
- Morning analysis orchestrator — `src/orchestrator/morning.rs` and CLI analyze
- ACE query and playbook stats — CLI commands implemented
- Phase 2 verification script — `scripts/verify_phase2.sh`

---

## Task Overview (Remaining P2 + New P3)

| Workstream | Tasks | Status | ETA |
|------------|-------|--------|-----|
| ACE Core (Playbook + Deltas) | P3-1 – P3-4 | Planned | 5-7 hours |
| Evening Review & Orchestration | P3-5 – P3-6 | Planned | 2-3 hours |
| Data Integrations | P2-9, P2-10 | Planned | 2-3 hours |
| ML Signals & Backtest | P2-14 – P2-15 | Planned | 2-3 hours |
| Quality & Observability | P2-16 – P2-20 | Planned | 2-3 hours |

Total Estimated Time: 13-18 hours

---

## Remaining Phase 2 Tasks

### Data Integrations

#### P2-9: Replace Reddit Stub with Real API
Priority: High | Dependencies: Config | ETA: 90 mins

Files:
- `src/data/sentiment.rs`

Implementation:
- OAuth with `REDDIT_CLIENT_ID/SECRET`; fetch posts/comments (r/wallstreetbets, r/stocks, r/options)
- Heuristic sentiment (lexicon + upvote-weighting); persist to `sentiment`

Verification:
- `traderjoe sentiment --source reddit` stores non-stub entries

#### P2-10: News & VIX Enrichment
Priority: Medium | Dependencies: Data layer | ETA: 60 mins

Files:
- `src/data/news.rs` (new), `src/data/market.rs`

Implementation:
- Add VIX fetch (Yahoo)
- Minimal news ingestion (RSS or NewsAPI); persist and surface in morning analysis

Verification:
- CLI subcommands return record counts; morning analysis shows enriched context

### ML Signals (Baselines)

#### P2-14: Technical Indicators in Rust
Priority: Medium | Dependencies: Market data | ETA: 60 mins

Files:
- `src/data/indicators.rs` (new)

Implementation:
- RSI(14), MACD(12,26,9), SMA(20/50/200); expose `compute(&[OHLCV]) -> Signals`
- Integrate into morning pipeline

Verification:
- Spot-check vs reference implementation; unit tests

#### P2-15: Backtest Baseline Strategy
Priority: Medium | Dependencies: P2-14 | ETA: 60 mins

Files:
- `src/cli/commands.rs` (Backtest)

Implementation:
- Rule: calls if SMA20>SMA50 and RSI<65; puts if SMA20<SMA50 and RSI>35; else flat
- Include slippage/commissions; print CAGR, Sharpe, max drawdown

Verification:
- Runs over last N months; prints metrics without panics

### Quality & Verification

#### P2-16: Tests (Embeddings, Vector, LLM)
Priority: High | ETA: 60 mins

Files:
- `tests/ace_integration.rs` (new)

Implementation:
- Mock LLM returns deterministic JSON
- Similarity search order checks
- Embedding dimension checks

#### P2-18: Observability Improvements
Priority: Low | ETA: 45 mins

Files:
- Add tracing spans for morning/evening pipelines
- Add error contexts for all external calls

#### P2-19: Documentation Update
Priority: Low | ETA: 45 mins

Files:
- `README.md` (Phase 3 usage), link `docs/ace_framework.md`

#### P2-20: Code Quality
Priority: Low | ETA: 30 mins

Commands:
- `cargo fmt`
- `cargo clippy -- -D warnings`
- `cargo test`

---

## Phase 3 Tasks

### ACE Core: Playbook + Deltas + Agents

#### P3-1: Playbook Schema & DAO
Priority: High | Dependencies: ace_contexts | ETA: 60 mins

Files:
- `migrations/004_playbook.sql` (new)
- `src/ace/playbook.rs` (new)

Implementation:
- Tables: `playbook_bullets(id uuid pk, section text, content text, helpful_count int, harmful_count int, confidence real, last_used timestamptz)`
- Optional relations to `ace_contexts`
- DAO: insert/update/query by section; staleness queries

Verification:
- Insert/retrieve/update roundtrip

#### P3-2: Delta Engine (`delta.rs`)
Priority: High | Dependencies: P3-1 | ETA: 90 mins

Files:
- `src/ace/delta.rs` (new)

Implementation:
- `Delta { op: Add|Update|Remove, section, content, bullet_id?: Uuid, meta }`
- Curator applies deltas to playbook tables or JSONB
- Dedup pass via cosine similarity threshold using embeddings

Verification:
- Unit tests for apply/merge/dedup

#### P3-3: Generator / Reflector / Curator
Priority: High | Dependencies: P3-2 | ETA: 2 hours

Files:
- `src/ace/generator.rs`, `src/ace/reflector.rs`, `src/ace/curator.rs` (new)

Implementation:
- Generator: build inputs + call LLM; emit candidate bullets
- Reflector: compare outcome vs expectation → produce deltas
- Curator: merge deltas; update helpful/harmful counts; adjust confidence

Verification:
- In-memory E2E with mocked LLM

#### P3-4: Grow-and-Refine Cycle
Priority: Medium | Dependencies: P3-3 | ETA: 60 mins

Files:
- `src/ace/refine.rs` (new)

Implementation:
- Periodic pruning/boosting of bullets based on outcomes and recency
- Similarity-based consolidation; staleness decay

Verification:
- Unit tests for lifecycle changes

### Evening Review & Orchestration

#### P3-5: Evening Review Pipeline
Priority: High | Dependencies: P3-3 | ETA: 90 mins

Files:
- `src/orchestrator/evening.rs` (new)
- CLI `review` command wiring

Implementation:
- Compute outcome (e.g., next-day close vs entry), excursions
- Run Reflector → Curator applies deltas → persist outcome

Verification:
- Playbook stats and bullet counters change after review

#### P3-6: Orchestrator Integration
Priority: Medium | Dependencies: P3-5 | ETA: 45 mins

Files:
- `src/orchestrator/morning.rs` (update)

Implementation:
- Feed playbook summaries into morning prompt inputs
- Enforce confidence gates in position sizing

Verification:
- Morning decisions reference recent playbook entries

---

## Success Criteria

Working System:
- `traderjoe analyze --symbol SPY` returns decision, confidence, reasoning
- `traderjoe ace-query "similar to today"` returns top-k contexts with scores
- Reddit and News/VIX enrich market state

ACE Ready:
- Playbook delta pipeline functioning (add/update/remove)
- Generator/Reflector/Curator produce and apply deltas end-to-end
- Grow-and-Refine maintains a non-stale playbook

Quality:
- Structured logs around each pipeline step
- Tests pass; verification script green

---

Last updated: 2025-10-22T00:05:00Z
