# TraderJoe - ACE Framework Production Tasks

**Status**: ✅ ALL FOUR CRITICAL FIXES VERIFIED WORKING (2025-10-29)
**Previous Test Results**: 4 trades, 0% win rate, -40% cumulative loss, 0.0h duration
**Final Test Results**:

- ✅ Slippage: 0.5% (was 3%)
- ✅ Duration: 30.5h (was 0.0h)
- ✅ Bullets: 104 created with correct historical timestamps (2024-09-03 to 09-06)
- ✅ Retrieval: Multi-day learning verified (Day 1: 9, Day 2: 4, Day 3: 9, Day 4: 3, Day 5: 11 entries)
**Current Status**: Phase 1 Complete - Ready for Phase 2 (Risk Management)

---

## 🎯 Recent Progress (2025-10-28 to 2025-10-29)

### Completed 2025-10-28

✅ **Task #1: Fix Playbook Retrieval for Backtests**

- Identified root cause: `INSERT INTO playbook_bullets` used `NOW()` instead of backtest date
- Implemented fix across 4 files: PlaybookDAO, DeltaEngine, Curator, EveningOrchestrator
- Added comprehensive unit test for time-travel queries
- Added detailed logging for timestamp usage

✅ **Task #2: Verify Playbook Entries Reach LLM Prompt**

- Created ablation test to verify playbook influence on decisions
- Test compares empty vs populated playbook for measurable differences
- Validates that playbook entries actually affect trading decisions

✅ **Task #3: Fix Immediate Trade Exits**

- Identified root cause: `exit_trade` used `Utc::now()` instead of historical time
- Created `exit_trade_with_time` function with optional timestamp parameter
- Updated evening orchestrator to pass historical exit time in backtest mode

✅ **Task #4: Fix Confidence Filter**

- Changed threshold from >0.6 to >0.45 to allow new bullets (start at 0.5)
- Updated in `src/orchestrator/morning.rs:421`

## 🔴 HIGH PRIORITY (P1) - Required for Functionality

### 5. Implement Confidence Calibration

**Location**: `src/orchestrator/morning.rs:generate_decision()`

**Problem**:

- System maintains 81-85% confidence after 4 consecutive losses
- Confidence should drop after repeated failures
- Currently decorrelated from actual performance

**Solution**:

```rust
// Add historical performance tracking
struct PerformanceTracker {
    recent_trades: VecDeque<(f32, bool)>, // (confidence, won)
    calibration_factor: f32,
}

// Adjust confidence based on recent accuracy
fn calibrate_confidence(
    raw_confidence: f32,
    recent_win_rate: f32,
    recent_losses: usize
) -> f32 {
    // Reduce confidence after consecutive losses
    let penalty = recent_losses as f32 * 0.05;
    (raw_confidence - penalty).max(0.3)
}
```

**Acceptance Criteria**:

- [ ] After 3+ losses in a row, confidence capped at 60%
- [ ] Confidence gradually recovers with wins
- [ ] Log: "Confidence adjusted: {raw} → {calibrated} (recent_win_rate: {rate})"

---

### 6. Enforce Risk Factor Weighting

**Location**: `src/ace/prompts.rs` (prompt template)

**Problem**:

- Every decision says: "sentiment_label is bearish but not concerning"
- System dismisses contrary indicators instead of weighting them
- Contradicts ACE framework principles

**Solution**:

- [ ] Update prompt to require explicit risk assessment:

```
You MUST address each risk factor:
- Bearish sentiment: How does this affect your confidence?
- High volatility: What's your specific risk mitigation?
- Conflicting signals: Explain how you weighted them

DO NOT dismiss risk factors. If they exist, they affect confidence.
```

- [ ] Add validation: reject decisions that ignore flagged risks
- [ ] Require confidence penalty for conflicting signals

**Acceptance Criteria**:

- [ ] No more "but not concerning" dismissals
- [ ] Risk factors reduce confidence by at least 5-10% each
- [ ] Test: bearish sentiment + high volatility → confidence < 70%

---

### 7. Implement Proper Risk Management

**Locations**: Multiple files

**Problem**:

- No stop-loss logic
- No position sizing based on volatility
- No risk limits per trade

**Solution**:

- [ ] Add stop-loss at entry: 15% for options
- [ ] Position sizing formula:

```rust
fn calculate_position_size(
    account_balance: f64,
    volatility: f64,
    confidence: f32,
) -> f64 {
    let base_risk = 0.02; // Risk 2% of account
    let vol_adjustment = (1.0 - volatility / 100.0).max(0.3);
    let conf_adjustment = confidence / 100.0;

    account_balance * base_risk * vol_adjustment * conf_adjustment
}
```

- [ ] Maximum 5% of account per trade
- [ ] No trading if VIX > 40 unless confidence > 85%

---

### 8. Prevent Playbook Inflation

**Location**: `src/ace/curator.rs`

**Problem**:

- Playbook grew from 107 → 119 bullets in 4 days (12% growth rate)
- Many low-confidence bullets (0.43-0.49)
- Suggests bullets aren't being pruned effectively

**Solution**:

```rust
// Aggressive pruning after each reflection
async fn prune_playbook(&self) -> Result<usize> {
    let criteria = PruneConfig {
        min_confidence: 0.45,  // Up from 0.40
        max_staleness_days: 14, // Down from 30
        min_effectiveness: 0.55, // helpful/(helpful+harmful)
        max_bullets_per_section: 20, // Hard cap
    };

    // Also deduplicate similar bullets
    let similar_threshold = 0.90; // cosine similarity

    self.deduplicate_and_prune(criteria).await
}
```

**Acceptance Criteria**:

- [ ] Playbook stays under 150 bullets total
- [ ] Bullets with confidence < 0.45 pruned after 14 days
- [ ] Duplicate/similar bullets merged
- [ ] Log pruning statistics each evening

---

## 🟡 MEDIUM PRIORITY (P2) - Improvements

### 9. Improve Vector Search Quality

**Location**: `src/vector/store.rs`

**Problem**:

- Finding contexts with only 0.25 similarity (very low)
- All "similar" contexts nearly identical
- Suggests poor embeddings or search parameters

**Solutions**:

- [ ] Improve market state representation for embedding:

```rust
fn market_state_to_embedding_text(state: &MarketState) -> String {
    format!(
        "Market: {} trend, VIX {:.1}, volume {}, \
         sentiment {}, regime {}, signals: {}",
        state.trend, state.vix, state.volume_label,
        state.sentiment_label, state.regime, state.key_signals
    )
}
```

- [ ] Tune HNSW index parameters (ef_construction, M)
- [ ] Lower similarity threshold or use MMR for diversity

---

### 10. Add Stop-Loss and Take-Profit Logic

**Location**: `src/trading/paper.rs`

**Current**: Trades held until next day regardless of movement

**Implement**:

- [ ] Stop-loss: -15% from entry (options) / -2% (shares)
- [ ] Take-profit: +50% (options) / +5% (shares)
- [ ] Trailing stop: After +30%, trail by 10%
- [ ] Time-based exit: Close at 3:50 PM if same-day trade

---

### 11. Enhance Decision Logging

**Location**: `src/orchestrator/morning.rs`

**Add Structured Metadata**:

```rust
info!(
    decision = %decision.action,
    confidence = %decision.confidence,
    playbook_entries_count = %playbook_entries.len(),
    similar_contexts_count = %similar_contexts.len(),
    avg_similarity = %avg_similarity,
    risk_factors = ?risk_factors,
    "Generated trading decision"
);
```

---

### 12. Implement Confidence Bounds Testing

**Location**: `tests/integration/`

**Create Test Suite**:

- [ ] Test: High confidence + loss → future confidence drops
- [ ] Test: Low confidence + win → future confidence rises
- [ ] Test: Confidence never exceeds historical win rate + 10%
- [ ] Test: 5 consecutive losses → confidence < 60%

---

### 14. Optimize Prompt Token Usage

**Location**: `src/ace/prompts.rs`

**Current**: Including all recent bullets (could be 20+)

**Optimization**:

- [ ] Semantic search for most relevant bullets to current state
- [ ] Limit to top 10 most relevant per section
- [ ] Summarize very similar bullets
- [ ] Use LLM to compress playbook if > 2000 tokens

---

### 15. Add Backtest Comparison Tool

**New File**: `src/cli/commands/compare_backtests.rs`

**Features**:

```bash
traderjoe compare-backtests \
  --baseline backtest_no_ace.json \
  --treatment backtest_with_ace.json
```

Output:

- Win rate comparison
- Sharpe ratio comparison
- Average P&L per trade
- Playbook growth rate
- Decision diversity metrics

---

## 🎯 Success Criteria (from ace_framework.md)

### Must Achieve Before Production

- [ ] **Win Rate > 55%** (Currently: 0%)
- [ ] **Sharpe Ratio > 1.5** (Currently: N/A)
- [ ] **Max Drawdown < 15%** (Currently: 40% per trade)
- [ ] **Playbook Contains Non-Obvious Insights** (Generated but not used)
- [ ] **System Learns from Mistakes** (Identifies issues but doesn't adapt)

### Must Demonstrate

- [ ] Playbook retrieval working in backtests
- [ ] Decisions change based on playbook content
- [ ] Confidence calibrates with performance
- [ ] Risk factors properly weighted
- [ ] No repeated identical mistakes

---

## 📝 Implementation Notes

### Backtest vs Live Workflow

- **Backtest**: Uses historical dates (2024-09-02 to 2024-10-02)
  - `get_recent_bullets(30, backtest_date)` ← pass backtest date
  - All time-based queries use backtest date as reference

- **Live**: Uses current dates (2025-10-28 onwards)
  - `get_recent_bullets(30, Utc::now())` ← use current time
  - Normal time-based operations

### Key Insight from Testing

The infrastructure works perfectly - data flows, trades execute, reflections generate insights, playbook updates persist. The **only** problem is the learned knowledge isn't being retrieved and applied. Once fixed, the system should show true learning behavior.

---

## 🔗 Related Documentation

- `docs/ace_framework.md` - ACE methodology and principles
- `docs/PRD.md` - Product requirements and architecture
- `src/ace/playbook.rs` - Playbook data structures (lines 62-109)
- `src/orchestrator/morning.rs` - Decision generation (lines 404-429)
