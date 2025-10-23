# TraderJoe: Task List & Phase 4 Roadmap

**Last Updated**: 2025-10-22
**Current Phase**: Phase 3 Complete → Phase 4 Preparation
**Goal**: 90-Day Paper Trading Validation

---

## REMAINING CodeRabbit Issues (Prioritized)

#### circuit_breaker.rs

**1. Semantic mismatch in manual_halt notes parameter**

- Current: Notes parameter used as `triggered_by` field
- Fix: Separate triggered_by ("Manual halt") from user notes
- Priority: LOW

#### circuit_breaker_concurrency.rs (Test Issues)

**1-5. Test isolation issues**

- Tests lack cleanup and share database state
- Fix: Use transactions with rollback OR run serially
- Priority: LOW (testing improvement)

#### context.rs

**1. ingested_at fallback creates inconsistent timestamps**

- Uses `unwrap_or_else(Utc::now)` which generates new timestamp on each fetch
- Fix: Add NOT NULL constraint or use fixed fallback
- Priority: LOW

#### pgvector_sqlx.rs

**1. Use official pgvector crate**

- Current manual encoding works but less efficient
- Priority: LOW (optimization)

#### evening.rs

**1. Symbol defaulting may mask errors**

- Defaults to "SPY" if symbol missing
- Fix: Return error instead
- Priority: LOW

---

## 📋 PHASE 4: PAPER TRADING PREPARATION

**Goal**: Complete all functionality needed for 90-day paper trading validation
**Timeline**: 2-3 weeks before starting validation

### PT-1: Complete Paper Trading CLI Commands

- [ ] Implement `execute` command in `src/cli/commands.rs`
  - [ ] Accept recommendation from morning analysis
  - [ ] Validate market is open
  - [ ] Check circuit breaker status
  - [ ] Execute paper trade via `PaperTradingEngine`
  - [ ] Log trade in `trade_journal` table
- [ ] Implement `positions` command to view open trades
- [ ] Implement `close` command to manually exit position
- [ ] Add `performance` command for daily/weekly stats:
  - [ ] Win rate, average win/loss
  - [ ] Sharpe ratio calculation
  - [ ] Max drawdown tracking
  - [ ] Profit factor
- [ ] Add `--dry-run` flag for testing without DB writes

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 6 hours
**Files**: `src/cli/commands.rs`, `src/cli/mod.rs`

---

### PT-2: Automated Daily Workflow

- [ ] Create shell scripts for automation:
  - [ ] `scripts/morning_routine.sh` - Run morning analysis at 9:00 AM
  - [ ] `scripts/evening_routine.sh` - Run evening review at 4:30 PM
  - [ ] `scripts/auto_exit.sh` - Close positions at 3:00 PM if still open
- [ ] Set up cron jobs (via `scripts/setup_cron.sh`):

  ```bash
  0 9 * * 1-5 /path/to/morning_routine.sh
  30 16 * * 1-5 /path/to/evening_routine.sh
  0 15 * * 1-5 /path/to/auto_exit.sh
  ```

- [ ] Add error notifications:
  - [ ] Email on circuit breaker trigger
  - [ ] Email on LLM failures
  - [ ] Email on data fetch failures
- [ ] Implement retry logic in scripts (max 3 retries, 5 min apart)
- [ ] Add logging to `~/traderjoe/logs/`

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 4 hours
**Files**: New `scripts/*.sh`

---

### PT-3: Auto-Exit System

- [ ] Implement auto-exit in `src/trading/auto_exit.rs`
- [ ] Add exit strategies:
  - [ ] Time-based: Exit all positions at 3:00 PM ET
  - [ ] Stop-loss: Exit if loss > 50% of position
  - [ ] Take-profit: Exit if gain > 50% of position
  - [ ] Trailing stop: Exit if price drops 20% from peak
- [ ] Integrate with `PaperTradingEngine::exit_trade()`
- [ ] Log exit reasons in `paper_trades.exit_reason`
- [ ] Add `--exit-time` CLI flag to customize (default 3:00 PM)

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 4 hours
**Files**: `src/trading/auto_exit.rs`, `src/cli/commands.rs`

---

### PT-4: Position Sizing Implementation

- [ ] Implement Kelly Criterion in `src/trading/position_sizing.rs`
- [ ] Add position size calculation:

  ```rust
  fn calculate_position_size(
      account_balance: f64,
      win_rate: f64,
      avg_win: f64,
      avg_loss: f64,
      confidence: f64,
      max_position_pct: f64, // 5% default
  ) -> f64
  ```

- [ ] Apply 1/4 Kelly for safety
- [ ] Scale by ACE confidence (confidence * base_size)
- [ ] Enforce hard limits:
  - [ ] Max 5% of account per trade
  - [ ] Max 1 open position at a time (initially)
  - [ ] Max 3 trades per day
- [ ] Add tests for edge cases (100% confidence, 0% confidence, etc.)

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 3 hours
**Files**: `src/trading/position_sizing.rs`

---

### PT-5: Performance Metrics & Reporting

- [ ] Implement metrics calculation in `src/trading/account.rs`
- [ ] Add functions:
  - [ ] `calculate_sharpe_ratio(returns: Vec<f64>) -> f64`
  - [ ] `calculate_max_drawdown(equity_curve: Vec<f64>) -> f64`
  - [ ] `calculate_win_rate(trades: Vec<PaperTrade>) -> f64`
  - [ ] `calculate_profit_factor(trades: Vec<PaperTrade>) -> f64`
  - [ ] `calculate_expectancy(trades: Vec<PaperTrade>) -> f64`
- [ ] Create daily performance report:
  - [ ] Text summary logged to console
  - [ ] JSON report saved to `reports/daily_YYYY-MM-DD.json`
- [ ] Create weekly deep analysis:
  - [ ] Pattern performance breakdown
  - [ ] ML model accuracy tracking
  - [ ] ACE confidence calibration
  - [ ] Trade distribution by action type

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 5 hours
**Files**: `src/trading/account.rs`, `src/cli/commands.rs`

---

### PT-6: Data Quality Checks

- [ ] Implement validation in `src/data/mod.rs`
- [ ] Add checks before using data for decisions:
  - [ ] Timestamps are recent (< 15 minutes old)
  - [ ] No missing required fields
  - [ ] Price ranges are valid (SPY > $0, VIX > 0)
  - [ ] News/sentiment data not empty
- [ ] Add cross-source validation:
  - [ ] Compare Yahoo Finance vs. broker API prices
  - [ ] Detect stale data (same price for > 5 minutes during market hours)
- [ ] Halt trading on data quality failures:
  - [ ] Trigger circuit breaker with `SystemError` reason
  - [ ] Log detailed error for debugging
- [ ] Add `--skip-data-checks` flag for testing (dangerous!)

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 4 hours
**Files**: New `src/data/validation.rs`

---

### PT-7: Comprehensive Logging System

- [ ] Configure `tracing-subscriber` with multiple outputs:
  - [ ] Console (INFO level) for interactive use
  - [ ] File (DEBUG level) in `logs/traderjoe_YYYY-MM-DD.log`
  - [ ] JSON (all levels) in `logs/structured_YYYY-MM-DD.json` for analysis
- [ ] Add structured logging spans:

  ```rust
  #[instrument(skip(self))]
  async fn morning_analysis() -> Result<Recommendation> {
      info!("Starting morning analysis");
      // ...
  }
  ```

- [ ] Log all trading decisions with context
- [ ] Implement log rotation (keep 90 days, max 1GB total)
- [ ] Add `logs/` to `.gitignore`

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 3 hours
**Files**: `src/main.rs`, `src/lib.rs`

---

### PT-8: ACE Context Pruning

- [ ] Implement context cleanup in `src/ace/context.rs`
- [ ] Add pruning strategies:
  - [ ] Keep all contexts from last 30 days
  - [ ] Keep top 10 best-performing patterns (by win rate)
  - [ ] Keep top 10 worst-performing patterns (for failure analysis)
  - [ ] Delete contexts with similarity < 0.3 to any recent context
- [ ] Add `prune_old_contexts(days: i32)` function
- [ ] Run pruning weekly via cron job
- [ ] Archive pruned contexts to `archive/` table for later analysis
- [ ] Add `--prune-contexts` CLI command

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 3 hours
**Files**: `src/ace/context.rs`, `migrations/007_archive_contexts.sql`

---

### PT-9: Hypothesis Testing Framework

- [ ] Create `src/analysis/hypothesis.rs` module
- [ ] Implement hypothesis tracking:

  ```rust
  struct Hypothesis {
      id: String,
      description: String,
      test_start_date: NaiveDate,
      test_duration_days: i32,
      criteria: HypothesisCriteria,
      results: Vec<HypothesisResult>,
      conclusion: Option<String>,
  }
  ```

- [ ] Add pre-defined hypotheses from PRD:
  - H1: "VIX < 15 + green futures = 60%+ win rate on calls"
  - H2: "ACE improves win rate by 10%+ vs. baseline ML"
  - H3: "Sentiment divergence predicts reversals"
- [ ] Track hypothesis results daily
- [ ] Generate hypothesis report after 30/60/90 days
- [ ] Add `hypothesis` command to CLI

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 5 hours
**Files**: New `src/analysis/hypothesis.rs`, `migrations/008_hypotheses.sql`

---

### PT-10: Baseline Strategy Implementation

- [ ] Implement simple baseline in `src/trading/baseline.rs`
- [ ] Create "Technical-Only" strategy:
  - [ ] Use only RSI, MACD, SMA signals
  - [ ] No ACE context or LLM reasoning
  - [ ] Fixed position sizing (3% of account)
- [ ] Run baseline in parallel with ACE strategy (paper trading only)
- [ ] Compare performance weekly:
  - [ ] ACE win rate vs. baseline win rate
  - [ ] ACE returns vs. baseline returns
  - [ ] ACE Sharpe vs. baseline Sharpe
- [ ] Track "ACE edge" = ACE performance - baseline performance

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 4 hours
**Files**: New `src/trading/baseline.rs`

---

### PT-11: Manual Override System

- [ ] Add manual intervention CLI commands:
  - [ ] `override --recommendation-id <id> --action STAY_FLAT` - Reject ACE recommendation
  - [ ] `manual-trade --action BUY_CALLS --size 100 --reason "gut feeling"` - Manual trade
  - [ ] `adjust-position --trade-id <id> --new-size 50` - Modify position size
- [ ] Log all manual interventions to `manual_overrides` table
- [ ] Track manual intervention performance separately
- [ ] Add weekly report: "Manual vs. ACE performance"
- [ ] Require `--confirm` flag for manual trades (prevent accidents)

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 3 hours
**Files**: `src/cli/commands.rs`, `migrations/009_manual_overrides.sql`

---

### PT-12: Stress Testing Suite

- [ ] Create `tests/stress_test.rs`
- [ ] Implement stress scenarios:
  - **Scenario 1**: VIX spike (15 → 40)
    - [ ] Verify ACE recommends STAY_FLAT
    - [ ] Verify position sizes reduced
  - **Scenario 2**: Data feed failure
    - [ ] Disable news API
    - [ ] Verify circuit breaker triggers
    - [ ] Verify graceful degradation
  - **Scenario 3**: Losing streak (10 consecutive losses)
    - [ ] Verify circuit breaker triggers at 5
    - [ ] Verify position sizes reduced
    - [ ] Verify ACE updates playbook with caution
  - **Scenario 4**: LLM timeout/failure
    - [ ] Verify fallback to OpenAI
    - [ ] Verify retry logic works
    - [ ] Verify trading halts if both fail
- [ ] Add `--stress-test` CLI command to run all scenarios
- [ ] Run stress tests weekly during paper trading

**Priority**: 🔵 P3 - PAPER TRADING
**Estimated Time**: 6 hours
**Files**: New `tests/stress_test.rs`

---

## 🎯 90-DAY VALIDATION MILESTONES

**Pre-Launch Checklist** (Day 0):

- [ ] All CRITICAL and HIGH-priority tasks complete
- [ ] Integration tests passing (>95% coverage on critical paths)
- [ ] Stress tests passing
- [ ] Cron jobs tested and working
- [ ] API keys secured and tested
- [ ] Database backed up
- [ ] Logs directory created with rotation configured
- [ ] Email notifications tested

**Week 1-2 Goals** (Days 1-14):

- [ ] System runs daily without manual intervention
- [ ] No circuit breaker false positives
- [ ] Data quality checks catch bad data
- [ ] ACE generates valid decisions
- [ ] Position sizing works correctly
- [ ] Auto-exit executes at 3:00 PM

**Week 3-4 Goals** (Days 15-30):

- [ ] First hypothesis test results available
- [ ] ACE playbook has 10+ patterns
- [ ] Win rate data statistically meaningful (>20 trades)
- [ ] Baseline strategy comparison working
- [ ] Weekly reports generated automatically

**Month 2 Goals** (Days 31-60):

- [ ] Win rate > 50% (target: >55%)
- [ ] ACE vs. baseline edge quantified
- [ ] Slippage model calibrated
- [ ] Playbook has 20+ validated patterns
- [ ] Max drawdown < 15%

**Month 3 Goals** (Days 61-90):

- [ ] Win rate > 55% sustained
- [ ] Sharpe ratio > 1.5
- [ ] ACE playbook contains non-obvious insights
- [ ] System autonomy: 85+ days without manual intervention
- [ ] All success criteria from PRD met or clear path to improvement

**End of 90 Days** (Day 90):

- [ ] Comprehensive performance report generated
- [ ] Decision: Proceed to live trading or iterate further
- [ ] If proceeding: Broker API integration begins (Phase 5)
- [ ] If iterating: Analyze failure modes and adjust strategy

---

## 📊 SUCCESS CRITERIA (FROM PRD)

### Phase 4 Success Metrics

- [ ] **Win rate > 55%** on directional calls
- [ ] **Sharpe ratio > 1.5**
- [ ] **Max drawdown < 15%**
- [ ] **ACE playbook** contains non-obvious, specific insights
- [ ] **System autonomy**: Runs 85+ days without manual intervention
- [ ] **Data quality**: No bad data caused trading decisions
- [ ] **Circuit breakers**: Triggered appropriately, no false positives
- [ ] **LLM reliability**: < 5% fallback rate to OpenAI

### Stretch Goals

- [ ] Match ACE paper performance (+10.6% over baseline)
- [ ] Playbook accumulates 50+ unique, validated patterns
- [ ] Zero manual interventions required after week 2
- [ ] Sharpe ratio > 2.0

---

## 📝 NOTES

**Task Prioritization**:

- **P0 (CRITICAL)**: Security vulnerabilities, blocking issues
- **P1 (HIGH)**: Required for safe paper trading
- **P2 (MEDIUM)**: Improves quality, can address during validation
- **P3 (PAPER TRADING)**: Core functionality for 90-day validation

**Estimated Total Time to Paper Trading Ready**:

- Critical tasks (SEC): ~10 hours
- High-priority tasks (HP): ~20 hours
- Paper trading prep (PT): ~60 hours
- **Total**: ~90 hours (~2-3 weeks at 6-8 hours/day)

**Risk Areas to Monitor**:

1. LLM output quality (prompt injection attempts)
2. Data quality (stale or missing data)
3. Circuit breaker tuning (false positives vs. true risk)
4. ACE playbook evolution (overfitting to recent patterns)

---

**Last Reviewed**: 2025-10-22
**Next Review**: After completing PT-1 and PT-2 tasks
