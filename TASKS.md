# TraderJoe - ACE Framework Production Tasks

Based on Anthropic's "Advancing Claude for Financial Services" (2025-10-28)

### Data Connector Enhancements

#### 16. Add Aiera Integration - Earnings Transcripts

**Priority**: HIGH - Significant value for SPY/QQQ holdings
**Location**: New file `src/data/aiera.rs`

**Features**:

- Earnings call transcripts for SPY/QQQ top holdings
- Investor event summaries
- Extract key metrics and sentiment from management commentary
- Feed into ACE playbook as "earnings insights"

**Implementation**:

```rust
pub struct AieraClient {
    // Fetch earnings transcripts
    async fn fetch_earnings_transcript(symbol: &str, date: NaiveDate) -> Result<Transcript>

    // Summarize key points for ACE consumption
    async fn extract_earnings_insights(transcript: &Transcript) -> Result<Vec<String>>
}
```

**Integration Points**:

- `src/orchestrator/morning.rs` - Check upcoming earnings before trading
- `src/orchestrator/evening.rs` - Analyze post-earnings price action
- `src/ace/curator.rs` - Add "earnings_insights" section to playbook

**Acceptance Criteria**:

- [ ] Fetch transcripts for SPY top 10 holdings
- [ ] Extract bullish/bearish sentiment
- [ ] Track earnings surprise impact on trading decisions
- [ ] Add to morning analysis: "Upcoming earnings: AAPL (2 days), MSFT (5 days)"

---

#### 17. Add LSEG Integration - Macroeconomic Indicators

**Priority**: HIGH - Critical for market regime detection
**Location**: `src/data/lseg.rs`

**Features**:

- Live market data (complement Polygon.io)
- Fixed income pricing (bond yields for risk-off signals)
- Forex rates (USD strength as market factor)
- Macroeconomic indicators (GDP, employment, CPI)

**Data Points to Track**:

- 10-year Treasury yield (risk sentiment)
- DXY (US Dollar Index) - affects SPY/QQQ
- Credit spreads (corporate vs Treasury)
- Economic calendar (FOMC, CPI, jobs reports)

**Implementation**:

```rust
pub struct LSEGClient {
    async fn fetch_treasury_yields() -> Result<TreasuryData>
    async fn fetch_dollar_index() -> Result<f64>
    async fn fetch_economic_calendar(days_ahead: u32) -> Result<Vec<EconomicEvent>>
}
```

**Integration Points**:

- `src/orchestrator/morning.rs` - Include macro regime in decision context
- `src/ace/prompts.rs` - Add macro context to prompt template
- Add playbook section: "macro_regime_patterns"

**Acceptance Criteria**:

- [ ] Fetch 10Y yield daily
- [ ] Track DXY strength/weakness
- [ ] Alert on major economic events (FOMC, CPI)
- [ ] Regime detection: "risk-on" vs "risk-off" based on yields + VIX

---

#### 18. Add MT Newswires Integration

**Priority**: MEDIUM - Upgrade from generic NewsAPI
**Location**: `src/data/news.rs` (refactor existing)

**Features**:

- Global financial market news (more targeted than NewsAPI)
- Real-time market-moving events
- Sector-specific news for SPY/QQQ components
- Breaking news alerts

**Implementation**:

- Replace or complement existing NewsAPI integration
- Focus on high-signal, low-noise financial news
- Filter by relevance to SPY/QQQ

**Acceptance Criteria**:

- [ ] Fetch top 20 market news items daily
- [ ] Extract sentiment and tickers mentioned
- [ ] Track correlation between news sentiment and VIX spikes
- [ ] Add to playbook: "News pattern: Fed hawkish → VIX up → avoid longs"

---

#### 19. Add Moody's Integration - Credit Ratings

**Priority**: LOW - Deep fundamental analysis (nice-to-have)
**Location**: `src/data/moodys.rs`

**Features**:

- Credit ratings for SPY/QQQ holdings
- Company intelligence on 600+ million entities
- Track rating changes (upgrades/downgrades)
- Credit risk as factor in trading decisions

**Use Cases**:

- Avoid trading during rating downgrades
- Sector rotation based on credit quality
- Risk-off signal: credit spread widening

**Implementation**:

```rust
pub struct MoodysClient {
    async fn fetch_credit_rating(symbol: &str) -> Result<CreditRating>
    async fn fetch_recent_rating_changes() -> Result<Vec<RatingChange>>
}
```

**Acceptance Criteria**:

- [ ] Track credit ratings for SPY top 50 holdings
- [ ] Alert on rating downgrades
- [ ] Add to risk assessment: "AAPL: AAA credit, low default risk"

---

### Financial Agent Skills - New Analysis Modules

#### 20. Implement DCF Valuation Module

**Priority**: MEDIUM - Fundamental analysis layer
**Location**: New file `src/analysis/valuation.rs`

**Features**:

- Discounted Cash Flow (DCF) models with WACC calculations
- Valuation multiples (P/E, P/S, EV/EBITDA)
- Compare current price vs intrinsic value
- Feed into confidence scoring: "SPY overvalued by 12% → reduce confidence"

**Implementation**:

```rust
pub struct ValuationAnalyzer {
    async fn calculate_dcf(symbol: &str) -> Result<DCFModel>
    async fn get_valuation_multiples(symbol: &str) -> Result<ValuationMultiples>

    // Compare to sector averages
    async fn relative_valuation(symbol: &str, sector: &str) -> Result<RelativeValuation>
}
```

**Integration Points**:

- `src/orchestrator/morning.rs` - Add valuation check before trades
- `src/ace/curator.rs` - Learn patterns: "When P/E > 25 + VIX > 20 → avoid"
- Add playbook section: "valuation_patterns"

**Acceptance Criteria**:

- [ ] Calculate simple DCF for SPY/QQQ
- [ ] Track P/E ratio vs 5-year average
- [ ] Adjust confidence: overvalued + high VIX → reduce 10%
- [ ] Playbook learns: "Overvalued markets correct faster in high VIX"

---

#### 21. Implement Comparable Company Analysis

**Priority**: MEDIUM - Sector rotation insights
**Location**: `src/analysis/comps.rs`

**Features**:

- Find comparable companies for SPY/QQQ holdings
- Compare valuation multiples
- Identify overvalued vs undervalued sectors
- Sector rotation signals

**Implementation**:

```rust
pub struct CompsAnalyzer {
    async fn find_comps(symbol: &str, count: usize) -> Result<Vec<Comparable>>
    async fn sector_valuation_analysis(sector: &str) -> Result<SectorMetrics>
}
```

**Use Cases**:

- "Tech sector trading at 30x P/E vs 5-year avg of 22x → overvalued"
- "Financials cheapest in 3 years → potential rotation"

**Acceptance Criteria**:

- [ ] Compare tech vs financials vs healthcare valuations
- [ ] Track sector rotation patterns
- [ ] Add to playbook: "When tech overvalued → reduce QQQ exposure"

---

#### 22. Implement Earnings Analysis Module

**Priority**: HIGH (if Aiera integrated)
**Location**: `src/analysis/earnings.rs`

**Features**:

- Extract key metrics from earnings transcripts (via Aiera)
- Track earnings surprises (beat/miss/in-line)
- Analyze guidance changes
- Post-earnings momentum patterns

**Implementation**:

```rust
pub struct EarningsAnalyzer {
    async fn analyze_earnings(symbol: &str, transcript: &Transcript) -> Result<EarningsInsights>

    // Track patterns: "Beat + guide up = 3-day pop 80% of time"
    async fn extract_earnings_patterns() -> Result<Vec<String>>
}

pub struct EarningsInsights {
    sentiment: EarningsSentiment,  // Bullish/Bearish/Neutral
    surprise: f64,  // Beat by X%
    guidance: GuidanceChange,  // Raised/Lowered/Maintained
    management_tone: String,
}
```

**Integration Points**:

- `src/orchestrator/morning.rs` - Avoid trading on earnings day unless high confidence
- `src/ace/curator.rs` - Learn: "AAPL post-earnings dips = buy opportunity"

**Acceptance Criteria**:

- [ ] Parse earnings transcripts for sentiment
- [ ] Track beat/miss patterns
- [ ] Avoid trading 1 day before earnings
- [ ] Exploit post-earnings momentum (if pattern learned)

---

#### 23. Implement Coverage Report Generator

**Priority**: LOW - Analysis tool (not trading-critical)
**Location**: `src/analysis/coverage.rs`

**Features**:

- Generate analyst-style coverage reports
- Industry analysis and competitive positioning
- Investment thesis (bull/bear cases)
- Price target estimation

**Use Cases**:

- Deep-dive analysis on SPY/QQQ top holdings
- Research reports for portfolio review
- Documentation of trading thesis

**Implementation**:

```rust
pub struct CoverageReportGenerator {
    async fn generate_report(symbol: &str) -> Result<CoverageReport>
}

pub struct CoverageReport {
    industry_overview: String,
    competitive_position: String,
    bull_case: Vec<String>,
    bear_case: Vec<String>,
    valuation_summary: String,
    recommendation: String,  // Buy/Hold/Sell
}
```

**Acceptance Criteria**:

- [ ] Generate PDF report for a symbol
- [ ] Include DCF, comps, earnings analysis
- [ ] CLI command: `traderjoe coverage AAPL`

---

### Strategic Enhancements

#### 24. Implement Earnings Season Mode

**Priority**: HIGH - High-impact trading strategy
**Location**: `src/orchestrator/earnings_mode.rs`

**Features**:

- Track earnings calendar for SPY/QQQ holdings
- Pre-earnings risk reduction (close positions 2 days before)
- Post-earnings opportunity detection (beat + guide up = entry signal)
- Earnings season volatility adjustment

**Implementation**:

```rust
pub struct EarningsSeasonOrchestrator {
    // Check upcoming earnings
    async fn get_upcoming_earnings(days_ahead: u32) -> Result<Vec<EarningsEvent>>

    // Adjust trading based on earnings proximity
    async fn adjust_for_earnings(
        symbol: &str,
        decision: &TradingDecision,
        days_until_earnings: u32
    ) -> Result<TradingDecision>
}
```

**Trading Rules**:

- **2+ days before earnings**: Reduce position size by 50%
- **1 day before earnings**: No new positions
- **Earnings day**: Close all positions in that symbol
- **1 day after earnings**: If beat + guide up → enter long (if other signals align)
- **Learn patterns**: Track which stocks pop vs fade after earnings

**Integration Points**:

- `src/orchestrator/morning.rs` - Check earnings calendar
- `src/data/aiera.rs` - Fetch earnings calendar
- `src/ace/curator.rs` - Learn: "AAPL beats 8/10 times → pre-earnings bullish"

**Acceptance Criteria**:

- [ ] Fetch earnings calendar 7 days ahead
- [ ] Auto-close positions before earnings
- [ ] Track post-earnings momentum (80% accuracy target)
- [ ] Playbook section: "earnings_momentum_patterns"

---

#### 25. Add Fundamental Analysis Layer

**Priority**: MEDIUM - Combine with technical analysis
**Location**: `src/analysis/fundamental.rs`

**Combines**:

- DCF valuation (Task #20)
- Comparable company analysis (Task #21)
- Credit ratings (Task #19)
- Macroeconomic regime (Task #17)

**Output**: Fundamental Score (0-100)

- 80-100: Strong fundamentals → increase confidence
- 60-79: Neutral → no adjustment
- 40-59: Weak → reduce confidence 10%
- 0-39: Very weak → avoid trading

**Implementation**:

```rust
pub struct FundamentalAnalyzer {
    async fn calculate_fundamental_score(symbol: &str) -> Result<FundamentalScore>
}

pub struct FundamentalScore {
    valuation_score: f32,     // DCF vs current price
    credit_score: f32,        // Credit rating quality
    macro_score: f32,         // Macro regime alignment
    earnings_score: f32,      // Earnings quality/momentum
    total_score: f32,
    interpretation: String,
}
```

**Integration**:

- Combine with existing technical signals
- Weight: 60% technical + 40% fundamental
- Add to morning prompt: "Fundamental score: 78/100 (fairly valued, strong credit)"

**Acceptance Criteria**:

- [ ] Calculate fundamental score for SPY/QQQ
- [ ] Combine with technical signals
- [ ] Backtest: does fundamental filter improve win rate?
- [ ] Target: +5% win rate improvement from fundamental filter

---

### Implementation Roadmap for Future Enhancements

**Phase A: High-Value Data (Month 1)**

- Task #16: Aiera earnings transcripts
- Task #17: LSEG macro indicators
- Task #22: Earnings analysis module
- Task #24: Earnings season mode

**Expected Impact**: +10-15% win rate improvement from earnings intelligence

---

**Phase B: Fundamental Analysis (Month 2)**

- Task #20: DCF valuation module
- Task #21: Comparable company analysis
- Task #25: Fundamental analysis layer

**Expected Impact**: Better risk-adjusted returns, fewer trades in overvalued markets

---

**Phase C: News & Credit (Month 3)**

- Task #18: MT Newswires integration
- Task #19: Moody's credit ratings

**Expected Impact**: Improved risk detection, sector rotation signals

---

**Phase D: Analysis Tools (Month 4)**

- Task #23: Coverage report generator

**Expected Impact**: Better documentation, research capabilities

---

## 🔧 Technical Implementation Details - Task #1 Fix (2025-10-28)

### Problem Analysis

**Initial Hypothesis**: `get_playbook_entries()` was using `Utc::now()` instead of backtest date.

**Reality**: The retrieval logic was CORRECT - it was already using `backtest_date` from config. The actual bug was in bullet **creation**, not retrieval.

**Root Cause Discovery**:
Running 4-day backtest (2024-09-02 to 2024-09-06) showed:

```
Day 1: Retrieved 0 playbook entries (expected - no bullets yet)
Day 2: Retrieved 0 playbook entries (BUG - should have ~80 from Day 1)
Day 3: Retrieved 0 playbook entries (BUG - should have bullets from Days 1-2)
Day 4: Retrieved 0 playbook entries (BUG)
Day 5: Retrieved 0 playbook entries (BUG)
```

Investigation revealed:

- Day 1 evening creates ~80 playbook bullets
- Database INSERT uses `DEFAULT NOW()` → bullets get `created_at = 2025-10-28` (current wall-clock time)
- Day 2 morning queries: `WHERE created_at >= 2024-08-03` (30 days before 2024-09-02)
- Finds 0 bullets because `2025-10-28 >= 2024-08-03` is false from 2024-09-02's perspective
- Bullets exist in "future" relative to backtest timeline

### Solution Implementation

**Architecture**: Thread `created_at` timestamp through entire bullet creation pipeline

#### 1. PlaybookDAO (`src/ace/playbook.rs`)

```rust
pub async fn insert_bullet(
    &self,
    section: PlaybookSection,
    content: String,
    source_context_id: Option<Uuid>,
    meta: Option<serde_json::Value>,
    created_at: Option<DateTime<Utc>>,  // NEW parameter
) -> Result<Uuid>
```

**Implementation**:

- If `created_at.is_some()`: Use explicit timestamp (backtest mode)
- If `created_at.is_none()`: Use `DEFAULT NOW()` (live mode)
- Logs timestamp source for debugging

#### 2. DeltaEngine (`src/ace/delta.rs`)

```rust
pub struct DeltaEngine {
    playbook_dao: PlaybookDAO,
    embedder: EmbeddingGemma,
    config: DeltaEngineConfig,
    created_at: Option<DateTime<Utc>>,  // NEW field
}

pub async fn new(
    playbook_dao: PlaybookDAO,
    config: Option<DeltaEngineConfig>,
    created_at: Option<DateTime<Utc>>,  // NEW parameter
) -> Result<Self>
```

**Changes**:

- Stores `created_at` in struct
- Passes to `insert_bullet()` when applying Add deltas
- All bullets in a batch use same timestamp (consistent snapshot)

#### 3. Curator (`src/ace/curator.rs`)

```rust
pub async fn new(
    playbook_dao: PlaybookDAO,
    llm_client: LLMClient,
    config: Option<CuratorConfig>,
    delta_config: Option<DeltaEngineConfig>,
    created_at: Option<DateTime<Utc>>,  // NEW parameter
) -> Result<Self>
```

**Changes**:

- Accepts `created_at` parameter
- Passes to DeltaEngine constructor
- No other changes needed (just threading)

#### 4. EveningOrchestrator (`src/orchestrator/evening.rs`)

```rust
pub async fn new(pool: PgPool, config: Config) -> Result<Self> {
    // Extract backtest date from config
    let created_at = config.backtest_date.map(|date| {
        date.and_hms_opt(0, 0, 0)
            .expect("Invalid time")
            .and_utc()
    });

    // Pass to Curator
    let curator = Curator::new(
        playbook_dao.clone(),
        llm_client,
        None,
        None,
        created_at,  // NEW parameter
    ).await?;
}
```

**Logic**:

- In backtest mode: `config.backtest_date` is `Some(2024-09-02)` → bullets use that date
- In live mode: `config.backtest_date` is `None` → bullets use `NOW()`

### Test Coverage

#### Unit Test: Time-Travel Queries (`tests/orchestration_integration.rs:587-731`)

```rust
#[tokio::test]
async fn test_playbook_retrieval_at_different_reference_dates()
```

**Tests**:

1. Creates bullets at T-40d, T-20d, T-10d
2. Queries from "now" perspective → finds bullets within window
3. Queries from T-35d perspective → finds only T-40d bullet
4. Queries from T-15d perspective → finds T-40d and T-20d bullets
5. Simulates backtest scenario with historical reference date

**Purpose**: Verifies `get_recent_bullets()` correctly filters by reference date.

#### Integration Test: Ablation Study (`tests/orchestration_integration.rs:733-883`)

```rust
#[tokio::test]
async fn test_playbook_ablation_empty_vs_populated()
```

**Tests**:

1. Generate decision with empty playbook (baseline)
2. Add 4 strong risk-focused bullets
3. Generate decision with populated playbook
4. Verify measurable difference in: action, confidence, reasoning, or concepts mentioned

**Purpose**: Confirms playbook entries actually influence LLM decisions.

### Expected Behavior After Fix

**Backtest Timeline**:

```
Day 1 (2024-09-02):
  Morning: Retrieved 0 entries (no history yet)
  Evening: Creates 80 bullets with created_at = 2024-09-02

Day 2 (2024-09-03):
  Morning: Retrieved 80 entries (created 1 day ago) ✅
  Evening: Updates existing bullets, adds new ones with created_at = 2024-09-03

Day 3 (2024-09-04):
  Morning: Retrieved 120 entries (created in last 2 days) ✅
  ...continues
```

**Live Mode**:

- All bullets use `NOW()` as before
- No behavioral change
- Backwards compatible

### Verification Steps

Once SQLX cache updated and build succeeds:

1. Re-run 4-day backtest
2. Check logs for: `"Retrieved X playbook entries (as of 2024-09-XX)"` where X > 0
3. Verify bullet `created_at` timestamps match backtest dates (query database)
4. Confirm decisions show influence from learned patterns
5. Run unit tests: `cargo test test_playbook_retrieval_at_different_reference_dates`

---

## 📚 References

- [Anthropic: Advancing Claude for Financial Services](https://www.anthropic.com/news/advancing-claude-for-financial-services) (2025-10-28)
- Claude Sonnet 4.5: 55.3% accuracy on Finance Agent benchmark (Vals AI)
- New financial agent skills: DCF, valuations, earnings analysis, coverage reports
## 🔗 Related Documentation
- `docs/ace_framework.md` - ACE methodology and principles
- `docs/PRD.md` - Product requirements and architecture
- `src/ace/playbook.rs` - Playbook data structures (lines 62-109)
- `src/orchestrator/morning.rs` - Decision generation (lines 404-429)