// Trade Execution with Realistic Slippage and Commission Modeling
// Simulates real-world trade execution with unfavorable fills

use anyhow::{bail, Result};
use chrono::{DateTime, Datelike, NaiveDate, NaiveTime, Utc};
use chrono_tz::America::New_York;

/// Execution parameters for realistic trade simulation
#[derive(Debug, Clone)]
pub struct ExecutionParams {
    /// Slippage percentage (e.g., 0.03 = 3%)
    pub slippage_pct: f64,

    /// Commission per contract/trade (default $0.65 for options)
    pub commission: f64,

    /// Simulated execution delay in milliseconds
    pub fill_time_delay_ms: u64,

    /// Market open time (ET) - default 9:30 AM
    pub market_open: NaiveTime,

    /// Market close time (ET) - default 4:00 PM
    pub market_close: NaiveTime,
}

/// Moneyness category for options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Moneyness {
    /// In-the-money (tighter spreads)
    ITM,
    /// At-the-money (moderate spreads)
    ATM,
    /// Out-of-the-money (wider spreads)
    OTM,
}

/// Parameters for dynamic slippage calculation
#[derive(Debug, Clone)]
pub struct DynamicSlippageParams {
    /// Current VIX level
    pub vix: f64,

    /// Option moneyness
    pub moneyness: Moneyness,
}

impl DynamicSlippageParams {
    /// Create dynamic slippage params from VIX and moneyness
    pub fn new(vix: f64, moneyness: Moneyness) -> Self {
        Self { vix, moneyness }
    }
}

impl Default for ExecutionParams {
    fn default() -> Self {
        Self {
            slippage_pct: 0.03, // 3% slippage for options
            commission: 0.65,   // $0.65 per contract
            fill_time_delay_ms: 500,
            market_open: NaiveTime::from_hms_opt(9, 30, 0)
                .expect("Invalid hardcoded time 9:30:00 - this is a bug"),
            market_close: NaiveTime::from_hms_opt(16, 0, 0)
                .expect("Invalid hardcoded time 16:00:00 - this is a bug"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Calculate dynamic slippage percentage based on VIX and moneyness
///
/// VIX-based slippage:
/// - VIX < 15: 2% slippage (low volatility)
/// - VIX 15-25: 3% slippage (normal volatility)
/// - VIX 25-35: 4% slippage (elevated volatility)
/// - VIX > 35: 5% slippage (high volatility)
///
/// Moneyness adjustments:
/// - ITM options: -0.5% (tighter spreads)
/// - ATM options: 0% (no adjustment)
/// - OTM options: +0.5% (wider spreads)
pub fn calculate_dynamic_slippage(params: &DynamicSlippageParams) -> f64 {
    // Base slippage from VIX
    let vix_slippage = if params.vix < 15.0 {
        0.02 // 2%
    } else if params.vix < 25.0 {
        0.03 // 3%
    } else if params.vix < 35.0 {
        0.04 // 4%
    } else {
        0.05 // 5%
    };

    // Moneyness adjustment
    let moneyness_adjustment = match params.moneyness {
        Moneyness::ITM => -0.005, // -0.5% for tighter spreads
        Moneyness::ATM => 0.0,    // No adjustment
        Moneyness::OTM => 0.005,  // +0.5% for wider spreads
    };

    // Ensure slippage stays positive
    f64::max(vix_slippage + moneyness_adjustment, 0.01)
}

/// Calculate the fill price with slippage applied in the unfavorable direction
///
/// For buys: market price + slippage (paying more)
/// For sells: market price - slippage (receiving less)
///
/// Returns 0.0 if market_price is not finite
pub fn calculate_fill_price(market_price: f64, side: OrderSide, params: &ExecutionParams) -> f64 {
    // Validate inputs
    if !market_price.is_finite() || market_price <= 0.0 {
        return 0.0;
    }

    let slippage = market_price * params.slippage_pct;

    match side {
        OrderSide::Buy => market_price + slippage,
        OrderSide::Sell => market_price - slippage,
    }
}

/// Calculate the fill price with dynamic slippage based on VIX and moneyness
///
/// For buys: market price + slippage (paying more)
/// For sells: market price - slippage (receiving less)
///
/// Returns 0.0 if market_price is not finite
pub fn calculate_fill_price_dynamic(
    market_price: f64,
    side: OrderSide,
    dynamic_params: &DynamicSlippageParams,
) -> f64 {
    // Validate inputs
    if !market_price.is_finite() || market_price <= 0.0 {
        return 0.0;
    }

    let slippage_pct = calculate_dynamic_slippage(dynamic_params);
    let slippage = market_price * slippage_pct;

    match side {
        OrderSide::Buy => market_price + slippage,
        OrderSide::Sell => market_price - slippage,
    }
}

/// Check if a given date is a NYSE/NASDAQ market holiday
///
/// 2025 Market Holidays (NYSE/NASDAQ):
/// - New Year's Day: January 1
/// - MLK Day: Third Monday in January (Jan 20)
/// - Presidents' Day: Third Monday in February (Feb 17)
/// - Good Friday: Friday before Easter (Apr 18)
/// - Memorial Day: Last Monday in May (May 26)
/// - Juneteenth: June 19
/// - Independence Day: July 4
/// - Labor Day: First Monday in September (Sep 1)
/// - Thanksgiving: Fourth Thursday in November (Nov 27)
/// - Christmas: December 25
pub fn is_market_holiday(date: NaiveDate) -> bool {
    let year = date.year();
    let month = date.month();
    let day = date.day();

    // Only 2025 holidays are hardcoded for now
    // TODO: Extend to future years or use a dynamic calendar
    if year != 2025 {
        return false;
    }

    match (month, day) {
        (1, 1) => true,   // New Year's Day
        (1, 20) => true,  // MLK Day 2025
        (2, 17) => true,  // Presidents' Day 2025
        (4, 18) => true,  // Good Friday 2025
        (5, 26) => true,  // Memorial Day 2025
        (6, 19) => true,  // Juneteenth
        (7, 4) => true,   // Independence Day
        (9, 1) => true,   // Labor Day 2025
        (11, 27) => true, // Thanksgiving 2025
        (12, 25) => true, // Christmas
        _ => false,
    }
}

/// Check if current time is within regular market hours
///
/// Regular hours: 9:30 AM - 4:00 PM ET (Monday-Friday, excluding holidays)
/// Properly handles timezone conversion with DST using chrono-tz
pub fn is_market_open(now: DateTime<Utc>, params: &ExecutionParams) -> bool {
    // Convert UTC to Eastern Time (handles DST automatically)
    let et_time = now.with_timezone(&New_York);
    let current_time = et_time.time();
    let current_date = et_time.date_naive();

    // Check if it's a weekend
    let weekday = et_time.weekday();
    if weekday == chrono::Weekday::Sat || weekday == chrono::Weekday::Sun {
        return false;
    }

    // Check if it's a market holiday
    if is_market_holiday(current_date) {
        return false;
    }

    // Check if within trading hours
    current_time >= params.market_open && current_time < params.market_close
}

/// Check if current time is during pre-market hours (9:00-9:30 AM ET)
pub fn is_pre_market(now: DateTime<Utc>) -> bool {
    let et_time = now.with_timezone(&New_York);
    let current_time = et_time.time();
    let pre_market_start =
        NaiveTime::from_hms_opt(9, 0, 0).expect("Invalid hardcoded time 9:00:00 - this is a bug");
    let market_open =
        NaiveTime::from_hms_opt(9, 30, 0).expect("Invalid hardcoded time 9:30:00 - this is a bug");

    current_time >= pre_market_start && current_time < market_open
}

/// Check if current time is during after-hours (4:00-8:00 PM ET)
pub fn is_after_hours(now: DateTime<Utc>) -> bool {
    let et_time = now.with_timezone(&New_York);
    let current_time = et_time.time();
    let market_close =
        NaiveTime::from_hms_opt(16, 0, 0).expect("Invalid hardcoded time 16:00:00 - this is a bug");
    let after_hours_end =
        NaiveTime::from_hms_opt(20, 0, 0).expect("Invalid hardcoded time 20:00:00 - this is a bug");

    current_time >= market_close && current_time < after_hours_end
}

/// Validate that a trade can be executed
pub fn validate_execution(
    market_price: f64,
    shares: f64,
    now: DateTime<Utc>,
    params: &ExecutionParams,
) -> Result<()> {
    // Check market hours
    if !is_market_open(now, params) {
        bail!(
            "Market is closed. Trading hours: {:?} - {:?} ET",
            params.market_open,
            params.market_close
        );
    }

    // Check price validity - must be finite and positive
    if !market_price.is_finite() {
        bail!("Invalid market price (NaN or infinity): {}", market_price);
    }
    if market_price <= 0.0 {
        bail!("Invalid market price: {}", market_price);
    }

    // Check shares validity - must be finite and positive
    if !shares.is_finite() {
        bail!("Invalid share count (NaN or infinity): {}", shares);
    }
    if shares <= 0.0 {
        bail!("Invalid share count: {}", shares);
    }

    Ok(())
}

/// Calculate the total cost including slippage and commission
pub fn calculate_total_cost(
    market_price: f64,
    shares: f64,
    side: OrderSide,
    params: &ExecutionParams,
) -> f64 {
    let fill_price = calculate_fill_price(market_price, side, params);
    let gross_cost = fill_price * shares;

    match side {
        OrderSide::Buy => gross_cost + params.commission,
        OrderSide::Sell => gross_cost - params.commission,
    }
}

/// Simulate bid-ask spread for options
///
/// Options typically have wider spreads than equities, especially for 0 DTE
/// This provides a more realistic fill price estimate
pub fn estimate_bid_ask_spread(market_price: f64, dte: i32) -> f64 {
    // Spread widens as expiration approaches
    let base_spread_pct = if dte == 0 {
        0.05 // 5% spread for 0 DTE
    } else if dte <= 7 {
        0.03 // 3% spread for weekly
    } else {
        0.02 // 2% spread for monthly+
    };

    market_price * base_spread_pct
}

/// Calculate the mid-price given bid and ask
pub fn calculate_mid_price(bid: f64, ask: f64) -> f64 {
    (bid + ask) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_fill_price_buy() {
        let params = ExecutionParams::default();
        let market_price = 100.0;
        let fill = calculate_fill_price(market_price, OrderSide::Buy, &params);

        // Should pay more than market price
        assert!(fill > market_price);
        assert_eq!(fill, 103.0); // 100 + 3% slippage
    }

    #[test]
    fn test_calculate_fill_price_sell() {
        let params = ExecutionParams::default();
        let market_price = 100.0;
        let fill = calculate_fill_price(market_price, OrderSide::Sell, &params);

        // Should receive less than market price
        assert!(fill < market_price);
        assert_eq!(fill, 97.0); // 100 - 3% slippage
    }

    #[test]
    fn test_total_cost_buy() {
        let params = ExecutionParams::default();
        let market_price = 2.0;
        let shares = 2.0; // 2 contracts

        let total = calculate_total_cost(market_price, shares, OrderSide::Buy, &params);

        // (2.0 * 1.03 * 2) + 0.65 = 4.12 + 0.65 = 4.77
        // Use approximate comparison for floating point
        assert!((total - 4.77).abs() < 0.01);
    }

    #[test]
    fn test_total_cost_sell() {
        let params = ExecutionParams::default();
        let market_price = 2.0;
        let shares = 2.0;

        let total = calculate_total_cost(market_price, shares, OrderSide::Sell, &params);

        // (2.0 * 0.97 * 2) - 0.65 = 3.88 - 0.65 = 3.23
        assert_eq!(total, 3.23);
    }

    #[test]
    fn test_bid_ask_spread() {
        let price = 2.0;

        let spread_0dte = estimate_bid_ask_spread(price, 0);
        assert_eq!(spread_0dte, 0.10); // 5% of 2.0

        let spread_weekly = estimate_bid_ask_spread(price, 5);
        assert_eq!(spread_weekly, 0.06); // 3% of 2.0

        let spread_monthly = estimate_bid_ask_spread(price, 30);
        assert_eq!(spread_monthly, 0.04); // 2% of 2.0
    }

    #[test]
    fn test_mid_price() {
        let bid = 2.0;
        let ask = 2.10;
        let mid = calculate_mid_price(bid, ask);
        assert_eq!(mid, 2.05);
    }

    #[test]
    fn test_validate_execution_invalid_price() {
        let params = ExecutionParams::default();
        // Use a time during market hours: Monday, Jan 6, 2025 at 10 AM ET
        let now = chrono::DateTime::parse_from_rfc3339("2025-01-06T15:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let result = validate_execution(-1.0, 1.0, now, &params);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid market price"));
    }

    #[test]
    fn test_validate_execution_invalid_shares() {
        let params = ExecutionParams::default();
        // Use a time during market hours: Monday, Jan 6, 2025 at 10 AM ET
        let now = chrono::DateTime::parse_from_rfc3339("2025-01-06T15:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let result = validate_execution(100.0, -1.0, now, &params);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid share count"));
    }

    #[test]
    fn test_is_market_holiday() {
        // Test 2025 holidays
        assert!(is_market_holiday(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap()
        )); // New Year's
        assert!(is_market_holiday(
            NaiveDate::from_ymd_opt(2025, 1, 20).unwrap()
        )); // MLK Day
        assert!(is_market_holiday(
            NaiveDate::from_ymd_opt(2025, 7, 4).unwrap()
        )); // July 4th
        assert!(is_market_holiday(
            NaiveDate::from_ymd_opt(2025, 12, 25).unwrap()
        )); // Christmas

        // Test non-holidays
        assert!(!is_market_holiday(
            NaiveDate::from_ymd_opt(2025, 1, 2).unwrap()
        ));
        assert!(!is_market_holiday(
            NaiveDate::from_ymd_opt(2025, 3, 15).unwrap()
        ));
    }

    #[test]
    fn test_market_open_weekend() {
        let params = ExecutionParams::default();

        // Saturday, January 4, 2025 at 10 AM ET (would be during market hours if it were a weekday)
        let saturday = chrono::DateTime::parse_from_rfc3339("2025-01-04T15:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        assert!(!is_market_open(saturday, &params));
    }

    #[test]
    fn test_market_open_holiday() {
        let params = ExecutionParams::default();

        // MLK Day 2025 (Monday, Jan 20) at 10 AM ET
        let mlk_day = chrono::DateTime::parse_from_rfc3339("2025-01-20T15:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        assert!(!is_market_open(mlk_day, &params));
    }

    #[test]
    fn test_pre_market_hours() {
        // 9:15 AM ET on a weekday (converted to UTC)
        let pre_market = chrono::DateTime::parse_from_rfc3339("2025-01-06T14:15:00Z")
            .unwrap()
            .with_timezone(&Utc);

        assert!(is_pre_market(pre_market));
    }

    #[test]
    fn test_after_hours() {
        // 5:00 PM ET on a weekday (converted to UTC)
        let after_hours = chrono::DateTime::parse_from_rfc3339("2025-01-06T22:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        assert!(is_after_hours(after_hours));
    }

    #[test]
    fn test_dynamic_slippage_low_vix() {
        // VIX < 15: 2% base slippage
        let params = DynamicSlippageParams::new(12.0, Moneyness::ATM);
        let slippage = calculate_dynamic_slippage(&params);
        assert_eq!(slippage, 0.02); // 2%
    }

    #[test]
    fn test_dynamic_slippage_normal_vix() {
        // VIX 15-25: 3% base slippage
        let params = DynamicSlippageParams::new(20.0, Moneyness::ATM);
        let slippage = calculate_dynamic_slippage(&params);
        assert_eq!(slippage, 0.03); // 3%
    }

    #[test]
    fn test_dynamic_slippage_elevated_vix() {
        // VIX 25-35: 4% base slippage
        let params = DynamicSlippageParams::new(30.0, Moneyness::ATM);
        let slippage = calculate_dynamic_slippage(&params);
        assert_eq!(slippage, 0.04); // 4%
    }

    #[test]
    fn test_dynamic_slippage_high_vix() {
        // VIX > 35: 5% base slippage
        let params = DynamicSlippageParams::new(40.0, Moneyness::ATM);
        let slippage = calculate_dynamic_slippage(&params);
        assert_eq!(slippage, 0.05); // 5%
    }

    #[test]
    fn test_dynamic_slippage_itm_adjustment() {
        // ITM options have tighter spreads: -0.5%
        let params = DynamicSlippageParams::new(20.0, Moneyness::ITM);
        let slippage = calculate_dynamic_slippage(&params);
        assert!((slippage - 0.025).abs() < 1e-10); // 3% - 0.5% = 2.5%
    }

    #[test]
    fn test_dynamic_slippage_otm_adjustment() {
        // OTM options have wider spreads: +0.5%
        let params = DynamicSlippageParams::new(20.0, Moneyness::OTM);
        let slippage = calculate_dynamic_slippage(&params);
        assert!((slippage - 0.035).abs() < 1e-10); // 3% + 0.5% = 3.5%
    }

    #[test]
    fn test_dynamic_slippage_edge_case_low_vix_itm() {
        // Low VIX (2%) + ITM (-0.5%) = 1.5%
        let params = DynamicSlippageParams::new(10.0, Moneyness::ITM);
        let slippage = calculate_dynamic_slippage(&params);
        assert_eq!(slippage, 0.015); // 2% - 0.5% = 1.5%
    }

    #[test]
    fn test_dynamic_slippage_edge_case_high_vix_otm() {
        // High VIX (5%) + OTM (+0.5%) = 5.5%
        let params = DynamicSlippageParams::new(45.0, Moneyness::OTM);
        let slippage = calculate_dynamic_slippage(&params);
        assert_eq!(slippage, 0.055); // 5% + 0.5% = 5.5%
    }

    #[test]
    fn test_calculate_fill_price_dynamic_buy_low_vix() {
        let params = DynamicSlippageParams::new(12.0, Moneyness::ATM);
        let market_price = 100.0;
        let fill = calculate_fill_price_dynamic(market_price, OrderSide::Buy, &params);

        // Should pay more than market price (2% slippage)
        assert!(fill > market_price);
        assert_eq!(fill, 102.0); // 100 + 2% slippage
    }

    #[test]
    fn test_calculate_fill_price_dynamic_sell_high_vix() {
        let params = DynamicSlippageParams::new(40.0, Moneyness::ATM);
        let market_price = 100.0;
        let fill = calculate_fill_price_dynamic(market_price, OrderSide::Sell, &params);

        // Should receive less than market price (5% slippage)
        assert!(fill < market_price);
        assert_eq!(fill, 95.0); // 100 - 5% slippage
    }

    #[test]
    fn test_calculate_fill_price_dynamic_itm_tighter_spreads() {
        let params = DynamicSlippageParams::new(20.0, Moneyness::ITM);
        let market_price = 100.0;
        let fill = calculate_fill_price_dynamic(market_price, OrderSide::Buy, &params);

        // Should pay less than ATM due to tighter spreads (2.5% vs 3%)
        assert_eq!(fill, 102.5); // 100 + 2.5% slippage
    }

    #[test]
    fn test_calculate_fill_price_dynamic_otm_wider_spreads() {
        let params = DynamicSlippageParams::new(20.0, Moneyness::OTM);
        let market_price = 100.0;
        let fill = calculate_fill_price_dynamic(market_price, OrderSide::Buy, &params);

        // Should pay more than ATM due to wider spreads (3.5% vs 3%)
        assert_eq!(fill, 103.5); // 100 + 3.5% slippage
    }
}
