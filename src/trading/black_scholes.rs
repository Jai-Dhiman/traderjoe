//! Black-Scholes Options Pricing Model
//!
//! Implements the Black-Scholes formula for European-style options pricing
//! Used for realistic P&L calculation in backtesting when historical options data unavailable

use std::f64::consts::{E, PI};

/// Standard normal cumulative distribution function
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / f64::sqrt(2.0)))
}

/// Calculate d1 parameter for Black-Scholes
fn calculate_d1(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    let numerator = f64::ln(spot / strike) + (rate + 0.5 * volatility.powi(2)) * time_to_expiry;
    let denominator = volatility * f64::sqrt(time_to_expiry);
    numerator / denominator
}

/// Calculate d2 parameter for Black-Scholes
fn calculate_d2(d1: f64, volatility: f64, time_to_expiry: f64) -> f64 {
    d1 - volatility * f64::sqrt(time_to_expiry)
}

/// Calculate the price of a European call option using Black-Scholes
///
/// # Arguments
/// * `spot` - Current price of the underlying asset
/// * `strike` - Strike price of the option
/// * `rate` - Risk-free interest rate (annualized)
/// * `volatility` - Implied volatility (annualized, as decimal, e.g., 0.20 for 20%)
/// * `time_to_expiry` - Time to expiration in years (e.g., 1/365 for 1 day)
pub fn call_price(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    // Handle edge cases
    if time_to_expiry <= 0.0 {
        return f64::max(0.0, spot - strike); // Intrinsic value at expiration
    }

    if volatility <= 0.0 {
        return f64::max(0.0, spot - strike); // No time value without volatility
    }

    let d1 = calculate_d1(spot, strike, rate, volatility, time_to_expiry);
    let d2 = calculate_d2(d1, volatility, time_to_expiry);

    let call = spot * norm_cdf(d1) - strike * E.powf(-rate * time_to_expiry) * norm_cdf(d2);
    f64::max(0.0, call) // Options can't have negative value
}

/// Calculate the price of a European put option using Black-Scholes
///
/// # Arguments
/// * `spot` - Current price of the underlying asset
/// * `strike` - Strike price of the option
/// * `rate` - Risk-free interest rate (annualized)
/// * `volatility` - Implied volatility (annualized, as decimal, e.g., 0.20 for 20%)
/// * `time_to_expiry` - Time to expiration in years (e.g., 1/365 for 1 day)
pub fn put_price(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    // Handle edge cases
    if time_to_expiry <= 0.0 {
        return f64::max(0.0, strike - spot); // Intrinsic value at expiration
    }

    if volatility <= 0.0 {
        return f64::max(0.0, strike - spot); // No time value without volatility
    }

    let d1 = calculate_d1(spot, strike, rate, volatility, time_to_expiry);
    let d2 = calculate_d2(d1, volatility, time_to_expiry);

    let put = strike * E.powf(-rate * time_to_expiry) * norm_cdf(-d2) - spot * norm_cdf(-d1);
    f64::max(0.0, put) // Options can't have negative value
}

/// Calculate option delta (first derivative of price with respect to spot)
/// Represents the rate of change of option price relative to underlying price
///
/// For calls: Delta is between 0 and 1
/// For puts: Delta is between -1 and 0
pub fn delta(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
    is_call: bool,
) -> f64 {
    if time_to_expiry <= 0.0 {
        // At expiration
        if is_call {
            return if spot > strike { 1.0 } else { 0.0 };
        } else {
            return if spot < strike { -1.0 } else { 0.0 };
        }
    }

    let d1 = calculate_d1(spot, strike, rate, volatility, time_to_expiry);

    if is_call {
        norm_cdf(d1)
    } else {
        norm_cdf(d1) - 1.0
    }
}

/// Calculate option gamma (second derivative of price with respect to spot)
/// Represents the rate of change of delta
/// Same for both calls and puts
pub fn gamma(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return 0.0;
    }

    let d1 = calculate_d1(spot, strike, rate, volatility, time_to_expiry);
    let n_prime_d1 = (1.0 / f64::sqrt(2.0 * PI)) * E.powf(-0.5 * d1.powi(2));

    n_prime_d1 / (spot * volatility * f64::sqrt(time_to_expiry))
}

/// Calculate option theta (time decay)
/// Represents the change in option price as time passes (per day)
/// Usually negative for long options
pub fn theta(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
    is_call: bool,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return 0.0;
    }

    let d1 = calculate_d1(spot, strike, rate, volatility, time_to_expiry);
    let d2 = calculate_d2(d1, volatility, time_to_expiry);
    let n_prime_d1 = (1.0 / f64::sqrt(2.0 * PI)) * E.powf(-0.5 * d1.powi(2));

    let term1 = -(spot * n_prime_d1 * volatility) / (2.0 * f64::sqrt(time_to_expiry));

    if is_call {
        let term2 = rate * strike * E.powf(-rate * time_to_expiry) * norm_cdf(d2);
        (term1 - term2) / 365.0 // Convert to per-day theta
    } else {
        let term2 = rate * strike * E.powf(-rate * time_to_expiry) * norm_cdf(-d2);
        (term1 + term2) / 365.0 // Convert to per-day theta
    }
}

/// Calculate option vega (sensitivity to volatility)
/// Represents the change in option price for a 1% change in implied volatility
/// Same for both calls and puts
pub fn vega(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return 0.0;
    }

    let d1 = calculate_d1(spot, strike, rate, volatility, time_to_expiry);
    let n_prime_d1 = (1.0 / f64::sqrt(2.0 * PI)) * E.powf(-0.5 * d1.powi(2));

    spot * n_prime_d1 * f64::sqrt(time_to_expiry) / 100.0 // Divide by 100 for 1% change
}

/// Helper function to select strike price based on target delta
/// For 0-1 DTE options, typical strategies use 30-40 delta
///
/// # Arguments
/// * `spot` - Current price of underlying
/// * `target_delta` - Desired delta (e.g., 0.30 for 30-delta call)
/// * `is_call` - true for call, false for put
///
/// # Returns
/// Approximate strike price that would give the target delta
pub fn strike_from_delta(
    spot: f64,
    target_delta: f64,
    volatility: f64,
    time_to_expiry: f64,
    is_call: bool,
) -> f64 {
    // Use iterative search to find strike that gives target delta
    // Start with ATM and search in 0.5% increments

    let rate = 0.05 / 365.0; // Assume 5% annual risk-free rate, per day
    let mut best_strike = spot;
    let mut best_diff = f64::MAX;

    // Search range: ±20% from spot
    let search_range = if is_call {
        // For calls, search above spot
        (spot * 0.95, spot * 1.20)
    } else {
        // For puts, search below spot
        (spot * 0.80, spot * 1.05)
    };

    let step = spot * 0.005; // 0.5% steps
    let mut current_strike = search_range.0;

    while current_strike <= search_range.1 {
        let current_delta = delta(spot, current_strike, rate, volatility, time_to_expiry, is_call);
        let diff = (current_delta - target_delta).abs();

        if diff < best_diff {
            best_diff = diff;
            best_strike = current_strike;
        }

        current_strike += step;
    }

    best_strike
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_price_atm() {
        // ATM call with 1 month to expiry, 20% vol
        let price = call_price(100.0, 100.0, 0.05, 0.20, 30.0 / 365.0);
        // Should be around $1-2 for this setup
        assert!(price > 0.5 && price < 3.0, "ATM call price: {}", price);
    }

    #[test]
    fn test_put_call_parity() {
        // Put-call parity: C - P = S - K * e^(-rT)
        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let vol = 0.20;
        let time = 30.0 / 365.0;

        let call = call_price(spot, strike, rate, vol, time);
        let put = put_price(spot, strike, rate, vol, time);
        let parity = call - put;
        let expected = spot - strike * E.powf(-rate * time);

        assert!((parity - expected).abs() < 0.01, "Put-call parity violated");
    }

    #[test]
    fn test_delta_bounds() {
        // Call delta should be between 0 and 1
        let call_delta = delta(100.0, 100.0, 0.05, 0.20, 30.0 / 365.0, true);
        assert!(call_delta >= 0.0 && call_delta <= 1.0);

        // Put delta should be between -1 and 0
        let put_delta = delta(100.0, 100.0, 0.05, 0.20, 30.0 / 365.0, false);
        assert!(put_delta >= -1.0 && put_delta <= 0.0);
    }

    #[test]
    fn test_zero_time_to_expiry() {
        // At expiration, call should equal intrinsic value
        let call = call_price(105.0, 100.0, 0.05, 0.20, 0.0);
        assert!((call - 5.0).abs() < 0.01, "Call at expiry should be intrinsic value");

        let put = put_price(95.0, 100.0, 0.05, 0.20, 0.0);
        assert!((put - 5.0).abs() < 0.01, "Put at expiry should be intrinsic value");
    }
}
