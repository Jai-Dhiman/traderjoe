// System Monitoring Module
// Provides memory monitoring and system resource checks

use anyhow::{bail, Result};
use sysinfo::{System, Pid};
use tracing::{info, warn, error};

const GB: u64 = 1024 * 1024 * 1024;
const MB: u64 = 1024 * 1024;

pub struct SystemMonitor {
    system: System,
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub used_memory_gb: f64,
    pub memory_usage_pct: f64,
}

impl SystemMonitor {
    pub fn new() -> Self {
        let system = System::new_all();

        Self { system }
    }

    /// Check available memory and return stats
    pub fn check_available_memory(&mut self) -> Result<MemoryStats> {
        self.system.refresh_memory();

        let total_memory = self.system.total_memory();
        let available_memory = self.system.available_memory();
        let used_memory = self.system.used_memory();

        let stats = MemoryStats {
            total_memory_gb: total_memory as f64 / GB as f64,
            available_memory_gb: available_memory as f64 / GB as f64,
            used_memory_gb: used_memory as f64 / GB as f64,
            memory_usage_pct: (used_memory as f64 / total_memory as f64) * 100.0,
        };

        info!(
            "Memory: {:.2} GB available / {:.2} GB total ({:.1}% used)",
            stats.available_memory_gb,
            stats.total_memory_gb,
            stats.memory_usage_pct
        );

        // Warn if < 4GB available
        if stats.available_memory_gb < 4.0 {
            warn!(
                "Low memory warning: only {:.2} GB available (threshold: 4 GB)",
                stats.available_memory_gb
            );
        }

        // Halt if < 2GB available
        if stats.available_memory_gb < 2.0 {
            error!(
                "Critical memory shortage: only {:.2} GB available (minimum: 2 GB)",
                stats.available_memory_gb
            );
            bail!(
                "Insufficient memory: {:.2} GB available, minimum 2 GB required",
                stats.available_memory_gb
            );
        }

        Ok(stats)
    }

    /// Check memory before loading a model
    pub fn check_memory_before_model_load(&mut self, model_name: &str, estimated_size_gb: f64) -> Result<()> {
        let stats = self.check_available_memory()?;

        if stats.available_memory_gb < estimated_size_gb {
            bail!(
                "Insufficient memory to load model '{}': {:.2} GB available, {:.2} GB required",
                model_name,
                stats.available_memory_gb,
                estimated_size_gb
            );
        }

        info!(
            "Memory check passed for model '{}': {:.2} GB available, {:.2} GB required",
            model_name,
            stats.available_memory_gb,
            estimated_size_gb
        );

        Ok(())
    }

    /// Get current process memory usage
    pub fn get_process_memory_usage(&mut self) -> Result<f64> {
        self.system.refresh_processes();

        let pid = std::process::id();

        if let Some(process) = self.system.process(Pid::from_u32(pid)) {
            let memory_mb = process.memory() / MB;
            Ok(memory_mb as f64 / 1024.0) // Convert to GB
        } else {
            bail!("Failed to get current process info")
        }
    }


    /// Get detailed system stats for logging
    pub fn get_system_stats(&mut self) -> Result<serde_json::Value> {
        self.system.refresh_all();

        let memory_stats = self.check_available_memory()?;
        let process_memory_gb = self.get_process_memory_usage()?;

        Ok(serde_json::json!({
            "timestamp": chrono::Utc::now(),
            "memory": {
                "total_gb": memory_stats.total_memory_gb,
                "available_gb": memory_stats.available_memory_gb,
                "used_gb": memory_stats.used_memory_gb,
                "usage_pct": memory_stats.memory_usage_pct,
            },
            "process_memory_gb": process_memory_gb,
            "cpu_count": self.system.cpus().len(),
        }))
    }

    /// Log memory stats to daily logs
    pub fn log_memory_stats(&mut self) -> Result<()> {
        let stats = self.get_system_stats()?;
        info!("System stats: {}", serde_json::to_string_pretty(&stats)?);
        Ok(())
    }
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Global function to check available memory (convenience wrapper)
pub fn check_available_memory() -> Result<MemoryStats> {
    let mut monitor = SystemMonitor::new();
    monitor.check_available_memory()
}

/// Global function to check memory before model load (convenience wrapper)
pub fn check_memory_before_model_load(model_name: &str, estimated_size_gb: f64) -> Result<()> {
    let mut monitor = SystemMonitor::new();
    monitor.check_memory_before_model_load(model_name, estimated_size_gb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_monitor_creation() {
        let monitor = SystemMonitor::new();
        assert!(monitor.system.total_memory() > 0);
    }

    #[test]
    fn test_check_available_memory() {
        let mut monitor = SystemMonitor::new();
        let stats = monitor.check_available_memory();

        // Should succeed unless system has < 2GB RAM
        if let Ok(stats) = stats {
            assert!(stats.total_memory_gb > 0.0);
            assert!(stats.available_memory_gb >= 0.0);
            assert!(stats.memory_usage_pct >= 0.0 && stats.memory_usage_pct <= 100.0);
        }
    }

    #[test]
    fn test_get_process_memory_usage() {
        let mut monitor = SystemMonitor::new();
        let memory_gb = monitor.get_process_memory_usage();

        assert!(memory_gb.is_ok());
        assert!(memory_gb.unwrap() > 0.0);
    }

    #[test]
    fn test_get_system_stats() {
        let mut monitor = SystemMonitor::new();
        let stats = monitor.get_system_stats();

        if stats.is_ok() {
            let stats_json = stats.unwrap();
            assert!(stats_json.get("memory").is_some());
            assert!(stats_json.get("process_memory_gb").is_some());
        }
    }
}
