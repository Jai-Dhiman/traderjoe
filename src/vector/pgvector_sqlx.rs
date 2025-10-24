// Custom pgvector type for sqlx to handle vector type encoding/decoding
// Uses binary protocol for correct encoding

use sqlx::encode::IsNull;
use sqlx::error::BoxDynError;
use sqlx::postgres::{PgArgumentBuffer, PgHasArrayType, PgTypeInfo, PgValueRef};
use sqlx::{Decode, Encode, Postgres, Type};

/// Wrapper type for pgvector's vector type
#[derive(Debug, Clone, PartialEq)]
pub struct PgVector(pub Vec<f32>);

impl PgVector {
    pub fn new(vec: Vec<f32>) -> Self {
        Self(vec)
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

impl From<Vec<f32>> for PgVector {
    fn from(vec: Vec<f32>) -> Self {
        Self(vec)
    }
}

impl From<PgVector> for Vec<f32> {
    fn from(vec: PgVector) -> Self {
        vec.0
    }
}

impl Type<Postgres> for PgVector {
    fn type_info() -> PgTypeInfo {
        PgTypeInfo::with_name("vector")
    }
}

impl PgHasArrayType for PgVector {
    fn array_type_info() -> PgTypeInfo {
        PgTypeInfo::with_name("_vector")
    }
}

impl Encode<'_, Postgres> for PgVector {
    fn encode_by_ref(&self, buf: &mut PgArgumentBuffer) -> IsNull {
        // pgvector binary format:
        // - 2 bytes: number of dimensions (u16, big-endian)
        // - 2 bytes: unused (0)
        // - for each dimension: 4 bytes (f32, big-endian / network byte order)

        let dim = self.0.len() as u16;

        // Write dimension count
        buf.extend_from_slice(&dim.to_be_bytes());

        // Write unused bytes
        buf.extend_from_slice(&[0u8, 0u8]);

        // Write float values in big-endian
        for &value in &self.0 {
            buf.extend_from_slice(&value.to_be_bytes());
        }

        IsNull::No
    }
}

impl Decode<'_, Postgres> for PgVector {
    fn decode(value: PgValueRef<'_>) -> Result<Self, BoxDynError> {
        // pgvector returns vectors in the format: [1.0,2.0,3.0]
        let s = <&str as Decode<Postgres>>::decode(value)?;

        // Remove brackets and parse floats
        let s = s.trim();
        if !s.starts_with('[') || !s.ends_with(']') {
            return Err(format!("Invalid vector format: expected [x,y,z], got {}", s).into());
        }

        let content = &s[1..s.len() - 1];
        if content.is_empty() {
            return Ok(PgVector(Vec::new()));
        }

        let floats: Result<Vec<f32>, _> = content
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect();

        match floats {
            Ok(vec) => Ok(PgVector(vec)),
            Err(e) => Err(format!("Failed to parse vector components: {}", e).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pgvector_creation() {
        let vec = vec![1.0, 2.0, 3.0];
        let pg_vec = PgVector::new(vec.clone());
        assert_eq!(pg_vec.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(pg_vec.into_inner(), vec);
    }

    #[test]
    fn test_pgvector_from() {
        let vec = vec![4.0, 5.0, 6.0];
        let pg_vec = PgVector::from(vec.clone());
        assert_eq!(Vec::from(pg_vec), vec);
    }
}
