//! The error type that crosses the VapourSynth boundary.
//!
//! `Filter::Error` must be `AsRef<CStr>`, so every failure is built as a message
//! once and handed over without allocating on the error path.

use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt::{self, Display};

use vapoursynth4_rs::map::MapPropertyError;

/// An error reported to VapourSynth from a filter.
#[derive(Debug)]
pub struct NImagesError {
    message: CString,
}

impl NImagesError {
    /// Builds an error from any message. a message holding a NUL byte is
    /// replaced rather than panicking, because this runs on the frame path.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        let message = CString::new(message).unwrap_or_else(|_| {
            CString::new("nimages error message contained a NUL byte")
                .expect("the replacement message is NUL-free")
        });
        Self { message }
    }

    /// Builds an error for a failed map operation, naming the property.
    #[must_use]
    pub fn property(name: impl Display, error: MapPropertyError) -> Self {
        Self::new(format!("{name}: {error}"))
    }
}

impl AsRef<CStr> for NImagesError {
    fn as_ref(&self) -> &CStr {
        &self.message
    }
}

impl Display for NImagesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.to_string_lossy().fmt(formatter)
    }
}

impl Error for NImagesError {}

/// The result type every filter entry point returns.
pub type Result<T> = std::result::Result<T, NImagesError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nul_byte_is_replaced_rather_than_panicking() {
        let error = NImagesError::new("bad\0message");
        assert_eq!(
            error.to_string(),
            "nimages error message contained a NUL byte"
        );
    }

    #[test]
    fn the_message_reaches_the_c_str() {
        let error = NImagesError::new("black level must be lower than white level");
        assert_eq!(
            error.as_ref().to_str().expect("ascii"),
            "black level must be lower than white level"
        );
    }

    #[test]
    fn display_and_error_are_implemented() {
        let error = NImagesError::property("NImagesBlackLevel", MapPropertyError::KeyNotFound);
        assert_eq!(
            error.to_string(),
            "NImagesBlackLevel: The requested key was not found in the map"
        );
        let _: &dyn Error = &error;
    }
}
