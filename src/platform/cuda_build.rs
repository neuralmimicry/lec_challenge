//! Supported toolkit releases of the locked cudarc 0.16.6 dependency.
//! Driver capability is an independent upper bound, never toolkit support.
pub fn supported(major: u32, minor: u32) -> bool {
    matches!((major, minor), (11, 4..=8) | (12, 0..=6) | (12, 8..=9))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dependency_support_is_not_the_driver_ceiling() {
        assert!(
            include_str!("../../Cargo.lock").contains("name = \"cudarc\"\nversion = \"0.16.6\"")
        );
        for (major, minor) in [(12, 9), (12, 8), (11, 8)] {
            assert!(supported(major, minor));
        }
        for (major, minor) in [(13, 3), (13, 0), (12, 7), (12, 10)] {
            assert!(!supported(major, minor));
        }
    }
}
