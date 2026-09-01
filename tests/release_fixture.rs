// Copyright 2025, 2026 Query Farm LLC - https://query.farm

//! Suite-wide opt-in gate for worker-backed integration tests.

use crate::common;

#[test]
fn required_release_fixture_exists() {
    // The shared locator asserts when VGI_REQUIRE_RELEASE_FIXTURE is enabled;
    // otherwise a missing fixture remains an ordinary local-development skip.
    let _ = common::example_worker();
}
