use std::path::PathBuf;

pub const REQUIRE_RELEASE_FIXTURE_ENV: &str = "VGI_REQUIRE_RELEASE_FIXTURE";

fn enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        )
    })
}

/// Locate the sibling Rust example worker. CI can set
/// `VGI_REQUIRE_RELEASE_FIXTURE=1` to turn a missing release fixture into a
/// hard failure; local runs retain the convenient skip behavior and may use a
/// debug fixture.
pub fn example_worker() -> Option<PathBuf> {
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("vgi-rust")
        .join("target");
    let executable = if cfg!(windows) {
        "vgi-example-worker.exe"
    } else {
        "vgi-example-worker"
    };
    let release = target.join("release").join(executable);
    if enabled(REQUIRE_RELEASE_FIXTURE_ENV) {
        assert!(
            release.is_file(),
            "{REQUIRE_RELEASE_FIXTURE_ENV}=1 but release fixture is missing: {}",
            release.display()
        );
        return Some(release);
    }
    [target.join("debug").join(executable), release]
        .into_iter()
        .find(|path| path.is_file())
}

#[allow(dead_code)]
pub fn sql_quote(value: &str) -> String {
    value.replace('\'', "''")
}
