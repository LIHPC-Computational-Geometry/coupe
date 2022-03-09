use anyhow::Context as _;
use anyhow::Result;
use std::env;
use std::path::PathBuf;

/// List of database migrations, to seamlessly update the database without losing data.
const MIGRATIONS: &[&str] = &[include_str!("migration-000-init.sql")];

/// Initialize, or update the database's schema to the latest version.
pub fn upgrade_schema(database: &mut rusqlite::Connection) -> Result<()> {
    let tx = database
        .transaction()
        .context("failed to begin transaction")?;

    // sqlite doesn't support 64-bit integer types, so we use a u32 for versions.
    // https://github.com/rusqlite/rusqlite/issues/250#issuecomment-285406949
    let my_version = MIGRATIONS.len();
    let db_version = tx
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .context("failed to query database version")?;

    if my_version == db_version {
        // Already at the latest version.
        return Ok(());
    }
    if my_version < db_version {
        anyhow::bail!(
            "Your version ({}) is older than the database's ({})",
            my_version,
            db_version
        );
    }

    eprintln!(
        "Upgrading database from version {} to version {}",
        db_version, my_version
    );
    for v in db_version..my_version {
        tx.execute_batch(MIGRATIONS[v as usize])
            .with_context(|| format!("failed to execute migration #{}", v))?;
    }

    // PRAGMA user_version does not seem to support prepared statements.
    tx.execute_batch(&format!("PRAGMA user_version = {}", my_version))
        .context("failed to update database version")?;
    tx.commit().context("failed to commit transaction")?;

    Ok(())
}

pub fn default_path() -> Result<PathBuf> {
    let cargo_target_dir = env::var("CARGO_TARGET_DIR").context("quoi")?;
    let mut path = PathBuf::from(cargo_target_dir);
    path.push("part");
    Ok(path)
}
