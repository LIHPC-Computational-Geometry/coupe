use anyhow::Context as _;
use anyhow::Result;
use rusqlite::OptionalExtension as _;
use std::env;
use std::path::PathBuf;

/// List of database migrations, to seamlessly update the database without losing data.
const MIGRATIONS: &[&str] = &[include_str!("migration-000-init.sql")];

/// Initialize, or update the database's schema to the latest version.
fn upgrade_schema(database: &mut rusqlite::Connection) -> Result<()> {
    let tx = database
        .transaction()
        .context("failed to begin transaction")?;

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

fn default_path() -> Result<PathBuf> {
    let cargo_target_dir = env::var("CARGO_TARGET_DIR").context("quoi")?;
    let mut path = PathBuf::from(cargo_target_dir);
    path.push("part");
    Ok(path)
}

pub struct Experiment<'a> {
    pub algorithm: &'a str,
    pub seed_id: i64,
    pub distribution_id: i64,
    pub weight_count: usize,
    pub iteration: usize,
    pub case_type: bool,
    pub imbalance: f64,
    pub algo_iterations: Option<usize>,
}

pub struct Database {
    conn: rusqlite::Connection,
}

pub fn open(filename: Option<String>) -> Result<Database> {
    let filename = match filename {
        Some(s) => PathBuf::from(s),
        None => default_path()?,
    };
    let mut conn = rusqlite::Connection::open(filename)?;
    upgrade_schema(&mut conn).context("failed to upgrade the database schema")?;
    Ok(Database { conn })
}

impl Database {
    pub fn insert_distribution(&mut self, name: &str, params: [f64; 3]) -> Result<i64> {
        let tx = self
            .conn
            .transaction()
            .context("failed to begin transaction")?;

        let id = tx
            .query_row(
                "SELECT id FROM distribution
                WHERE name = ? AND param1 = ? AND param2 = ? AND param3 = ?",
                rusqlite::params![name, params[0], params[1], params[2]],
                |row| row.get(0),
            )
            .optional()
            .context("failed to query database")?;

        if let Some(id) = id {
            return Ok(id);
        }

        tx.execute(
            "INSERT INTO
            distribution (name, param1, param2, param3)
            VALUES       (?   , ?     , ?     , ?     )",
            rusqlite::params![name, params[0], params[1], params[2]],
        )
        .context("failed to insert row")?;
        let id = tx.last_insert_rowid();
        tx.commit().context("failed to commit transaction")?;

        Ok(id)
    }

    pub fn insert_seed(&mut self, seed: &[u8]) -> Result<i64> {
        let tx = self
            .conn
            .transaction()
            .context("failed to begin transaction")?;

        let id = tx
            .query_row("SELECT id FROM seed WHERE bytes = ?", [seed], |row| {
                row.get(0)
            })
            .optional()
            .context("failed to query database")?;

        if let Some(id) = id {
            return Ok(id);
        }

        tx.execute("INSERT INTO seed(bytes) VALUES(?)", [seed])
            .context("failed to insert row")?;
        let id = tx.last_insert_rowid();
        tx.commit().context("failed to commit transaction")?;

        Ok(id)
    }

    pub fn insert_experiment(&mut self, e: Experiment<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO
            experiment (algorithm, seed, distribution, sample_size, iteration, case_type, imbalance, algo_iterations)
            VALUES     (?        , ?   , ?           , ?          , ?        , ?        , ?        , ?              )",
            rusqlite::params![
                e.algorithm,
                e.seed_id,
                e.distribution_id,
                e.weight_count,
                e.iteration,
                e.case_type,
                e.imbalance,
                e.algo_iterations,
            ],
        )?;
        Ok(())
    }
}
