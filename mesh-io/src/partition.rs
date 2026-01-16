//! Partition file format encoder/decoder.
//!
//! See `mesh-part(1)` for a specification.

use std::fmt;
use std::io;

// TODO compile_error when sizeof(usize) < sizeof(u64)

#[derive(Debug)]
pub enum Error {
    BadHeader,
    UnsupportedVersion,
    Io(io::Error),
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::Io(err)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadHeader => write!(f, "bad file header"),
            Error::UnsupportedVersion => write!(f, "unsupported file version"),
            Error::Io(_) => write!(f, "read/write error"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Wrapping `r` in a [`std::io::BufReader`] is recommended.
pub fn read<R>(mut r: R) -> Result<Vec<usize>>
where
    R: io::Read,
{
    let mut header = [0x00; 4];
    r.read_exact(&mut header)?;
    if &header != b"MePe" {
        return Err(Error::BadHeader);
    }

    let mut count_buf = [0x00; 8];
    r.read_exact(&mut count_buf)?;
    let count = u64::from_le_bytes(count_buf) as usize;

    let mut partition = Vec::with_capacity(count);
    for _ in 0..count {
        let mut weight_buf = [0x00; 8];
        r.read_exact(&mut weight_buf)?;
        partition.push(u64::from_le_bytes(weight_buf) as usize);
    }

    Ok(partition)
}

/// Wrapping `w` in a [`std::io::BufWriter`] is recommended.
pub fn write<I, W>(mut w: W, array: I) -> io::Result<()>
where
    I: IntoIterator<Item = usize>,
    I::IntoIter: ExactSizeIterator,
    W: io::Write,
{
    let array = array.into_iter();

    write!(w, "MePe")?;
    w.write_all(&u64::to_le_bytes(array.len() as u64))?;

    for id in array {
        w.write_all(&u64::to_le_bytes(id as u64))?;
    }

    Ok(())
}
