//! Weight file format encoder/decoder.
//!
//! See `weight-gen(1)` for a specification.

use std::any::TypeId;
use std::fmt;
use std::io;

// TODO compile_error when sizeof(usize) < sizeof(u64)

const VERSION: u8 = 1;
const FLAG_INTEGER: u8 = 1 << 0;

#[derive(Debug)]
pub enum Array {
    Integers(Vec<Vec<i64>>),
    Floats(Vec<Vec<f64>>),
}

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

fn read_inner<F, T, R>(mut r: R, criterion_count: usize, from_bytes: F) -> Result<Vec<Vec<T>>>
where
    F: Fn(&[u8]) -> T + Copy,
    R: io::Read,
{
    let mut count_buf = [0x00; 8];
    r.read_exact(&mut count_buf)?;
    let weight_count = u64::from_le_bytes(count_buf) as usize;

    let mut weights = Vec::with_capacity(weight_count);

    for _ in 0..weight_count {
        let mut weight_buf = vec![0x00; criterion_count * 8];
        r.read_exact(&mut weight_buf)?;
        let weight = weight_buf.chunks_exact(8).map(from_bytes).collect();
        weights.push(weight);
    }

    Ok(weights)
}

/// Wrapping `r` in a [`std::io::BufReader`] is recommended.
///
/// TODO checksum input data
pub fn read<R>(mut r: R) -> Result<Array>
where
    R: io::Read,
{
    let mut header = [0x00; 4];
    r.read_exact(&mut header)?;
    if &header != b"MeWe" {
        return Err(Error::BadHeader);
    }

    let mut flags = [0x00; 4];
    r.read_exact(&mut flags)?;
    let version = flags[0];
    if version != VERSION {
        return Err(Error::UnsupportedVersion);
    }
    let is_integer = (flags[1] & FLAG_INTEGER) != 0;
    let criterion_count = u16::from_le_bytes([flags[2], flags[3]]) as usize;
    if criterion_count == 0 {
        return Ok(Array::Integers(Vec::new()));
    }

    Ok(if is_integer {
        Array::Integers(read_inner(r, criterion_count, |bytes| {
            let bytes = <[u8; 8]>::try_from(bytes).unwrap();
            i64::from_le_bytes(bytes)
        })?)
    } else {
        Array::Floats(read_inner(r, criterion_count, |bytes| {
            let bytes = <[u8; 8]>::try_from(bytes).unwrap();
            f64::from_le_bytes(bytes)
        })?)
    })
}

fn write_inner<I, F, T, W>(mut w: W, array: I, to_bytes: F) -> Result<()>
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    I::Item: IntoIterator<Item = T>,
    <I::Item as IntoIterator>::IntoIter: ExactSizeIterator,
    F: Fn(T) -> [u8; 8],
    T: 'static,
    W: io::Write,
{
    let flags: u8 = if TypeId::of::<T>() == TypeId::of::<i64>() {
        FLAG_INTEGER
    } else {
        0
    };

    let mut array = array.into_iter();
    let len = array.len();
    let first = match array.next() {
        Some(v) => v.into_iter(),
        None => {
            let buf = [
                b'M', b'e', b'W', b'e', VERSION, flags, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ];
            w.write_all(&buf)?;
            return Ok(());
        }
    };
    let criterion_count = first.len();
    // assert!(
    //     criterion_count < std::mem::size_of::<u16>(),
    //     "Too many criterions",
    // );

    write!(w, "MeWe")?;
    w.write_all(&[VERSION, flags])?;
    w.write_all(&u16::to_le_bytes(criterion_count as u16))?;
    w.write_all(&u64::to_le_bytes(len as u64))?;

    let mut write_weight = move |weight: <I::Item as IntoIterator>::IntoIter| -> Result<()> {
        for criterion in weight {
            w.write_all(&to_bytes(criterion))?;
        }
        Ok(())
    };

    write_weight(first)?;
    array
        .map(IntoIterator::into_iter)
        .try_for_each(write_weight)
}

/// Wrapping `r` in a [`std::io::BufWriter`] is recommended.
///
/// TODO checksum input data
pub fn write_integers<I, W>(w: W, array: I) -> Result<()>
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    I::Item: IntoIterator<Item = i64>,
    <I::Item as IntoIterator>::IntoIter: ExactSizeIterator,
    W: io::Write,
{
    write_inner(w, array, i64::to_le_bytes)
}

/// Wrapping `w` in a [`std::io::BufWriter`] is recommended.
///
/// TODO checksum input data
pub fn write_floats<I, W>(w: W, array: I) -> Result<()>
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    I::Item: IntoIterator<Item = f64>,
    <I::Item as IntoIterator>::IntoIter: ExactSizeIterator,
    W: io::Write,
{
    write_inner(w, array, f64::to_le_bytes)
}
