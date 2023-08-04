//! Wraps a function into another one.
//!
//! See [`macro@wrap`].

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use quote::ToTokens;

// Taken from tokio-macros
fn token_stream_with_error(mut tokens: TokenStream2, error: syn::Error) -> TokenStream2 {
    tokens.extend(error.into_compile_error());
    tokens
}

fn expand(wrapper: syn::Path, input: syn::ItemFn) -> TokenStream2 {
    let mut tokens = TokenStream2::new();

    for mut attr in input.attrs {
        attr.style = syn::AttrStyle::Outer;
        attr.to_tokens(&mut tokens);
    }

    input.vis.to_tokens(&mut tokens);
    input.sig.to_tokens(&mut tokens);

    let braces = input.block.brace_token;
    let body = input.block;
    braces.surround(&mut tokens, |tokens| {
        quote![#wrapper(|| #body)].to_tokens(tokens);
    });

    tokens
}

fn wrap2(attr: TokenStream2, item: TokenStream2) -> TokenStream2 {
    let wrapper: syn::Path = match syn::parse2(attr.clone()) {
        Ok(v) => v,
        Err(err) => return token_stream_with_error(attr, err),
    };

    let input: syn::ItemFn = match syn::parse2(item.clone()) {
        Ok(v) => v,
        Err(err) => return token_stream_with_error(item, err),
    };

    expand(wrapper, input)
}

/// Wraps a function into another one.
///
/// Use this as `#[wrap(path::to::wrapper)]`.
///
/// # Example
///
/// Example where a panicking function is wrapped into
/// [`std::panic::catch_unwind`] to avoid unwinding accross FFI boundary:
///
/// ```
/// # use std::panic::UnwindSafe;
/// use wrap::wrap;
///
/// // Return -1 if `f` panics.
/// fn catch_unwind(f: impl FnOnce() -> i32 + UnwindSafe) -> i32 {
///     std::panic::catch_unwind(f).unwrap_or(-1)
/// }
///
/// #[wrap(catch_unwind)]
/// unsafe extern "C" fn might_panic(a: i32, b: i32) -> i32 {
///     assert!(a > 0);
///     assert!(b > 0);
///     a / b
/// }
///
/// assert_eq!(unsafe { might_panic(0, 0) }, -1);
/// ```
///
/// The above example expands to this:
///
/// ```
/// # use std::panic::UnwindSafe;
/// fn catch_unwind(f: impl FnOnce() -> i32 + UnwindSafe) -> i32 {
///     std::panic::catch_unwind(f).unwrap_or(-1)
/// }
///
/// unsafe extern "C" fn might_panic(a: i32, b: i32) -> i32 {
///     catch_unwind(|| {
///         assert!(a > 0);
///         assert!(b > 0);
///         a / b
///     })
/// }
///
/// assert_eq!(unsafe { might_panic(0, 0) }, -1);
/// ```
#[proc_macro_attribute]
pub fn wrap(attr: TokenStream, item: TokenStream) -> TokenStream {
    wrap2(attr.into(), item.into()).into()
}
