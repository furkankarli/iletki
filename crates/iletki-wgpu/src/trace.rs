//! Render instrumentation is compiled out of ordinary builds.

#[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
pub(crate) use crate::profiling::{pass, scope};

#[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
pub(crate) use disabled::{pass, scope};

#[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
mod disabled {
    pub(crate) enum Pass {}

    impl Pass {
        pub(crate) fn writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
            // This method is never reached. The uninstrumented pass constructor returns None.
            match *self {}
        }
    }

    pub(crate) const fn pass(_label: &'static str) -> Option<Pass> {
        None
    }

    pub(crate) struct Scope;

    pub(crate) const fn scope(_label: &'static str) -> Scope {
        Scope
    }
}
