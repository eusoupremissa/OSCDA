//! HIP tensor interop for the OSCDA inference example.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ptr::{NonNull, null_mut};

const INVALID_VALUE: c_int = 1;
const INITIALIZATION_ERROR: c_int = 3;
const BLOCK_SIZE: c_uint = 256;

#[repr(C)]
struct Dim3 {
    x: c_uint,
    y: c_uint,
    z: c_uint,
}

impl Dim3 {
    fn linear(x: c_uint) -> Self {
        Self { x, y: 1, z: 1 }
    }
}

unsafe extern "C" {
    fn oscModuleLoad(module: *mut *mut c_void, path: *const c_char) -> c_int;
    fn oscModuleUnload(module: *mut c_void) -> c_int;
    fn oscModuleGetFunction(
        function: *mut *mut c_void,
        module: *mut c_void,
        name: *const c_char,
    ) -> c_int;
    fn oscLaunchKernelOnHipStream(
        function: *mut c_void,
        grid: Dim3,
        block: Dim3,
        shared_memory_bytes: c_uint,
        arguments: *mut *mut c_void,
        hip_stream: *mut c_void,
    ) -> c_int;
}

fn status(value: c_int) -> Result<(), c_int> {
    if value == 0 { Ok(()) } else { Err(value) }
}

struct Module(Option<NonNull<c_void>>);

impl Module {
    fn load(path: &CStr) -> Result<Self, c_int> {
        let mut module = null_mut();
        // HIP writes the module handle into this local and does not retain the path.
        status(unsafe { oscModuleLoad(&mut module, path.as_ptr()) })?;
        Ok(Self(Some(
            NonNull::new(module).ok_or(INITIALIZATION_ERROR)?,
        )))
    }

    fn kernel(&self, name: &CStr) -> Result<NonNull<c_void>, c_int> {
        let mut function = null_mut();
        let module = self.0.ok_or(INITIALIZATION_ERROR)?;
        // The loaded module remains live while its function handle is used.
        status(unsafe { oscModuleGetFunction(&mut function, module.as_ptr(), name.as_ptr()) })?;
        NonNull::new(function).ok_or(INITIALIZATION_ERROR)
    }

    fn unload(&mut self) -> c_int {
        // Take the handle before unloading so Drop cannot unload it again.
        self.0
            .take()
            .map_or(0, |module| unsafe { oscModuleUnload(module.as_ptr()) })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        self.unload();
    }
}

pub struct ReluKernel {
    module: Module,
    function: NonNull<c_void>,
}

fn geometry(elements: u64) -> Result<(c_uint, c_uint), c_int> {
    let count = c_uint::try_from(elements).map_err(|_| INVALID_VALUE)?;
    if count == 0 {
        return Err(INVALID_VALUE);
    }
    Ok((count, count.div_ceil(BLOCK_SIZE)))
}

/// Load the translated ReLU code object for the current HIP device.
///
/// # Safety
/// `path` must be a valid C string. `out` must be writable and must not hold
/// a live kernel. Keep the same HIP device selected for launch and unload.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oscda_relu_load(path: *const c_char, out: *mut *mut ReluKernel) -> c_int {
    if path.is_null() || out.is_null() {
        return INVALID_VALUE;
    }
    unsafe { out.write(null_mut()) };
    let module = match Module::load(unsafe { CStr::from_ptr(path) }) {
        Ok(module) => module,
        Err(error) => return error,
    };
    let function = match module.kernel(c"applio_relu") {
        Ok(function) => function,
        Err(error) => return error,
    };
    unsafe { out.write(Box::into_raw(Box::new(ReluKernel { module, function }))) };
    0
}

/// Enqueue FP32 ReLU without copying tensors or synchronizing the stream.
///
/// # Safety
/// `kernel` must be live. Buffers must hold `elements` contiguous FP32 values
/// on the selected device and remain alive until the work completes. The
/// borrowed stream and buffers must belong to the HIP runtime used to load
/// the module. The caller must serialize against unloading the kernel.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oscda_relu_launch(
    kernel: *const ReluKernel,
    mut input: u64,
    mut output: u64,
    elements: u64,
    stream: *mut c_void,
) -> c_int {
    let Some(kernel) = (unsafe { kernel.as_ref() }) else {
        return INVALID_VALUE;
    };
    if input == 0 || output == 0 {
        return INVALID_VALUE;
    }
    let (mut count, blocks) = match geometry(elements) {
        Ok(geometry) => geometry,
        Err(error) => return error,
    };
    let mut arguments = [
        (&raw mut input).cast::<c_void>(),
        (&raw mut output).cast::<c_void>(),
        (&raw mut count).cast::<c_void>(),
    ];
    unsafe {
        oscLaunchKernelOnHipStream(
            kernel.function.as_ptr(),
            Dim3::linear(blocks),
            Dim3::linear(BLOCK_SIZE),
            0,
            arguments.as_mut_ptr(),
            stream,
        )
    }
}

/// Unload a kernel created by `oscda_relu_load`.
///
/// # Safety
/// Pass a valid, uniquely owned kernel once, on the same HIP device. No other
/// thread may use the handle during or after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oscda_relu_unload(kernel: *mut ReluKernel) -> c_int {
    if kernel.is_null() {
        return INVALID_VALUE;
    }
    let mut kernel = unsafe { Box::from_raw(kernel) };
    kernel.module.unload()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_covers_tail_without_overflow() {
        assert_eq!(geometry(0), Err(INVALID_VALUE));
        assert_eq!(geometry(256), Ok((256, 1)));
        assert_eq!(geometry(257), Ok((257, 2)));
        assert_eq!(geometry(u64::from(u32::MAX)), Ok((u32::MAX, 16_777_216)));
        assert_eq!(geometry(u64::from(u32::MAX) + 1), Err(INVALID_VALUE));
    }
}
