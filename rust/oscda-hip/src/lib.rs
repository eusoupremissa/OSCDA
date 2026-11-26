//! Load and launch AMDGPU code objects on streams borrowed from HIP.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ptr::{NonNull, null_mut};

const INVALID_VALUE: c_int = 1;
const INITIALIZATION_ERROR: c_int = 3;

#[repr(C)]
pub struct Dim3 {
    pub x: c_uint,
    pub y: c_uint,
    pub z: c_uint,
}

impl Dim3 {
    pub fn linear(x: c_uint) -> Self {
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

mod legacy_relu;
pub use legacy_relu::{ReluKernel, oscda_relu_launch, oscda_relu_load, oscda_relu_unload};

/// A kernel and the loaded module that owns its function handle.
pub struct Kernel {
    module: Module,
    function: NonNull<c_void>,
}

impl Kernel {
    /// Load an AMDGPU code object and resolve one entry point.
    ///
    /// # Safety
    /// Keep the same HIP device selected when launching or dropping this kernel.
    pub unsafe fn load(path: &CStr, symbol: &CStr) -> Result<Self, c_int> {
        let module = Module::load(path)?;
        let function = module.kernel(symbol)?;
        Ok(Self { module, function })
    }

    /// Enqueue work on a borrowed HIP stream. Null selects the HIP default stream.
    ///
    /// # Safety
    /// Arguments must match the device kernel signature. The HIP device and runtime
    /// must own the stream and any device buffers, which must outlive the queued work.
    pub unsafe fn launch(
        &self,
        grid: Dim3,
        block: Dim3,
        shared: c_uint,
        arguments: &mut [*mut c_void],
        stream: *mut c_void,
    ) -> Result<(), c_int> {
        let arguments = if arguments.is_empty() {
            null_mut()
        } else {
            arguments.as_mut_ptr()
        };
        status(unsafe { self.launch_raw(grid, block, shared, arguments, stream) })
    }

    unsafe fn launch_raw(
        &self,
        grid: Dim3,
        block: Dim3,
        shared: c_uint,
        arguments: *mut *mut c_void,
        stream: *mut c_void,
    ) -> c_int {
        unsafe {
            oscLaunchKernelOnHipStream(
                self.function.as_ptr(),
                grid,
                block,
                shared,
                arguments,
                stream,
            )
        }
    }

    /// Synchronize and unload, reporting any HIP error.
    pub fn close(mut self) -> Result<(), c_int> {
        status(self.module.unload())
    }
}

/// Load a named kernel for the current HIP device.
///
/// # Safety
/// Both strings must be valid. The output must be writable and must not hold
/// a live kernel. Keep the device selected for launches and unload.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oscda_kernel_load(
    path: *const c_char,
    symbol: *const c_char,
    out: *mut *mut Kernel,
) -> c_int {
    if path.is_null() || symbol.is_null() || out.is_null() {
        return INVALID_VALUE;
    }
    unsafe { out.write(null_mut()) };
    match unsafe { Kernel::load(CStr::from_ptr(path), CStr::from_ptr(symbol)) } {
        Ok(kernel) => {
            unsafe { out.write(Box::into_raw(Box::new(kernel))) };
            0
        }
        Err(error) => error,
    }
}

/// Enqueue a named kernel on a HIP stream owned by the caller.
///
/// # Safety
/// The handle must be live, the argument array must match the entry point ABI,
/// and the stream and buffers must belong to the current HIP runtime and device.
/// Keep buffers and the stream alive until the queued work completes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oscda_kernel_launch(
    kernel: *const Kernel,
    grid: Dim3,
    block: Dim3,
    shared: c_uint,
    arguments: *mut *mut c_void,
    stream: *mut c_void,
) -> c_int {
    match unsafe { kernel.as_ref() } {
        Some(kernel) => unsafe { kernel.launch_raw(grid, block, shared, arguments, stream) },
        None => INVALID_VALUE,
    }
}

/// Unload a kernel from the current HIP device.
///
/// # Safety
/// Pass one uniquely owned kernel returned by oscda_kernel_load. Do not use
/// the handle again or allow another thread to use it during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oscda_kernel_unload(kernel: *mut Kernel) -> c_int {
    if kernel.is_null() {
        return INVALID_VALUE;
    }
    let kernel = unsafe { Box::from_raw(kernel) };
    kernel.close().err().unwrap_or(0)
}
