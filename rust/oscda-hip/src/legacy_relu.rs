use super::*;

const BLOCK_SIZE: c_uint = 256;

pub type ReluKernel = Kernel;

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
    unsafe { oscda_kernel_load(path, c"applio_relu".as_ptr(), out) }
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
        oscda_kernel_launch(
            kernel,
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
    unsafe { oscda_kernel_unload(kernel) }
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
