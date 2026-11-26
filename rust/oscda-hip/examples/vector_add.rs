use oscda_hip::{Dim3, Kernel};
use std::ffi::{CString, c_int, c_void};
use std::os::unix::ffi::OsStrExt;
use std::ptr::null_mut;

unsafe extern "C" {
    fn cudaMalloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFree(pointer: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
}

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;

fn check(operation: &str, status: c_int) {
    assert_eq!(status, 0, "{operation} returned {status}");
}

struct Buffer(*mut c_void);

impl Buffer {
    fn from_host(values: &[f32]) -> Self {
        let mut ptr = null_mut();
        let bytes = std::mem::size_of_val(values);
        check("cudaMalloc", unsafe { cudaMalloc(&mut ptr, bytes) });
        let buffer = Self(ptr);
        check("cudaMemcpy", unsafe {
            cudaMemcpy(buffer.0, values.as_ptr().cast(), bytes, HOST_TO_DEVICE)
        });
        buffer
    }

    fn copy_to_host(&self, output: &mut [f32]) {
        check("cudaMemcpy", unsafe {
            cudaMemcpy(
                output.as_mut_ptr().cast(),
                self.0,
                std::mem::size_of_val(output),
                DEVICE_TO_HOST,
            )
        });
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { cudaFree(self.0) };
    }
}

fn main() {
    let path = std::env::args_os().nth(1).expect("pass vector_add.hsaco");
    let path = CString::new(path.as_os_str().as_bytes()).expect("invalid code object path");
    let mut count = 4099_u32;
    let a: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..count).map(|i| (i % 19) as f32).collect();
    let mut result = vec![0.0; count as usize];
    let mut da = Buffer::from_host(&a);
    let mut db = Buffer::from_host(&b);
    let mut output = Buffer::from_host(&result);

    let kernel = unsafe { Kernel::load(&path, c"vector_add") }.expect("load vector_add");
    let mut arguments: [*mut c_void; 4] = [
        (&raw mut da.0).cast(),
        (&raw mut db.0).cast(),
        (&raw mut output.0).cast(),
        (&raw mut count).cast(),
    ];
    unsafe {
        kernel.launch(
            Dim3::linear(count.div_ceil(256)),
            Dim3::linear(256),
            0,
            &mut arguments,
            null_mut(),
        )
    }
    .expect("enqueue vector_add");
    check("cudaDeviceSynchronize", unsafe { cudaDeviceSynchronize() });
    output.copy_to_host(&mut result);
    assert!(
        result
            .iter()
            .zip(a.iter().zip(&b))
            .all(|(&actual, (&x, &y))| actual == x + y)
    );
    kernel.close().expect("unload vector_add");
    println!("Rust HIP: {count} vector results matched the CPU");
}
