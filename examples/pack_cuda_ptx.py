"""Package the vector-add test with NVIDIA nvFatbin without requiring a GPU."""

import argparse
import ctypes as c
from pathlib import Path


def main(library, output, compression):
    lib = c.CDLL(str(Path(library).resolve()))
    handle_type = c.c_void_p
    lib.nvFatbinCreate.argtypes = [c.POINTER(handle_type), c.POINTER(c.c_char_p), c.c_size_t]
    lib.nvFatbinAddPTX.argtypes = [
        handle_type, c.c_char_p, c.c_size_t, c.c_char_p, c.c_char_p, c.c_char_p,
    ]
    lib.nvFatbinSize.argtypes = [handle_type, c.POINTER(c.c_size_t)]
    lib.nvFatbinGet.argtypes = [handle_type, c.c_void_p]
    lib.nvFatbinDestroy.argtypes = [c.POINTER(handle_type)]
    lib.nvFatbinGetErrorString.argtypes = [c.c_int]
    lib.nvFatbinGetErrorString.restype = c.c_char_p

    def check(result):
        if result:
            raise SystemExit(lib.nvFatbinGetErrorString(result).decode())

    handle = handle_type()
    flags = [b"-compress=false"] if compression == "none" else [
        b"-compress-all", f"-compress-mode={compression}".encode(),
    ]
    options = (c.c_char_p * len(flags))(*flags)
    check(lib.nvFatbinCreate(c.byref(handle), options, 1))
    try:
        source = Path(__file__).with_name("vector_add.ptx").read_bytes() + b"\0"
        check(lib.nvFatbinAddPTX(handle, source, len(source), b"70", b"vector_add", None))
        size = c.c_size_t()
        check(lib.nvFatbinSize(handle, c.byref(size)))
        image = c.create_string_buffer(size.value)
        check(lib.nvFatbinGet(handle, image))
        Path(output).write_bytes(image.raw)
    finally:
        check(lib.nvFatbinDestroy(c.byref(handle)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("output")
    parser.add_argument("--compression", choices=["none", "speed", "balance", "size"], default="none")
    args = parser.parse_args()
    main(args.library, args.output, args.compression)
