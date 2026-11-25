import ctypes
import importlib.util
from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("osc_ptx", ROOT / "tools/osc_ptx.py")
PTX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PTX)
VECTOR_ADD = (ROOT / "examples/vector_add.ptx").read_text(encoding="utf-8")
MATMUL = (ROOT / "examples/matmul.ptx").read_text(encoding="utf-8")


def fatbin_entry(kind, payload, compressed_size=0, flags=0, decoded_size=0):
    header = bytearray(88)
    struct.pack_into("<HHIQIIHHI", header, 0, kind, 0x101, len(header), len(payload),
                     compressed_size, len(header) - 16, 0, 7, 70)
    struct.pack_into("<Q", header, 40, flags)
    struct.pack_into("<Q", header, 56, decoded_size)
    return bytes(header) + payload


def fatbin(*entries):
    body = b"".join(entries)
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body


def host_elf(payload):
    names = b"\0.shstrtab\0.nv_fatbin\0"
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<Q", header, 40, 64)
    struct.pack_into("<HHH", header, 58, 64, 3, 1)
    section = struct.Struct("<IIQQQQIIQQ")
    offset = 64 + section.size * 3
    return (bytes(header) + bytes(section.size)
            + section.pack(1, 3, 0, 0, offset, len(names), 0, 0, 1, 0)
            + section.pack(11, 1, 0, 0, offset + len(names), len(payload), 0, 0, 8, 0)
            + names + payload)


class PtxTest(unittest.TestCase):
    def compressed_ptx(self, codec, flags, payload):
        try:
            ctypes.CDLL(codec)
        except OSError:
            self.skipTest(f"{codec} unavailable")
        source = ".target sm_70\n.address_size 64\n.entry sample() { ret; }\n"
        raw = source.encode() + b"\0"
        encoded = payload(raw)
        padding = bytes(-len(encoded) % 8)
        entry = fatbin_entry(1, encoded + padding, len(encoded), flags, len(raw))
        image = fatbin(entry)
        for data in (image, host_elf(image)):
            self.assertEqual(PTX.read_ptx(data), ".version 7.0\n" + source)
        invalid = fatbin(fatbin_entry(1, encoded, len(encoded), flags, len(raw) + 1))
        with self.assertRaisesRegex(ValueError, "decoded size mismatch"):
            PTX.read_ptx(invalid)

    def test_lz4_ptx(self):
        self.compressed_ptx("liblz4.so.1", 0x2000,
                            lambda raw: bytes((0xf0, len(raw) - 15)) + raw)

    def test_zstd_ptx(self):
        self.compressed_ptx("libzstd.so.1", 0x8000,
                            lambda raw: b"\x28\xb5\x2f\xfd\x20" + bytes((len(raw),))
                            + ((len(raw) << 3) | 1).to_bytes(3, "little") + raw)

    def test_vector_add(self):
        kernel = PTX.translate(VECTOR_ADD)
        self.assertIn('__asm__("vector_add")', kernel)
        self.assertIn("if (r_p) return;", kernel)
        self.assertIn("r_rd3 = (unsigned long long)r_r4 * 4u;", kernel)
        self.assertIn("*(global_float *)r_rd6 = r_f2;", kernel)

    def test_rejects_unsupported_or_invalid_ptx(self):
        for original, replacement in (
            ("add.rn.f32", "add.ftz.f32"),
            ("%f0, %f1;", "%f0, %f2;"),
            ("[%rd4]", "[%r4]"),
            ("ret;\n}", "ret\n}"),
        ):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                PTX.translate(VECTOR_ADD.replace(original, replacement))

    def test_shared_matmul(self):
        kernel = PTX.translate(MATMUL)
        self.assertIn("__attribute__((shared)) float s_tile_a[256];", kernel)
        self.assertIn("*(shared_float *)r_d6 = r_left;", kernel)
        self.assertIn("if (!r_p2) goto label_dot;", kernel)
        self.assertEqual(kernel.count("__builtin_amdgcn_s_barrier();"), 2)

    def test_rejects_invalid_control_flow_and_barriers(self):
        for original, replacement, error in (
            ("bra dot;", "bra missing;", "unknown label"),
            ("dot:", "tile_loop:", "duplicate label"),
            ("mov.f32 %left, 0f00000000;", "", "possibly uninitialized register %left"),
            ("bar.sync 0;", "bar.sync 1;", "unsupported instruction"),
            ("bar.sync 0;", "@%p0 bar.sync 0;", "unsupported instruction"),
        ):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, error):
                PTX.translate(MATMUL.replace(original, replacement, 1))

    def test_ptx_from_fatbin_and_elf(self):
        payload = MATMUL.removeprefix(".version 7.0").encode() + b"\0" * 4
        binary = fatbin(fatbin_entry(2, b"cubin"), fatbin_entry(1, payload))
        for data in (binary, host_elf(binary)):
            self.assertEqual(PTX.translate(PTX.read_ptx(data)), PTX.translate(MATMUL))
        embedded_version = fatbin(fatbin_entry(1, MATMUL.encode()))
        self.assertEqual(PTX.read_ptx(embedded_version), MATMUL)

    def test_rejects_bad_binary_inputs(self):
        ptx = fatbin_entry(1, VECTOR_ADD.encode())
        for data, error in (
            (fatbin(ptx)[:-1], "truncated fatbin body"),
            (fatbin(ptx[:80]), "invalid fatbin entry length"),
            (host_elf(fatbin(ptx))[:-1], "truncated ELF"),
            (fatbin(ptx, ptx), "multiple PTX"),
            (fatbin(fatbin_entry(2, b"cubin")), "no PTX"),
            (fatbin(fatbin_entry(1, b"encoded", 8, 0x2000, 16)), "invalid compressed PTX length"),
            (fatbin(fatbin_entry(1, b"encoded", 7, 0x8000, 1 << 27)), "invalid decoded PTX size"),
            (fatbin(fatbin_entry(1, b"encoded", 7, 0x1000, 16)), "unsupported fatbin PTX compression"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                PTX.read_ptx(data)


if __name__ == "__main__":
    unittest.main()
