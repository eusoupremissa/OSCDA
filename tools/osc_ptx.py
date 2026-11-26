#!/usr/bin/env python3
"""Compile PTX from text, a CUDA fatbin, or a CUDA host ELF to AMDGPU."""

import argparse
import ctypes
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


FATBIN_MAGIC = 0xBA55ED50
FATBIN_COMPRESSION_MASK = 0xB000
FATBIN_LZ4 = 0x2000
FATBIN_ZSTD = 0x8000
MAX_PTX_SIZE = 64 * 1024 * 1024
ELF_SECTION = struct.Struct("<IIQQQQIIQQ")


def checked_slice(data, offset, size, description):
    if offset > len(data) or size > len(data) - offset:
        raise ValueError(f"truncated {description}")
    return data[offset:offset + size]


def elf_fatbin(data):
    if len(data) < 64 or data[4:7] != b"\x02\x01\x01":
        raise ValueError("expected a little-endian 64-bit host ELF")
    if struct.unpack_from("<H", data, 18)[0] == 190:
        raise ValueError("NVIDIA cubin contains machine code and cannot be translated as PTX")
    section_offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count, names_index = struct.unpack_from("<HHH", data, 58)
    if not section_offset or stride != ELF_SECTION.size or not 0 < names_index < count:
        raise ValueError("unsupported ELF section table")
    table = checked_slice(data, section_offset, stride * count, "ELF section table")
    names = ELF_SECTION.unpack_from(table, names_index * stride)
    if names[1] != 3:
        raise ValueError("invalid ELF section-name table")
    strings = checked_slice(data, names[4], names[5], "ELF section names")
    candidates = []
    for section in ELF_SECTION.iter_unpack(table):
        name, kind, _, _, offset, size, *_ = section
        if kind == 1 and strings.startswith(b".nv_fatbin\x00", name):
            candidates.append((offset, size))
    if len(candidates) != 1:
        raise ValueError("expected one .nv_fatbin section in the host ELF")
    return checked_slice(data, *candidates[0], "ELF .nv_fatbin section")


def decompress_ptx(payload, size, compression):
    if not 0 < size <= MAX_PTX_SIZE:
        raise ValueError("invalid decoded PTX size")
    if compression == FATBIN_LZ4:
        library = "liblz4.so.1"
    elif compression == FATBIN_ZSTD:
        library = "libzstd.so.1"
    else:
        raise ValueError(f"unsupported fatbin PTX compression flags 0x{compression:x}")
    try:
        codec = ctypes.CDLL(library)
    except OSError as error:
        raise ValueError(f"compressed PTX requires {library}: {error}") from error

    output = ctypes.create_string_buffer(size)
    if compression == FATBIN_LZ4:
        decompress = codec.LZ4_decompress_safe
        decompress.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        decompress.restype = ctypes.c_int
        decoded = decompress(payload, output, len(payload), size)
    else:
        decompress = codec.ZSTD_decompress
        decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        decompress.restype = ctypes.c_size_t
        decoded = decompress(output, size, payload, len(payload))
    if decoded != size:
        raise ValueError("invalid compressed PTX data or decoded size mismatch")
    return output.raw


def fatbin_images(data):
    position = 0
    while position < len(data):
        header = checked_slice(data, position, 16, "fatbin header")
        magic, version, header_size, body_size = struct.unpack("<IHHQ", header)
        if (magic, version, header_size) != (FATBIN_MAGIC, 1, 16) or not body_size:
            raise ValueError("unsupported CUDA fatbin header")
        position += header_size
        end = position + body_size
        if end > len(data):
            raise ValueError("truncated fatbin body")
        while position < end:
            if end - position < 16:
                raise ValueError("truncated fatbin entry")
            kind, _, entry_size, payload_size = struct.unpack_from("<HHIQ", data, position)
            if entry_size < 16 or entry_size + payload_size > end - position:
                raise ValueError("invalid fatbin entry length")
            payload = position + entry_size
            if kind == 1:
                if entry_size < 48 or not payload_size:
                    raise ValueError("invalid fatbin PTX entry")
                compressed_size = struct.unpack_from("<I", data, position + 16)[0]
                minor, major = struct.unpack_from("<HH", data, position + 24)
                flags = struct.unpack_from("<Q", data, position + 40)[0]
                compression = flags & FATBIN_COMPRESSION_MASK
                if compressed_size or compression:
                    if (entry_size < 64 or not 0 < compressed_size <= payload_size or
                            compressed_size > MAX_PTX_SIZE):
                        raise ValueError("invalid compressed PTX length")
                    decoded_size = struct.unpack_from("<Q", data, position + 56)[0]
                    raw = decompress_ptx(data[payload:payload + compressed_size], decoded_size,
                                         compression)
                else:
                    raw = data[payload:payload + payload_size]
                ptx = raw.rstrip(b"\x00").decode("utf-8")
                if "\x00" in ptx or not ptx.strip():
                    raise ValueError("invalid text in fatbin PTX image")
                # nvFatbin can move the version directive from the payload to metadata.
                prefix = re.sub(r"/\*.*?\*/|//[^\n]*", " ", ptx, flags=re.S).lstrip()
                yield f".version {major}.{minor}\n{ptx}" if prefix.startswith(".target") else ptx
            position = payload + payload_size


def ptx_images(data):
    if data.startswith(b"\x7fELF"):
        yield from fatbin_images(elf_fatbin(data))
    elif data.startswith(struct.pack("<I", FATBIN_MAGIC)):
        yield from fatbin_images(data)
    else:
        yield data.decode("utf-8")


def read_ptx(data):
    images = ptx_images(data)
    image = next(images, None)
    if image is None:
        raise ValueError("fatbin contains no PTX image; a cubin alone cannot be translated")
    if next(images, None) is not None:
        raise ValueError("fatbin contains multiple PTX images; select an entry with --kernel")
    return image


NAME = r"[A-Za-z_][A-Za-z0-9_]*"
REG = rf"%{NAME}"
TYPES = {"u32": "unsigned int", "u64": "unsigned long long",
         "f32": "float", "pred": "bool"}
BIT_TYPES = {"b32": "u32", "b64": "u64"}
SPECIAL = {f"%{reg}.{axis}": f"__builtin_amdgcn_{builtin}_{axis}()"
           for reg, builtin in (("tid", "workitem_id"), ("ctaid", "workgroup_id"))
           for axis in "xyz"}
SPECIAL.update({f"%ntid.{axis}": f"dispatch[{index}]"
                for index, axis in enumerate("xyz", 2)})
BINARY = {
    "add.u32": ("u32", "u32", "+"),
    "add.u64": ("u64", "u64", "+"),
    "mul.lo.u32": ("u32", "u32", "*"),
    "mul.wide.u32": ("u64", "u32", "*"),
    "add.rn.f32": ("f32", "f32", "+"),
    "mul.rn.f32": ("f32", "f32", "*"),
    "setp.ge.u32": ("pred", "u32", ">="),
    "setp.lt.f32": ("pred", "f32", "<"),
}


def validate_flow(flow, labels):
    successors = []
    for index, (statement, _, _, fallthrough, target) in enumerate(flow):
        edges = [index + 1] if fallthrough else []
        if target is not None:
            if target not in labels:
                raise ValueError(f"{statement!r}: unknown label {target}")
            edges.append(labels[target])
        successors.append(edges)

    # A register is initialized at a join only if every incoming path writes it.
    incoming = {0: set()}
    pending = [0]
    while pending:
        index = pending.pop()
        _, _, writes, _, _ = flow[index]
        outgoing = incoming[index] | writes
        for successor in successors[index]:
            merged = incoming[successor] & outgoing if successor in incoming else outgoing
            if successor not in incoming or merged != incoming[successor]:
                incoming[successor] = merged
                pending.append(successor)
    for index, (statement, reads, _, _, _) in enumerate(flow):
        if index in incoming and (missing := reads - incoming[index]):
            raise ValueError(f"{statement!r}: possibly uninitialized register {min(missing)}")


ENTRY = re.compile(rf"\s*(?:\.visible\s+)?\.entry\s+({NAME})"
                   r"\s*\(([^()]*)\)\s*\{([^{}]*)\}")


def parse_entries(source):
    source = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S).rstrip()
    header = re.match(r"\s*\.version\s+[1-8]\.\d+\s+\.target\s+sm_\d+"
                      r"\s+\.address_size\s+64\b", source)
    if not header:
        raise ValueError("expected a standard 64-bit PTX module header")
    position = header.end()
    entries = {}
    while position < len(source):
        entry = ENTRY.match(source, position)
        if not entry:
            raise ValueError("expected a PTX .entry; unsupported module declaration or entry syntax")
        name, signature, body = entry.groups()
        if name in entries:
            raise ValueError(f"duplicate PTX entry {name}")
        entries[name] = (signature, body)
        position = entry.end()
    if not entries:
        raise ValueError("PTX module contains no .entry")
    return entries


def translate_selected(data, name):
    selected = None
    has_ptx = False
    for source in ptx_images(data):
        has_ptx = True
        entries = parse_entries(source)
        if name in entries:
            if selected is not None:
                raise ValueError(f"PTX entry {name} is ambiguous across fatbin images")
            selected = entries[name]
    if not has_ptx:
        raise ValueError("fatbin contains no PTX image; a cubin alone cannot be translated")
    if selected is None:
        raise ValueError(f"PTX entry {name} not found")
    return translate_entries({name: selected})


def translate(source):
    return translate_entries(parse_entries(source))


def translate_entries(entries):
    return "\n".join([
        "typedef __attribute__((address_space(1))) float global_float;",
        "typedef __attribute__((address_space(3))) float shared_float;",
        *(translate_entry(name, signature, body, index)
          for index, (name, (signature, body)) in enumerate(entries.items())),
    ])


def translate_entry(name, signature, body, index):
    params = {}
    for parameter in signature.split(",") if signature.strip() else ():
        match = re.fullmatch(rf"\s*\.param\s+\.(u32|u64|f32|b32|b64)\s+({NAME})\s*",
                             parameter)
        if not match or match[2] in params:
            raise ValueError(f"invalid parameter: {parameter.strip()}")
        params[match[2]] = BIT_TYPES.get(match[1], match[1])

    registers = {}
    shared = {}
    labels = {}
    code = []
    flow = []

    def read(operand, kind):
        if registers.get(operand) == kind:
            reads.add(operand)
            return "r_" + operand[1:]
        if kind == "f32" and re.fullmatch(r"0[fF][0-9a-fA-F]{8}", operand):
            return f"__builtin_bit_cast(float, 0x{operand[2:]}u)"
        if kind in ("u32", "u64") and re.fullmatch(r"0x[0-9a-fA-F]+|[0-9]+", operand):
            value = int(operand, 16 if operand.startswith("0x") else 10)
            if value < 1 << int(kind[1:]):
                return str(value) + ("u" if kind == "u32" else "ull")
        raise ValueError(f"expected .{kind} operand, got {operand}")

    def write(register, kind, expression):
        if registers.get(register) != kind:
            raise ValueError(f"expected declared .{kind} destination, got {register}")
        writes.add(register)
        return f"r_{register[1:]} = {expression};"

    def address(operand):
        if not re.fullmatch(rf"\[\s*{REG}\s*\]", operand):
            raise ValueError("expected an address in a 64-bit register")
        return read(operand[1:-1].strip(), "u64")

    statements = [part.strip() for part in body.split(";")]
    if statements[-1] or len(statements) < 2 or statements[-2] != "ret":
        raise ValueError("kernel must end with ret;")
    for statement in statements[:-1]:
        try:
            while match := re.match(rf"({NAME})\s*:\s*", statement):
                label = match[1]
                if label in labels:
                    raise ValueError(f"duplicate label {label}")
                labels[label] = len(flow)
                code.append(f"label_{label}: ;")
                flow.append((label, set(), set(), True, None))
                statement = statement[match.end():]
            declaration = re.fullmatch(rf"\.reg\s+\.(\w+)\s+({REG})(?:<(\d+)>)?",
                                       statement)
            if declaration and not code:
                kind, register, count = declaration.groups()
                kind = BIT_TYPES.get(kind, kind)
                if kind not in TYPES or (count and not 1 <= int(count) <= 4096):
                    raise ValueError("unsupported register declaration")
                names = [register + str(i) for i in range(int(count))] if count else [register]
                for register in names:
                    if register in registers:
                        raise ValueError(f"duplicate register {register}")
                    registers[register] = kind
                continue

            tile = re.fullmatch(rf"\.shared\s+\.align\s+(4|8|16)\s+\.f32\s+({NAME})\[(\d+)\]",
                                statement)
            if tile and not code:
                alignment, symbol, count = tile.groups()
                if symbol in shared or not 1 <= int(count) <= 16384:
                    raise ValueError("invalid shared array")
                shared[symbol] = f"alignas({alignment}) __attribute__((shared)) float s_{symbol}[{int(count)}];"
                continue

            reads, writes = set(), set()
            control = re.fullmatch(rf"(?:@(!?)({REG})\s+)?(ret|bra(?:\.uni)?\s+({NAME}))",
                                   statement)
            if control:
                negated, predicate, _, target = control.groups()
                prefix = f"if ({negated}{read(predicate, 'pred')}) " if predicate else ""
                code.append(prefix + (f"goto label_{target};" if target else "return;"))
                flow.append((statement, reads, writes, bool(predicate), target))
                continue
            instruction = statement.split(None, 1)
            op = instruction[0] if instruction else ""
            args = [arg.strip() for arg in instruction[1].split(",")] if len(instruction) == 2 else []
            if op in BINARY and len(args) == 3:
                dst, src, operator = BINARY[op]
                left, right = (read(arg, src) for arg in args[1:])
                code.append(write(args[0], dst, f"({TYPES[dst] if dst != 'pred' else TYPES[src]})"
                                  f"{left} {operator} {right}"))
            elif op == "mad.lo.u32" and len(args) == 4:
                left, right, addend = (read(arg, "u32") for arg in args[1:])
                code.append(write(args[0], "u32", f"{left} * {right} + {addend}"))
            elif op in ("mov.u32", "mov.u64", "mov.f32", "cvta.to.global.u64") and len(args) == 2:
                kind = op.rsplit(".", 1)[1]
                if op == "mov.u32" and args[1] in SPECIAL:
                    value = SPECIAL[args[1]]
                elif op == "mov.u64" and args[1] in shared:
                    value = f"(unsigned long long)(shared_float *)s_{args[1]}"
                else:
                    value = read(args[1], kind)
                code.append(write(args[0], kind, value))
            elif op in ("ld.param.u32", "ld.param.u64", "ld.param.f32") and len(args) == 2:
                kind = op.rsplit(".", 1)[1]
                match = re.fullmatch(rf"\[\s*({NAME})\s*\]", args[1])
                if not match or params.get(match[1]) != kind:
                    raise ValueError(f"expected .{kind} parameter")
                code.append(write(args[0], kind, "p_" + match[1]))
            elif op in ("ld.global.f32", "ld.shared.f32") and len(args) == 2:
                space = op.split(".")[1]
                code.append(write(args[0], "f32", f"*(const {space}_float *){address(args[1])}"))
            elif op in ("st.global.f32", "st.shared.f32") and len(args) == 2:
                space = op.split(".")[1]
                code.append(f"*({space}_float *){address(args[0])} = {read(args[1], 'f32')};")
            elif op == "bar.sync" and args == ["0"]:
                # shortcut: one converged full-workgroup barrier, until named barriers are lowered.
                code.extend(['__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");',
                             "__builtin_amdgcn_s_barrier();",
                             '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'])
            else:
                raise ValueError("unsupported instruction or operand count")
            flow.append((statement, reads, writes, True, None))
        except ValueError as error:
            raise ValueError(f"{statement!r}: {error}") from error

    validate_flow(flow, labels)
    signature = ", ".join(f"{TYPES[kind]} p_{param}" for param, kind in params.items())
    declarations = [f"{TYPES[kind]} r_{register[1:]};" for register, kind in registers.items()]
    # Treat the PTX name as data, even if it coincides with a C++ keyword.
    return "\n".join([
        f'extern "C" __attribute__((global)) void osc_entry_{index}({signature}) __asm__("{name}");',
        f'extern "C" __attribute__((global)) void osc_entry_{index}({signature}) {{',
        "    const auto *dispatch = (const __attribute__((address_space(4))) unsigned short *)"
        "__builtin_amdgcn_dispatch_ptr();",
        *("    " + line for line in [*shared.values(), *declarations, *code]),
        "}", "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="PTX, CUDA fatbin, or CUDA host ELF")
    parser.add_argument("--kernel", help="translate only this PTX entry")
    parser.add_argument("--arch", required=True, help="AMD target, e.g. gfx1032")
    parser.add_argument("--clang", default=os.environ.get("OSC_CLANG") or "clang++",
                        help="path to ROCm LLVM clang++")
    parser.add_argument("-o", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"gfx[0-9a-f]+(?::sramecc[+-])?(?::xnack[+-])?", args.arch):
        parser.error("--arch must be an AMD gfx target")
    if args.source.resolve() == args.o.resolve():
        parser.error("input and output must be different files")
    try:
        data = args.source.read_bytes()
        hip = translate_selected(data, args.kernel) if args.kernel else translate(read_ptx(data))
        with tempfile.TemporaryDirectory(dir=args.o.parent) as directory:
            output = Path(directory) / "kernel.hsaco"
            subprocess.run([
                args.clang, "-x", "hip", "--cuda-device-only", "--no-gpu-bundle-output",
                f"--offload-arch={args.arch}", "-nogpuinc", "-nogpulib", "-mcode-object-version=4",
                "-O2", "-ffp-contract=off", "-fdenormal-fp-math=ieee", "-", "-o", str(output),
            ], input=hip, text=True, check=True)
            output.replace(args.o)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"osc-ptx: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
