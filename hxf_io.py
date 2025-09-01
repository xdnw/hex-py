# hxf_io.py
from __future__ import annotations
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

HXF_MAGIC = b"HXF1"
HXF_VERSION = 1
ORIENT_POINTY = 1  # 0=flat, 1=pointy
LE = "<"  # little-endian for struct


def _pack_uints_lsb(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned integers (values >=0) using LSB-first bit packing."""
    if bits <= 0:
        return b""
    mask = (1 << bits) - 1
    out = bytearray()
    acc = 0
    acc_bits = 0
    for v in map(int, values):
        acc |= (v & mask) << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits > 0:
        out.append(acc & 0xFF)
    return bytes(out)


def _unpack_uints_lsb(data: bytes, count: int, bits: int) -> np.ndarray:
    """Inverse of _pack_uints_lsb."""
    if bits <= 0:
        return np.zeros(count, dtype=np.uint32)
    mask = (1 << bits) - 1
    vals = np.empty(count, dtype=np.uint32)
    acc = 0
    acc_bits = 0
    i = 0
    for b in data:
        acc |= int(b) << acc_bits
        acc_bits += 8
        while acc_bits >= bits and i < count:
            vals[i] = acc & mask
            acc >>= bits
            acc_bits -= bits
            i += 1
        if i >= count:
            break
    if i != count:
        raise ValueError("Not enough data to unpack requested count.")
    return vals


def write_hxf(
    path: str | Path,
    centers_xy: np.ndarray,
    *,
    epsg: int = 8857,
    radius_m: float,
    orientation: int = ORIENT_POINTY,
    class_is_land: np.ndarray | None = None,  # bool array: True=land, False=sea
    continent_names: list[str] | None = None,  # list parallel to centers or None
) -> None:
    """
    Write a compact HXF v1 file.

    centers_xy: shape (N, 2), float32/float64 in Equal Earth meters
    class_is_land: optional bool array of length N
    continent_names: optional list[str] of length N (include 'sea' where appropriate)
    """
    path = Path(path)
    centers_xy = np.asarray(centers_xy, dtype=np.float32)
    if centers_xy.ndim != 2 or centers_xy.shape[1] != 2:
        raise ValueError("centers_xy must be (N,2)")

    n = centers_xy.shape[0]
    has_class = class_is_land is not None
    has_cont = continent_names is not None

    # Header basic info
    bbox = np.array(
        [centers_xy[:, 0].min(), centers_xy[:, 1].min(),
         centers_xy[:, 0].max(), centers_xy[:, 1].max()],
        dtype=np.float32,
    )

    # Prepare class bits
    class_bytes = b""
    if has_class:
        class_is_land = np.asarray(class_is_land, dtype=bool)
        if class_is_land.shape[0] != n:
            raise ValueError("class_is_land length mismatch")
        # 1 bit per cell, bit=1 -> land, bit=0 -> sea
        class_bytes = np.packbits(class_is_land.astype(np.uint8), bitorder="little").tobytes()

    # Prepare continent coding
    cont_labels: list[str] = []
    cont_codes_bytes = b""
    cont_bits = 0
    if has_cont:
        if len(continent_names) != n:
            raise ValueError("continent_names length mismatch")
        # Build label table (stable order of first appearance)
        seen = {}
        cont_labels = []
        codes = np.empty(n, dtype=np.uint32)
        for i, name in enumerate(continent_names):
            s = "" if name is None else str(name)
            if s not in seen:
                seen[s] = len(cont_labels)
                cont_labels.append(s)
            codes[i] = seen[s]
        k = max(1, len(cont_labels))
        cont_bits = max(1, math.ceil(math.log2(k))) if k > 1 else 1
        cont_codes_bytes = _pack_uints_lsb(codes, cont_bits)

    # Compose header
    # magic(4) | version(u16) | endian(u8=1) | orient(u8) | n(u32) | epsg(i32) |
    # radius(f32) | bbox(4*f32) | has_class(u8) | has_cont(u8)
    header = bytearray()
    header += HXF_MAGIC
    header += struct.pack(LE + "HBB", HXF_VERSION, 1, orientation)
    header += struct.pack(LE + "Ii", n, epsg)
    header += struct.pack(LE + "f", float(radius_m))
    header += struct.pack(LE + "4f", *bbox.tolist())
    header += struct.pack(LE + "BB", 1 if has_class else 0, 1 if has_cont else 0)

    # Centers block
    centers_bytes = centers_xy.astype("<f4", copy=False).tobytes(order="C")

    # Class block length (so readers can skip quickly)
    # continent block: n_labels(u16) + [len(u16)+utf8]* + bits(u8) + code_len(u32) + codes
    out = bytearray()
    out += header
    out += centers_bytes

    if has_class:
        out += struct.pack(LE + "I", len(class_bytes))
        out += class_bytes
    else:
        out += struct.pack(LE + "I", 0)

    if has_cont:
        # labels
        out += struct.pack(LE + "H", len(cont_labels))
        for label in cont_labels:
            b = label.encode("utf-8")
            out += struct.pack(LE + "H", len(b))
            out += b
        out += struct.pack(LE + "B", cont_bits)
        out += struct.pack(LE + "I", len(cont_codes_bytes))
        out += cont_codes_bytes
    else:
        out += struct.pack(LE + "H", 0)  # n_labels=0
    print(f"Writing HXF: {path}, cells={n}, class={'yes' if has_class else 'no'}, continents={'yes' if has_cont else 'no'}")
    Path(path).write_bytes(out)


def read_hxf(path: str | Path) -> dict[str, Any]:
    """Read HXF v1 and return a dict with centers, radius, epsg, class mask, continents, codes."""
    data = Path(path).read_bytes()
    mv = memoryview(data)
    off = 0

    if mv[off:off+4].tobytes() != HXF_MAGIC:
        raise ValueError("Not an HXF file")
    off += 4

    version, endian_flag, orientation = struct.unpack_from(LE + "HBB", mv, off)
    off += struct.calcsize(LE + "HBB")
    if version != HXF_VERSION or endian_flag != 1:
        raise ValueError("Unsupported HXF version or endianness")

    n, epsg = struct.unpack_from(LE + "Ii", mv, off)
    off += struct.calcsize(LE + "Ii")
    (radius_m,) = struct.unpack_from(LE + "f", mv, off)
    off += struct.calcsize(LE + "f")
    bbox = struct.unpack_from(LE + "4f", mv, off)
    off += struct.calcsize(LE + "4f")
    has_class, has_cont = struct.unpack_from(LE + "BB", mv, off)
    off += struct.calcsize(LE + "BB")

    # Centers
    n_float32 = n * 2
    n_bytes = n_float32 * 4
    centers = np.frombuffer(mv[off:off + n_bytes], dtype="<f4").reshape(n, 2)
    off += n_bytes

    # Class
    (class_len,) = struct.unpack_from(LE + "I", mv, off)
    off += 4
    class_mask = None
    if has_class and class_len > 0:
        class_bytes = mv[off:off + class_len].tobytes()
        off += class_len
        # little-order packbits -> unpackbits with bitorder="little"
        bits = np.unpackbits(np.frombuffer(class_bytes, dtype=np.uint8), bitorder="little")
        class_mask = bits[:n].astype(bool)
    else:
        if class_len:
            off += class_len

    # Continents
    (n_labels,) = struct.unpack_from(LE + "H", mv, off)
    off += 2
    continent_labels: list[str] = []
    continent_codes = None
    if has_cont and n_labels > 0:
        for _ in range(n_labels):
            (ln,) = struct.unpack_from(LE + "H", mv, off)
            off += 2
            label = mv[off:off + ln].tobytes().decode("utf-8")
            off += ln
            continent_labels.append(label)
        (cont_bits,) = struct.unpack_from(LE + "B", mv, off)
        off += 1
        (codes_len,) = struct.unpack_from(LE + "I", mv, off)
        off += 4
        codes_bytes = mv[off:off + codes_len].tobytes()
        off += codes_len
        continent_codes = _unpack_uints_lsb(codes_bytes, n, cont_bits).astype(np.uint32)

    return {
        "version": version,
        "epsg": epsg,
        "orientation": orientation,
        "radius_m": float(radius_m),
        "bbox": tuple(float(v) for v in bbox),
        "centers": centers,                 # (N,2) float32
        "class_is_land": class_mask,        # (N,) bool or None
        "continent_labels": continent_labels if n_labels else None,
        "continent_codes": continent_codes, # (N,) uint32 or None
    }