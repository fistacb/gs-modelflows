"""Lightweight, fast NumPy-only binary little-endian 3DGS PLY reader & writer."""
from pathlib import Path
import numpy as np

PLY_TYPE_MAP = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2",
    "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4",
    "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4",
    "double": "<f8", "float64": "<f8",
}

NP_TO_PLY_TYPE = {
    np.dtype("i1"): "char",
    np.dtype("u1"): "uchar",
    np.dtype("<i2"): "short",
    np.dtype("<u2"): "ushort",
    np.dtype("<i4"): "int",
    np.dtype("<u4"): "uint",
    np.dtype("<f4"): "float",
    np.dtype("<f8"): "double",
}


def read_ply(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read binary_little_endian PLY vertex element into a structured NumPy array."""
    path = Path(path)
    with open(path, "rb") as f:
        header_lines = []
        vertex_count = None
        in_vertex = False
        props = []
        fmt = None
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            line = raw.decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = (parts[1] == "vertex")
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"List property not supported in vertex element: {line}")
                props.append((parts[2], PLY_TYPE_MAP[parts[1]]))

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {fmt}")
        if vertex_count is None or not props:
            raise ValueError(f"Missing vertex element or properties in {path}")

        dtype = np.dtype(props)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
        if len(data) != vertex_count:
            raise ValueError(f"Expected {vertex_count} vertices, got {len(data)}")
    return data, header_lines


def write_ply(path: str | Path, data: np.ndarray) -> None:
    """Write a structured NumPy array as a binary_little_endian PLY file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(data)}",
    ]
    for name in data.dtype.names:
        dt = data.dtype.fields[name][0]
        ply_t = NP_TO_PLY_TYPE[np.dtype(dt).newbyteorder("<") if dt.itemsize > 1 else dt]
        lines.append(f"property {ply_t} {name}")
    lines.append("end_header")
    header_bytes = ("\n".join(lines) + "\n").encode("ascii")

    # Ensure little-endian contiguous buffer
    le_dtype = np.dtype([
        (name, data.dtype.fields[name][0].newbyteorder("<") if data.dtype.fields[name][0].itemsize > 1 else data.dtype.fields[name][0])
        for name in data.dtype.names
    ])
    out = np.ascontiguousarray(data.astype(le_dtype, copy=False))
    with open(path, "wb") as f:
        f.write(header_bytes)
        out.tofile(f)
