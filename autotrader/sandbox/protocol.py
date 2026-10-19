"""Framing and safe payload encoding between the parent and the sandbox worker."""

import base64
import io
import struct

import pandas as pd
import pyarrow as pa

MAX_FRAME = 512 * 1024 * 1024
_LEN = struct.Struct(">I")


def write_frame(stream, payload: bytes) -> None:
    stream.write(_LEN.pack(len(payload)) + payload)
    stream.flush()


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("sandbox stream closed")
        buf += chunk
    return bytes(buf)


def frame_length(header: bytes) -> int:
    (n,) = _LEN.unpack(header)
    if n > MAX_FRAME:
        raise ValueError(f"frame of {n} bytes exceeds the limit")
    return n


def encode_frame(df: pd.DataFrame | None) -> str | None:
    """DataFrame -> base64 Arrow IPC (index and dtypes preserved)."""
    if df is None:
        return None
    table = pa.Table.from_pandas(df, preserve_index=True)
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return base64.b64encode(sink.getvalue()).decode()


def decode_frame(data: str | None) -> pd.DataFrame | None:
    if data is None:
        return None
    with pa.ipc.open_stream(base64.b64decode(data)) as reader:
        return reader.read_all().to_pandas()
