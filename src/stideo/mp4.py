"""拼接 FFmpeg 生成的单轨 MP4。保留每种解码配置和每帧显示时间，音视频交错写入。"""

import struct
import sys
from array import array
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import accumulate, chain, groupby
from typing import TYPE_CHECKING, BinaryIO
lazy from pathlib import Path

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def box(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(data), kind) + data


def boxes(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    offset = 0
    while offset < len(data):
        size, kind = struct.unpack_from(">I4s", data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError("MP4 元数据损坏")
        yield kind, data[offset + 8 : offset + size]
        offset += size


def child(data: bytes, *names: bytes) -> bytes:
    for name in names:
        data = next((v for k, v in boxes(data) if k == name), None)
        if data is None:
            raise ValueError(f"MP4 缺少 {name!r}")
    return data


def replace(data: bytes, changes: dict[bytes, bytes | None]) -> bytes:
    remaining = changes.copy()
    parts = []
    for kind, original in boxes(data):
        value = remaining.pop(kind, original)
        if value is not None:
            parts.append(box(kind, value))
    parts.extend(box(k, v) for k, v in remaining.items() if v is not None)
    return b"".join(parts)


def headers(path: Path) -> tuple[bytes, bytes]:
    result = {}
    with path.open("rb") as stream:
        while header := stream.read(8):
            size, kind = struct.unpack(">I4s", header)
            header_size = 8
            if size == 1:
                (size,) = struct.unpack(">Q", stream.read(8))
                header_size = 16
            if size == 0:
                break
            if size < header_size:
                raise ValueError(f"{path.name} 的 MP4 文件头无效")
            if kind in (b"ftyp", b"moov"):
                result[kind] = stream.read(size - header_size)
            else:
                stream.seek(size - header_size, 1)
            if len(result) == 2:
                break
    if len(result) != 2:
        raise ValueError(f"{path.name} 缺少 MP4 轨道信息")
    return result[b"ftyp"], result[b"moov"]


def values(data: bytes, code: str = "I") -> array[int]:
    result = array(code)
    result.frombytes(data)
    if sys.byteorder == "little":
        result.byteswap()
    return result


def packed(numbers: Iterable[int], code: str = "I") -> bytes:
    result = array(code, numbers)
    if sys.byteorder == "little":
        result.byteswap()
    return result.tobytes()


def table(rows: list[tuple[int, ...]], form: str) -> bytes:
    return bytes(4) + struct.pack(">I", len(rows)) + b"".join(struct.pack(form, *r) for r in rows)


def timing_runs(durations: Iterable[int]) -> list[tuple[int, int]]:
    return [(sum(1 for _ in copies), value) for value, copies in groupby(durations)]


def expand(data: bytes, *, signed: bool = False) -> array[int]:
    result = array("q")
    for n, val in struct.iter_unpack(">Ii" if signed else ">II", data[8:]):
        result.extend(array("q", [val]) * n)
    return result


def time_header(data: bytes, kind: bytes, duration: int, scale_or_id: int) -> bytes:
    # mvhd/mdhd 和 tkhd 的 v1 均用 64 位创建时间、修改时间和时长。
    old64 = data[0] == 1
    pos = 4 + (16 if old64 else 8)
    gap = 8 if kind == b"tkhd" else 4
    tail = data[pos + gap + (8 if old64 else 4) :]
    created, modified = struct.unpack_from(">QQ" if old64 else ">II", data, 4)
    return (
        bytes([1])
        + data[1:4]
        + struct.pack(">QQI", created, modified, scale_or_id)
        + (bytes(4) if kind == b"tkhd" else b"")
        + struct.pack(">Q", duration)
        + tail
    )


@dataclass
class Chunk:
    at: float
    path: Path
    source: int
    size: int
    count: int
    description: int
    offset: int = 0


class Track:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.ftyp, self.moov = headers(path)
        self.trak = child(self.moov, b"trak")
        self.mdia = child(self.trak, b"mdia")
        self.minf = child(self.mdia, b"minf")
        self.stbl = child(self.minf, b"stbl")
        self.atoms = dict(boxes(self.stbl))
        mdhd = child(self.mdia, b"mdhd")
        (self.scale,) = struct.unpack_from(">I", mdhd, 20 if mdhd[0] else 12)
        size, count = struct.unpack_from(">II", self.atoms[b"stsz"], 4)
        self.sizes = array("I", [size]) * count if size else values(self.atoms[b"stsz"][12:])
        self.delta = expand(self.atoms[b"stts"])
        if len(self.sizes) != count or len(self.delta) != count or not count:
            raise ValueError(f"{path.name} 的 MP4 样本数量无效")
        (self.entry,) = [box(k, v) for k, v in boxes(self.atoms[b"stsd"][8:])]

    def presentation(self, span: int) -> tuple[Iterable[int], Iterable[int]]:
        """Align the edit boundary and return local presentation offsets and key samples."""
        ctts = self.atoms.get(b"ctts")
        local = expand(ctts, signed=bool(ctts[0])) if ctts else array("q", [0]) * len(self.delta)
        if len(local) != len(self.sizes):
            raise ValueError("视频显示时间表不完整")
        # stts 按解码顺序计时，ctts 存储每个样本的显示偏移。将片头显示时刻移到零。
        first_pts = min(t + o for t, o in zip(accumulate(self.delta, initial=0), local, strict=False))
        if local[0] != first_pts:
            raise ValueError(f"{self.path.name} 开头不是独立可显示的画面")
        last = span - (sum(self.delta) - self.delta[-1])
        if last <= 0:
            raise ValueError(f"{self.path.name} 的解码时间超出片段边界")
        self.delta[-1] = last
        keys = values(self.atoms[b"stss"][8:]) if b"stss" in self.atoms else range(1, len(self.sizes) + 1)
        if not keys or keys[0] != 1:
            raise ValueError(f"{self.path.name} 片头不是关键帧")
        return (o - first_pts for o in local), keys

    def chunks(self, start: float, description: int) -> Iterator[Chunk]:
        t = self.atoms
        positions = values(t[b"co64"][8:], "Q") if b"co64" in t else values(t[b"stco"][8:])
        mapping = list(struct.iter_unpack(">III", t[b"stsc"][8:]))
        sample = row = ticks = 0
        for i, position in enumerate(positions, 1):
            pos = position
            if row + 1 < len(mapping) and i == mapping[row + 1][0]:
                row += 1
            _, num, index = mapping[row]
            if index != 1:
                raise ValueError("单段 MP4 含多套解码配置")
            end = sample + num
            while sample < end:
                first, begin, length = sample, ticks, 0
                while sample < end and length < 1024 * 1024 and ticks - begin < self.scale / 2:
                    length += self.sizes[sample]
                    ticks += self.delta[sample]
                    sample += 1
                yield Chunk(start + begin / self.scale, self.path, pos, length, sample - first, description)
                pos += length
        if sample != len(self.sizes):
            raise ValueError("MP4 分块与样本数量不一致")


def chunk_tables(chunks: list[Chunk]) -> dict[bytes, bytes | None]:
    runs = []
    for i, c in enumerate(chunks, 1):
        if not runs or runs[-1][1:] != (c.count, c.description):
            runs.append((i, c.count, c.description))
    return {
        b"stsc": table(runs, ">III"),
        b"stco": None,
        b"co64": bytes(4) + struct.pack(">I", len(chunks)) + packed((c.offset for c in chunks), "Q"),
    }


def track_header(trak: bytes, duration: int, rotation: int) -> bytes:
    tkhd = bytearray(time_header(child(trak, b"tkhd"), b"tkhd", duration, 1))
    width, height = struct.unpack_from(">II", tkhd, len(tkhd) - 8)
    one = 65536
    matrix = {
        0: (one, 0, 0, one, 0, 0),
        90: (0, -one, one, 0, 0, width),
        180: (-one, 0, 0, -one, width, height),
        270: (0, one, -one, 0, height, 0),
    }[rotation % 360]
    a, b, c, d, x, y = matrix
    struct.pack_into(">9i", tkhd, len(tkhd) - 44, a, b, 0, c, d, 0, x, y, 1 << 30)
    return bytes(tkhd)


def assemble(parts: list[tuple[Path, int]], audio_path: Path, output: Path, rotation: int) -> None:
    """Parts 中每项为单轨视频路径及目标时长（共同媒体时基的整数刻度）。"""
    entries, video_chunks = [], []
    deltas, offsets, sizes, sync = array("q"), array("q"), array("I"), array("I")
    total = 0
    tracks = ((Track(path), span) for path, span in parts)
    first, first_span = next(tracks)
    for v, span in chain([(first, first_span)], tracks):
        if v.scale != first.scale:
            raise ValueError("视频轨道时基不一致")
        # 独立编码的片段可能有不同的 SPS/PPS 或 AV1 配置。每块引用自己的 stsd 条目。
        if v.entry not in entries:
            entries.append(v.entry)
        local, keys = v.presentation(span)
        offsets.extend(local)
        sync.extend(len(sizes) + k for k in keys)
        video_chunks.extend(v.chunks(total / v.scale, entries.index(v.entry) + 1))
        deltas.extend(v.delta)
        sizes.extend(v.sizes)
        total += span

    audio = Track(audio_path)
    audio_chunks = list(audio.chunks(0.0, 1))
    chunks = sorted(video_chunks + audio_chunks, key=lambda c: c.at)
    movie_scale = audio.scale
    movie_duration = (total * movie_scale + first.scale // 2) // first.scale
    # B 帧允许显示顺序早于解码序号。统一抬高 ctts，用整条轨道的一项 edit 还原零点。
    bias = max(0, -min(offsets))
    video_tables = {
        b"stsd": bytes(4) + struct.pack(">I", len(entries)) + b"".join(entries),
        b"stts": table(timing_runs(deltas), ">II"),
        b"stsz": bytes(8) + struct.pack(">I", len(sizes)) + packed(sizes),
    }
    if bias or any(offsets):
        video_tables[b"ctts"] = table(timing_runs(o + bias for o in offsets), ">II")
    if len(sync) != len(sizes):
        video_tables[b"stss"] = bytes(4) + struct.pack(">I", len(sync)) + packed(sync)
    edit = box(b"elst", b"\x01\0\0\0" + struct.pack(">IQqI", 1, movie_duration, bias, 65536))
    tkhd = track_header(first.trak, movie_duration, rotation)
    mvhd = time_header(child(first.moov, b"mvhd"), b"mvhd", movie_duration, movie_scale)
    mvhd = mvhd[:-4] + struct.pack(">I", 3)

    def movie() -> bytes:
        vt = b"".join(
            box(k, v) for k, v in {**video_tables, **chunk_tables(video_chunks)}.items() if v is not None
        )
        vm = replace(
            first.mdia,
            {
                b"minf": replace(first.minf, {b"stbl": vt}),
                b"mdhd": time_header(child(first.mdia, b"mdhd"), b"mdhd", total, first.scale),
            },
        )
        video_track = replace(first.trak, {b"tkhd": tkhd, b"edts": edit, b"mdia": vm})
        at = replace(audio.stbl, chunk_tables(audio_chunks))
        am = replace(audio.mdia, {b"minf": replace(audio.minf, {b"stbl": at})})
        audio_track = replace(
            audio.trak,
            {b"mdia": am, b"tkhd": time_header(child(audio.trak, b"tkhd"), b"tkhd", movie_duration, 2)},
        )
        return box(b"moov", box(b"mvhd", mvhd) + box(b"trak", video_track) + box(b"trak", audio_track))

    ftyp = box(b"ftyp", first.ftyp)
    # co64 的条目是定长的，先计算 moov 长度，即可在写媒体数据之前填好所有文件偏移。
    start = len(ftyp) + len(movie()) + 16
    pos = start
    for c in chunks:
        c.offset = pos
        pos += c.size
    with output.open("xb") as dst:
        dst.write(ftyp + movie() + struct.pack(">I4sQ", 1, b"mdat", pos - start + 16))
        copy_chunks(dst, audio_path, chunks)


def copy_chunks(dst: BinaryIO, audio_path: Path, chunks: list[Chunk]) -> None:
    with audio_path.open("rb") as sound, ExitStack() as pictures:
        picture = None
        for chunk in chunks:
            if chunk.path == audio_path:
                source = sound
            else:
                if picture is None or picture.name != str(chunk.path):
                    pictures.close()
                    picture = pictures.enter_context(chunk.path.open("rb"))
                source = picture
            source.seek(chunk.source)
            data = source.read(chunk.size)
            if len(data) != chunk.size:
                raise ValueError(f"{chunk.path.name} 的压缩样本不完整")
            dst.write(data)
