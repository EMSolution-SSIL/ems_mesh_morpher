from __future__ import annotations

from pathlib import Path

import numpy as np


def _line_ending(lines: list[str]) -> str:
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
    return "\n"


def _read_int_tokens(lines: list[str], start: int, count: int) -> tuple[int, list[str]]:
    consumed: list[str] = []
    found = 0
    index = start
    while found < count:
        line = lines[index]
        consumed.append(line)
        found += len(line.split())
        index += 1
    if found != count:
        raise ValueError("unexpected Gmsh node tag layout")
    return index, consumed


def _skip_coordinate_lines(lines: list[str], start: int, count: int, entries_per_node: int) -> int:
    found = 0
    index = start
    while found < count:
        found += len(lines[index].split()) // entries_per_node
        index += 1
    if found != count:
        raise ValueError("unexpected Gmsh node coordinate layout")
    return index


def read_gmsh4_node_coordinates(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "$Nodes")
    except StopIteration as exc:
        raise ValueError("Gmsh file does not contain a $Nodes section") from exc

    header = lines[start + 1].split()
    if len(header) != 4:
        raise ValueError("unsupported Gmsh $Nodes header")
    block_count = int(header[0])
    node_count = int(header[1])

    points = np.zeros((node_count, 3), dtype=float)
    point_index = 0
    index = start + 2
    for _ in range(block_count):
        block_header = lines[index].split()
        index += 1
        if len(block_header) != 4:
            raise ValueError("unsupported Gmsh node block header")
        parametric = int(block_header[2])
        block_node_count = int(block_header[3])
        entries_per_node = 3 + parametric

        index, _ = _read_int_tokens(lines, index, block_node_count)
        values: list[float] = []
        while len(values) < block_node_count * entries_per_node:
            values.extend(float(v) for v in lines[index].split())
            index += 1
        if len(values) != block_node_count * entries_per_node:
            raise ValueError("unexpected Gmsh node coordinate values")

        coords = np.asarray(values, dtype=float).reshape(block_node_count, entries_per_node)[:, :3]
        points[point_index : point_index + block_node_count] = coords
        point_index += block_node_count

    if lines[index].strip() != "$EndNodes":
        raise ValueError("Gmsh $Nodes section did not end where expected")
    return points


def write_gmsh4_with_updated_nodes(
    template_path: str | Path,
    output_path: str | Path,
    points: np.ndarray,
) -> None:
    lines = Path(template_path).read_text(encoding="utf-8").splitlines(keepends=True)
    ending = _line_ending(lines)

    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "$Nodes")
    except StopIteration as exc:
        raise ValueError("Gmsh template does not contain a $Nodes section") from exc

    header = lines[start + 1].split()
    if len(header) != 4:
        raise ValueError("unsupported Gmsh $Nodes header")
    block_count = int(header[0])
    node_count = int(header[1])

    if int(points.shape[0]) != node_count:
        raise ValueError(f"point count mismatch: template has {node_count}, mesh has {points.shape[0]}")

    out_lines = lines[:start]
    out_lines.append("$Nodes" + ending)
    out_lines.append(lines[start + 1])

    point_index = 0
    index = start + 2
    for _ in range(block_count):
        block_header_line = lines[index]
        block_header = block_header_line.split()
        index += 1
        if len(block_header) != 4:
            raise ValueError("unsupported Gmsh node block header")
        parametric = int(block_header[2])
        block_node_count = int(block_header[3])
        if parametric != 0:
            raise ValueError("parametric Gmsh nodes are not supported by the coordinate replacement writer")

        out_lines.append(block_header_line)
        index, tag_lines = _read_int_tokens(lines, index, block_node_count)
        out_lines.extend(tag_lines)
        index = _skip_coordinate_lines(lines, index, block_node_count, entries_per_node=3)

        for row in points[point_index : point_index + block_node_count]:
            out_lines.append(f"{row[0]:.17g} {row[1]:.17g} {row[2]:.17g}{ending}")
        point_index += block_node_count

    if lines[index].strip() != "$EndNodes":
        raise ValueError("Gmsh $Nodes section did not end where expected")
    out_lines.append(lines[index])
    out_lines.extend(lines[index + 1 :])

    Path(output_path).write_text("".join(out_lines), encoding="utf-8", newline="")
