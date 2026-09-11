from pathlib import Path

import meshio

from ems_file_format_converter import read_mesh, write_mesh
from .gmsh4 import write_gmsh4_with_updated_nodes


def read_mesh_auto(path: str | Path, file_format: str | None = None) -> meshio.Mesh:
    return read_mesh(path, file_format=file_format)


def write_mesh_auto(
    path: str | Path,
    mesh: meshio.Mesh,
    file_format: str | None = None,
    template_path: str | Path | None = None,
) -> None:
    out_path = Path(path)
    resolved_format = file_format
    if out_path.suffix.lower() == ".msh" and resolved_format in (None, "gmsh", "gmsh4", "gmsh41"):
        if template_path is not None:
            write_gmsh4_with_updated_nodes(template_path, out_path, mesh.points)
        else:
            write_mesh(out_path, mesh, file_format="gmsh4")
        return
    write_mesh(out_path, mesh, file_format=resolved_format)
