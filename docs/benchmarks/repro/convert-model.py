"""Convert IfcOpenShell geometry or OCCT glTF exports into an iletki benchmark fixture."""
import argparse
import json
from math import isfinite
from pathlib import Path
import struct
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np


VERTEX = np.dtype([("position", "<f4", 3), ("normal", "<f4", 3), ("rgba", "u1", 4)])
BASE_RGBA = np.array([0.45, 0.60, 0.70, 1.0])
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
HEADER = struct.Struct("<8sI")
PART = struct.Struct("<I3dI")


def coarse_vertices(data, cell_size):
    triangles = data.reshape(-1, 3)
    if len(triangles) < 64:
        return data, 0.0
    original = triangles["position"].astype(np.float64)
    flat = original.reshape(-1, 3)
    with np.errstate(over="ignore"):
        snapped = np.clip(np.rint(original / cell_size) * cell_size, flat.min(0), flat.max(0))
    snapped = snapped.astype(np.float32).astype(np.float64)
    area = np.cross(snapped[:, 1] - snapped[:, 0], snapped[:, 2] - snapped[:, 0])
    retained = np.any(area != 0, axis=1)
    if not retained.any() or retained.all():
        return data, 0.0

    # Remove a collapsed face only when its point or line is also on a retained face.
    # Keep unsupported collapsed faces at their original positions.
    points, indices = np.unique(snapped.reshape(-1, 3), axis=0, return_inverse=True)
    indices = np.sort(indices.reshape(-1, 3).astype(np.uint64), axis=1)
    faces = indices[retained]
    edges = np.concatenate([
        faces[:, 0] * len(points) + faces[:, 1],
        faces[:, 0] * len(points) + faces[:, 2],
        faces[:, 1] * len(points) + faces[:, 2],
    ])
    start, end = indices[:, 0], indices[:, 2]
    supported = np.where(
        start == end,
        np.isin(start, faces.ravel()),
        np.isin(start * len(points) + end, edges),
    )
    keep = retained | ~supported
    if keep.all():
        return data, 0.0
    output = triangles[keep].copy()
    output["position"][retained[keep]] = snapped[retained]
    displacement = float(np.linalg.norm(snapped - original, axis=2).max())
    return output.reshape(-1), displacement


def write_lod_fixture(source, destination, cell_size):
    if source.resolve() == destination.resolve():
        raise ValueError("source and LOD output must be different files")
    with source.open("rb") as original, NamedTemporaryFile(
        mode="wb", dir=destination.parent, delete=False
    ) as coarse:
        temporary = Path(coarse.name)
        try:
            header = original.read(HEADER.size)
            magic, count = HEADER.unpack(header)
            if magic != b"ILTKM01\0":
                raise ValueError("invalid fixture header")
            coarse.write(header)
            triangles = 0
            max_error = 0.0
            for _ in range(count):
                ident, x, y, z, size = PART.unpack(original.read(PART.size))
                if size == 0 or size % 3 or size > (512 << 20) // VERTEX.itemsize:
                    raise ValueError("invalid fixture mesh")
                data = np.fromfile(original, dtype=VERTEX, count=size)
                if len(data) != size or not np.isfinite(data["position"]).all():
                    raise ValueError("invalid fixture positions")
                result, error = coarse_vertices(data, cell_size)
                coarse.write(PART.pack(ident, x, y, z, len(result)))
                result.tofile(coarse)
                triangles += len(result) // 3
                max_error = max(max_error, error)
            if original.read(1):
                raise ValueError("unexpected trailing fixture data")
            coarse.close()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"{destination.name} {count} parts {triangles} triangles, max_world_error={max_error:.17g}")


def to_srgb(rgba):
    values = np.clip(np.nan_to_num(rgba, nan=1.0), 0, 1)
    rgb = values[..., :3]
    values[..., :3] = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.round(values * 255).astype(np.uint8)


def vertices(triangles, rgba, normals=None):
    flat = triangles.reshape(-1, 3)
    if normals is None:
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals = np.repeat(face_normals, 3, axis=0)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    unit = np.divide(normals, length, out=np.zeros_like(normals), where=length > 1e-12)
    result = np.empty(len(flat), dtype=VERTEX)
    result["position"] = flat
    result["normal"] = unit
    result["rgba"] = np.repeat(to_srgb(np.broadcast_to(rgba, (len(triangles), 4))), 3, axis=0)
    return result


def ifc_parts(path):
    import ifcopenshell
    from ifcopenshell import geom
    from ifcopenshell.util import shape

    model = ifcopenshell.open(path)
    settings = geom.settings()
    settings.set("use-world-coords", False)
    for element in geom.iterate(settings, model, 4, exclude=["IfcSpace", "IfcOpeningElement"]):
        mesh = element.geometry
        faces = shape.get_faces(mesh)
        if len(faces) == 0:
            continue
        matrix = shape.get_shape_matrix(element)
        local = shape.get_vertices(mesh).reshape(-1, 3) @ matrix[:3, :3].T
        palette = shape.get_material_colors(mesh)
        material = shape.get_faces_material_style_ids(mesh)
        rgba = np.tile(BASE_RGBA, (len(faces), 1))
        if len(material) == len(faces) and len(palette):
            valid = (material >= 0) & (material < len(palette))
            rgba[valid] = palette[material[valid]]
        yield element.id, matrix[:3, 3].copy(), vertices(local[faces], rgba)


def step_parts(path, first_id):
    gltf = json.loads(path.read_text())
    buffers = [(path.parent / buf["uri"]).read_bytes() for buf in gltf["buffers"]]
    accessors = gltf["accessors"]
    widths = {"SCALAR": 1, "VEC3": 3}
    types = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}

    def values(i):
        spec = accessors[i]
        view = gltf["bufferViews"][spec["bufferView"]]
        t = np.dtype(types[spec["componentType"]])
        width = widths[spec["type"]]
        return np.ndarray(
            shape=(spec["count"], width), dtype=t, buffer=buffers[view["buffer"]],
            offset=view.get("byteOffset", 0) + spec.get("byteOffset", 0),
            strides=(view.get("byteStride", width * t.itemsize), t.itemsize),
        )

    def visit(i, parent):
        node = gltf["nodes"][i]
        transform = np.array(node.get("matrix", np.eye(4).flatten()), dtype=float).reshape(4, 4).T
        if any(key in node for key in ("rotation", "translation", "scale")):
            raise ValueError("export glTF with OCCT -trsfFormat mat4")
        matrix = parent @ transform
        if "mesh" in node:
            offset = Y_UP_TO_Z_UP @ matrix[:3, 3]
            basis = Y_UP_TO_Z_UP @ matrix[:3, :3]
            records = []
            for prim in gltf["meshes"][node["mesh"]]["primitives"]:
                attrs = prim["attributes"]
                if prim.get("mode", 4) != 4 or attrs.get("POSITION", -1) < 0:
                    continue
                points = values(attrs["POSITION"]) @ basis.T
                ids = (values(prim["indices"]).reshape(-1) if "indices" in prim
                       else np.arange(len(points))).reshape(-1, 3)
                normals = None
                if "NORMAL" in attrs:
                    unit = values(attrs["NORMAL"]) @ np.linalg.inv(basis)
                    normals = unit[ids].reshape(-1, 3)
                mats = gltf.get("materials", [])
                rgba = BASE_RGBA
                if "material" in prim:
                    rgba = mats[prim["material"]].get("pbrMetallicRoughness", {}).get("baseColorFactor", rgba)
                records.append(vertices(points[ids], np.asarray(rgba), normals))
            if records:
                yield offset, np.concatenate(records)
        for child in node.get("children", []):
            yield from visit(child, matrix)

    roots = gltf["scenes"][gltf.get("scene", 0)]["nodes"]
    entries = (part for root in roots for part in visit(root, np.eye(4)))
    for obj, (origin, data) in enumerate(entries, first_id):
        yield obj, origin, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("format", choices=["ifc", "step-gltf", "lod"])
    parser.add_argument("output", type=Path)
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--cell-size", type=float)
    args = parser.parse_args()
    if args.format == "lod":
        if len(args.input) != 1 or args.cell_size is None or not isfinite(args.cell_size) or args.cell_size <= 0:
            parser.error("LOD requires one fixture and a finite positive --cell-size")
        write_lod_fixture(args.input[0], args.output, args.cell_size)
        return
    started = perf_counter()
    if args.format == "ifc":
        parts = []
        for path in args.input:
            parts.extend(sorted(ifc_parts(path), key=lambda part: part[0]))
        if len(args.input) > 1:
            # STEP entity IDs are local to their IFC file. Use unique pick IDs in a federation.
            parts = [(i, origin, data) for i, (_, origin, data) in enumerate(parts, 1)]
    else:
        parts = [part for i, path in enumerate(args.input) for part in step_parts(path, i * 100_000 + 1)]
    triangles = sum(len(data) // 3 for _, _, data in parts)
    if not parts or not triangles:
        raise SystemExit("model has no triangle geometry")
    with args.output.open("wb") as output:
        output.write(struct.pack("<8sI", b"ILTKM01\0", len(parts)))
        for object_id, origin, data in parts:
            output.write(struct.pack("<I3dI", object_id, *origin, len(data)))
            data.tofile(output)
    print(f"{args.output.name} {len(parts)} parts {triangles} triangles, {(perf_counter() - started) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
