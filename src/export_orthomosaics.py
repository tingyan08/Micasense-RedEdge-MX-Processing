import os
import sys
import glob
import time
import shutil
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Code/ root

import Metashape
from metashape import metashape_pipeline, get_capture_gdf

TARGET_CRS = "EPSG:32616"
DATA_ROOT = "Data"

JOBS = [
    ("PPAC-B3",             "2021", "073021_PPAC-B3"),
    ("PPAC-B3",             "2022", "072922_PPAC-B3"),
    ("PPAC-B3",             "2023", "071023_PPAC-B3"),
    ("PPAC-B3",             "2024", "072324_PPAC-B3"),
    ("RiceFarm-SouthPivot", "2021", "080421_RiceFarm-SouthPivot"),
    ("SWPAC",               "2021", "080221_SWPAC"),
    ("Rice-NorthPivot",     "2022", "071222_Rice-NorthPivot"),
    ("Rice-SouthPivot",     "2023", "071423_Rice-SouthPivot"),
    ("SEPAC-D3",            "2022", "070722_SEPAC-D3"),
    ("SEPAC-D3",            "2023", "080423_SEPAC-D3"),
]


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def ortho_chunk(doc):
    return next((c for c in doc.chunks
                 if c.label == "full_field" and c.orthomosaic is not None), None)


def export_ortho(chunk, out_tif):
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    proj = Metashape.OrthoProjection()
    proj.crs = Metashape.CoordinateSystem(TARGET_CRS)
    chunk.exportRaster(path=out_tif, source_data=Metashape.OrthomosaicData,
                       image_format=Metashape.ImageFormatTIFF, save_alpha=True,
                       projection=proj)
    return sum(os.path.getsize(f) for f in glob.glob(out_tif.replace(".tif", "*"))
               if os.path.isfile(f))


def process(field, year, exp):
    root = os.path.join(DATA_ROOT, field, year, exp)
    wf = os.path.join(root, "Metashape", "whole_field")
    out_tif = os.path.join(wf, "Orthomosaics", f"{exp}_orthomosaic.tif")
    psx = os.path.join(wf, f"{exp}_whole_field.psx")

    # 1) If a completed whole_field project already exists, just export its orthomosaic.
    if os.path.exists(psx):
        for lk in glob.glob(os.path.join(wf, "**", "lock"), recursive=True):
            try:
                os.remove(lk)
            except OSError:
                pass
        doc = Metashape.Document()
        doc.open(psx, read_only=False)
        ch = ortho_chunk(doc)
        if ch is not None:
            size = export_ortho(ch, out_tif)
            print(f"[OK] {exp}: exported existing ortho -> {out_tif} ({human(size)})", flush=True)
            return
        del doc  # project exists but has no orthomosaic: rebuild below

    # 2) Otherwise build in a disposable project, keep only the exported orthomosaic.
    build = os.path.join(root, "Metashape", "_build")
    if os.path.isdir(build):
        shutil.rmtree(build, ignore_errors=True)
    os.makedirs(build, exist_ok=True)

    capture_gdf = get_capture_gdf(root).to_crs(TARGET_CRS)
    doc = Metashape.Document()
    doc.save(os.path.join(build, f"{exp}.psx"))
    images = [os.path.join(root, "Images", f"{i}_{j}.tif")
              for i in capture_gdf["image_name"].tolist() for j in range(1, 6)]
    panels = glob.glob(os.path.join(root, "Panel", "*.tif"))

    t0 = time.time()
    metashape_pipeline(build, doc, images, panels, TARGET_CRS,
                       chunk_label="full_field", export=False)
    dt = time.time() - t0

    ch = doc.chunks[-1]
    size = export_ortho(ch, out_tif)
    del doc, ch
    shutil.rmtree(build, ignore_errors=True)  # discard heavy project, keep the ortho GeoTIFF
    print(f"[OK] {exp}: built {dt/60:.1f} min -> {out_tif} ({human(size)})", flush=True)


def main():
    for field, year, exp in JOBS:
        print(f"===== {field} {year} {exp} =====", flush=True)
        try:
            process(field, year, exp)
        except Exception:
            traceback.print_exc()
            print(f"[FAIL] {exp}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
