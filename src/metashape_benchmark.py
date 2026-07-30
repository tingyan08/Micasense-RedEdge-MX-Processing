import os
import sys
import glob
import time
import shutil
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Code/ root

import Metashape
from metashape import metashape_pipeline, get_capture_gdf

TARGET_CRS = "EPSG:32616"  # WGS84 / UTM 16N (Indiana)
DATA_ROOT = "Data"
OUT_MD = "METASHAPE_BENCHMARK.md"

# One representative (mid-season, panel-bearing) date per field-year.
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


def run_job(field, year, exp):
    root = os.path.join(DATA_ROOT, field, year, exp)
    # Isolated, disposable benchmark project — never touches the real whole_field projects.
    bench = os.path.join(root, "Metashape", "benchmark")
    if os.path.isdir(bench):
        shutil.rmtree(bench, ignore_errors=True)
    os.makedirs(bench, exist_ok=True)

    capture_gdf = get_capture_gdf(root).to_crs(TARGET_CRS)
    ncap = len(capture_gdf)

    psx = os.path.join(bench, f"{exp}_bench.psx")
    doc = Metashape.Document()
    doc.save(psx)  # fresh, writable, single-chunk project

    images = [os.path.join(root, "Images", f"{i}_{j}.tif")
              for i in capture_gdf["image_name"].tolist() for j in range(1, 6)]
    panels = glob.glob(os.path.join(root, "Panel", "*.tif"))

    t0 = time.time()
    metashape_pipeline(bench, doc, images, panels, TARGET_CRS,
                       chunk_label="full_field", export=False)
    proc_time = time.time() - t0

    chunk = doc.chunks[-1]  # the chunk this pipeline just built
    ortho = chunk.orthomosaic
    w, h, res = ortho.width, ortho.height, ortho.resolution  # px, px, m/px

    # Export a single-file GeoTIFF to measure on-disk storage size.
    proj = Metashape.OrthoProjection()
    proj.crs = Metashape.CoordinateSystem(TARGET_CRS)
    ortho_tif = os.path.join(bench, f"{exp}_orthomosaic.tif")
    chunk.exportRaster(path=ortho_tif, source_data=Metashape.OrthomosaicData,
                       image_format=Metashape.ImageFormatTIFF, save_alpha=True,
                       projection=proj)
    size = sum(os.path.getsize(f) for f in glob.glob(ortho_tif.replace(".tif", "*"))
               if os.path.isfile(f))

    # Discard the disposable benchmark project entirely (bounds disk use).
    del doc, chunk, ortho
    shutil.rmtree(bench, ignore_errors=True)

    return dict(field=field, year=year, exp=exp, ncap=ncap, time_s=proc_time,
                w=w, h=h, res=res, size=size)


def write_md(rows):
    lines = [
        "# Metashape whole-field benchmark", "",
        "One representative date per field-year, full pipeline "
        "(align → depth maps → point cloud → DEM → orthomosaic) on GPU. "
        "Projection size = orthomosaic raster dimensions and ground coverage in "
        f"UTM 16N; storage size = exported single GeoTIFF ({TARGET_CRS}).", "",
        "| Field | Year | Date | Captures | Proc. time | Ortho pixels (W×H) | Resolution | Ground coverage | Storage size |",
        "|-------|------|------|---------:|-----------:|--------------------|-----------:|-----------------|-------------:|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['field']} | {r['year']} | {r['exp'].split('_')[0]} | "
                         f"{r.get('ncap','?')} | FAILED | — | — | — | — |")
            continue
        mins = r["time_s"] / 60
        gw, gh = r["w"] * r["res"], r["h"] * r["res"]
        lines.append(
            f"| {r['field']} | {r['year']} | {r['exp'].split('_')[0]} | {r['ncap']} | "
            f"{mins:.1f} min | {r['w']:,} × {r['h']:,} | {r['res']*100:.2f} cm/px | "
            f"{gw:.0f} × {gh:.0f} m | {human(r['size'])} |"
        )
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}", flush=True)


def main():
    rows = []
    for field, year, exp in JOBS:
        print(f"===== {field} {year} {exp} =====", flush=True)
        try:
            r = run_job(field, year, exp)
            rows.append(r)
            print(f"[OK] {exp}: {r['time_s']/60:.1f} min | {r['w']}x{r['h']} px @ "
                  f"{r['res']*100:.2f} cm | {human(r['size'])}", flush=True)
        except Exception:
            traceback.print_exc()
            rows.append(dict(field=field, year=year, exp=exp, error=True))
            print(f"[FAIL] {exp}", flush=True)
        write_md(rows)  # rewrite after each job so partial results survive
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
