import os
import re
import glob
import fnmatch
import shutil
import subprocess

import exiftool
import micasense.image as image
import micasense.capture as capture

DATA_FUSION = "/media/tingyan/Data/data_fusion"
DEST_BASE = "/media/tingyan/Research/USDA-NIFA/Code/Data"
SUMMARY_PATH = "/media/tingyan/Research/USDA-NIFA/Code/DATASET_SUMMARY.md"

EXPECTED_BANDS = 5   # MicaSense RedEdge-M
BOUNDARY_K = 25      # panels sit within the first/last K captures (acquisition order)


def short_name(field_dir: str) -> str:
    return os.path.basename(field_dir).replace("Corn_US_IN_", "").replace("_", "-")


def find_uav(field_dir: str) -> str | None:
    out = subprocess.run(
        ["find", field_dir, "-maxdepth", "2", "-type", "d", "-iname", "UAV"],
        capture_output=True, text=True,
    ).stdout.split("\n")
    dirs = sorted(d for d in out if d.strip())
    return dirs[0] if dirs else None


def folder_date(date_dir: str, year: str) -> str | None:
    """MMDDYY parsed from the date folder name (robust to bad/epoch image clocks).

    Handles YYYYMMDD (2021/2023), YYMMDD (e.g. PPAC/Rice 2022) and MMDDYY (e.g. SEPAC 2022),
    disambiguating the 6-digit forms using the known collection year.
    """
    digits = "".join(c for c in os.path.basename(date_dir) if c.isdigit())
    yy = year[2:]
    if len(digits) == 8:                       # YYYYMMDD
        return digits[4:6] + digits[6:8] + digits[2:4]
    if len(digits) == 6:
        if digits[:2] == yy:                   # YYMMDD
            return digits[2:4] + digits[4:6] + digits[:2]
        if digits[4:6] == yy:                  # MMDDYY
            return digits
    return None


def load_captures(date_dir: str, expected_bands: int = EXPECTED_BANDS) -> list:
    """Recursively load captures, robust to corrupt files and duplicate "Combined" folders.

    - Skips zero-byte/unreadable images (which otherwise abort ImageSet.from_directory).
    - Groups by capture_id (falls back to dir+IMG-prefix when capture_id is missing) and keeps
      one image per band, preferring the original *SET copy over a "Combined" duplicate. This
      de-duplicates datasets that ship a "Combined" folder mirroring the SET folders.
    - Drops incomplete captures (fewer than expected_bands).
    """
    matches = []
    for root, _, files in os.walk(date_dir):
        for fn in fnmatch.filter(files, "*.tif"):
            p = os.path.join(root, fn)
            if os.path.getsize(p) > 0:
                matches.append(p)

    images = []
    with exiftool.ExifToolHelper() as exift:
        for p in matches:
            try:
                images.append(image.Image(p, exiftool_obj=exift))
            except Exception:
                continue

    index = {}
    for img in images:
        cid = img.capture_id or (os.path.dirname(img.path),
                                 os.path.basename(img.path).rsplit("_", 1)[0])
        band = os.path.basename(img.path).rsplit("_", 1)[-1]  # e.g. "1.tif"
        bands = index.setdefault(cid, {})
        if band not in bands or ("Combined" in bands[band].path and "Combined" not in img.path):
            bands[band] = img

    caps = []
    for bands in index.values():
        if len(bands) != expected_bands:
            continue
        try:
            caps.append(capture.Capture(list(bands.values())))
        except Exception:
            continue
    return caps


def acq_key(cap):
    """Acquisition order = (SET directory, IMG index) from an original (non-Combined) image."""
    p = min((im.path for im in cap.images if "Combined" not in im.path),
            default=cap.images[0].path)
    m = re.search(r"IMG_(\d+)", os.path.basename(p))
    return (os.path.dirname(p), int(m.group(1)) if m else 0)


def copy_capture(cap, dest_dir: str, new_idx: int) -> None:
    for img in cap.images:
        band = os.path.basename(img.path).rsplit("_", 1)[-1]  # e.g. "1.tif"
        shutil.copy(img.path, os.path.join(dest_dir, f"IMG_{new_idx:04d}_{band}"))


def process_date(date_dir: str, year: str, short: str) -> None:
    mmddyy = folder_date(date_dir, year)
    exp = f"{mmddyy}_{short}" if mmddyy else None
    if exp is not None:
        images_dir = os.path.join(DEST_BASE, short, year, exp, "Images")
        if glob.glob(os.path.join(images_dir, "*.tif")):
            print(f"[DONE] {short}/{year}/{exp}: already populated, skipping", flush=True)
            return

    caps = load_captures(date_dir)
    if not caps:
        print(f"[SKIP] {year}/{short}/{os.path.basename(date_dir)}: no captures", flush=True)
        return
    caps.sort(key=acq_key)

    if mmddyy is None:  # folder name unparseable: fall back to latest (non-epoch) image time
        mmddyy = max(c.utc_time() for c in caps).strftime("%m%d%y")
        exp = f"{mmddyy}_{short}"

    # Panels are calibration captures at the very start or end of acquisition.
    n = len(caps)
    boundary = sorted(set(range(min(BOUNDARY_K, n))) | set(range(max(0, n - BOUNDARY_K), n)))
    panel_idx = set()
    for i in boundary:
        try:
            if caps[i].panels_in_all_expected_images():
                panel_idx.add(i)
        except Exception:
            pass
        caps[i].clear_image_data()

    panels = [caps[i] for i in range(n) if i in panel_idx]
    flight = [caps[i] for i in range(n) if i not in panel_idx]

    dest = os.path.join(DEST_BASE, short, year, exp)
    images_dir = os.path.join(dest, "Images")
    panel_dir = os.path.join(dest, "Panel")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(panel_dir, exist_ok=True)
    for i, cap in enumerate(flight):
        copy_capture(cap, images_dir, i)
    for i, cap in enumerate(panels):
        copy_capture(cap, panel_dir, i)

    print(f"{short}/{year}/{exp}: {n} captures, {len(panels)} panel, {len(flight)} flight",
          flush=True)


def reorganize() -> None:
    for year in sorted(os.listdir(DATA_FUSION)):
        year_dir = os.path.join(DATA_FUSION, year)
        if not (year.isdigit() and os.path.isdir(year_dir)):
            continue
        for field in sorted(os.listdir(year_dir)):
            field_dir = os.path.join(year_dir, field)
            if not os.path.isdir(field_dir):
                continue
            uav = find_uav(field_dir)
            if uav is None:
                print(f"[SKIP] {year}/{field}: no UAV folder", flush=True)
                continue
            short = short_name(field_dir)
            for date_dir in sorted(glob.glob(os.path.join(uav, "*"))):
                if not os.path.isdir(date_dir):
                    continue
                try:
                    process_date(date_dir, year, short)
                except Exception as e:
                    print(f"[ERROR] {year}/{short}/{os.path.basename(date_dir)}: {e!r}", flush=True)


def count_captures(folder: str) -> int:
    return len(glob.glob(os.path.join(folder, "*.tif"))) // EXPECTED_BANDS


def write_summary() -> None:
    rows = []  # (field, year, iso_date, captures, panels)
    for short in sorted(os.listdir(DEST_BASE)):
        field_dir = os.path.join(DEST_BASE, short)
        if not os.path.isdir(field_dir):
            continue
        for year in sorted(os.listdir(field_dir)):
            year_dir = os.path.join(field_dir, year)
            if not (year.isdigit() and os.path.isdir(year_dir)):
                continue
            for exp in sorted(os.listdir(year_dir)):
                images = os.path.join(year_dir, exp, "Images")
                if not os.path.isdir(images):
                    continue
                mmddyy = exp.split("_")[0]
                iso = f"20{mmddyy[4:6]}-{mmddyy[0:2]}-{mmddyy[2:4]}"
                rows.append((short, year, iso,
                             count_captures(images),
                             count_captures(os.path.join(year_dir, exp, "Panel"))))

    fields = sorted(set(r[0] for r in rows))
    lines = [
        "# MicaSense Dataset Summary", "",
        f"Reorganized from `{DATA_FUSION}` into "
        "`Data/<field>/<year>/<MMDDYY_field>/{Images,Panel}`.",
        "Captures are 5-band RedEdge-M sets; panels are the calibration-panel captures "
        "detected by micasense.", "",
        f"**Totals:** {len(fields)} fields, {len(rows)} flight dates, "
        f"{sum(r[3] for r in rows)} flight captures, {sum(r[4] for r in rows)} panel captures.",
        "", "## Per-field / per-date", "",
    ]
    for field in fields:
        frows = sorted((r for r in rows if r[0] == field), key=lambda r: r[2])
        years = sorted(set(r[1] for r in frows))
        lines += [
            f"### {field}",
            f"{len(frows)} dates across {', '.join(years)} — "
            f"{sum(r[3] for r in frows)} captures, {sum(r[4] for r in frows)} panels.", "",
            "| Date | Year | Captures | Panels |",
            "|------|------|----------|--------|",
        ]
        lines += [f"| {iso} | {year} | {caps} | {pan} |" for _, year, iso, caps, pan in frows]
        lines.append("")

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote summary -> {SUMMARY_PATH} ({len(rows)} dates)", flush=True)


if __name__ == "__main__":
    reorganize()
    write_summary()
    print("DONE", flush=True)
