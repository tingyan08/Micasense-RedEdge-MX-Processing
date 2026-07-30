import os
import glob
import shutil

import micasense.imageset as imageset

SESSION_GAP = 120.0    # seconds; splits pre-flight/battery-leg sessions apart
BOUNDARY_K = 10        # panels sit within the first/last K captures of a raw session
MIN_FLIGHT_SESSION = 10  # keep+merge flight sessions this large; drop smaller takeoff/stray sessions

YEAR_SRC = {
    "2022": "/media/tingyan/Data/data_fusion/2022/Corn_US_IN_PPAC_B3/Digital_Imagery/UAV",
    "2023": "/media/tingyan/Data/data_fusion/2023/Corn_US_IN_PPAC_B3/Digital_Imagery/UAV",
}
DEST_ROOT = "/media/tingyan/Research/USDA-NIFA/Code/Data/PPAC-B3"


def dest_date_name(year: str, raw: str) -> str:
    if year == "2022":  # YYMMDD
        yy, mm, dd = raw[0:2], raw[2:4], raw[4:6]
    else:               # 2023 YYYYMMDD
        yy, mm, dd = raw[2:4], raw[4:6], raw[6:8]
    return f"{mm}{dd}{yy}_PPAC-B3"


def raw_sessions(caps: list, gap: float) -> list:
    sessions = []
    current = [caps[0]]
    for prev, cur in zip(caps, caps[1:]):
        if (cur.utc_time() - prev.utc_time()).total_seconds() > gap:
            sessions.append(current)
            current = [cur]
        else:
            current.append(cur)
    sessions.append(current)
    return sessions


def detect_panel_flags(session: list, k: int) -> list:
    n = len(session)
    check = set(range(min(k, n))) | set(range(max(0, n - k), n))
    flags = [False] * n
    for i in sorted(check):
        cap = session[i]
        try:
            flags[i] = cap.panels_in_all_expected_images()
        except Exception:
            flags[i] = False
        cap.clear_image_data()
    return flags


def copy_capture(cap, dest_dir: str, new_idx: int) -> None:
    for img in cap.images:
        band = os.path.basename(img.path).rsplit("_", 1)[-1]  # e.g. "1.tif"
        dst = os.path.join(dest_dir, f"IMG_{new_idx:04d}_{band}")
        shutil.copy(img.path, dst)


def process_date(year: str, raw: str) -> None:
    date_dir = os.path.join(YEAR_SRC[year], raw)
    imgset = imageset.ImageSet.from_directory(date_dir)
    caps = sorted(imgset.captures, key=lambda c: c.utc_time())

    sessions = raw_sessions(caps, SESSION_GAP)

    panels = []
    session_flights = []
    for sess in sessions:
        flags = detect_panel_flags(sess, BOUNDARY_K)
        flights = [c for c, f in zip(sess, flags) if not f]
        panels += [c for c, f in zip(sess, flags) if f]
        session_flights.append(flights)

    kept = [f for f in session_flights if len(f) >= MIN_FLIGHT_SESSION]
    if not kept:
        kept = [max(session_flights, key=len)]
    main_flight = sorted(
        [c for f in kept for c in f], key=lambda c: c.utc_time()
    )

    dest = os.path.join(DEST_ROOT, year, dest_date_name(year, raw))
    images_dir = os.path.join(dest, "Images")
    panel_dir = os.path.join(dest, "Panel")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(panel_dir, exist_ok=True)

    for i, cap in enumerate(main_flight):
        copy_capture(cap, images_dir, i)
    for i, cap in enumerate(sorted(panels, key=lambda c: c.utc_time())):
        copy_capture(cap, panel_dir, i)

    print(
        f"{year}/{raw} -> {dest_date_name(year, raw)}: "
        f"{len(caps)} captures, {len(panels)} panel, {len(main_flight)} flight "
        f"(dropped {len(caps) - len(panels) - len(main_flight)})",
        flush=True,
    )


def main() -> None:
    for year, src in YEAR_SRC.items():
        for date_dir in sorted(glob.glob(os.path.join(src, "*"))):
            if not os.path.isdir(date_dir):
                continue
            process_date(year, os.path.basename(date_dir))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
