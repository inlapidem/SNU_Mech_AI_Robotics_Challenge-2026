"""Generate print-ready ArUco marker sheets for the Set 2 real-capture pipeline.

The real cubes (datasets/6C1.STL, 80 mm) get one marker on each of 3 faces; footage of
them is turned into fruit-classifier training data by training/composite_set2_real.py
(detect marker quad -> composite a fruit texture over it -> auto-label). Cube k carries
ids [3k, 3k+1, 3k+2] so faces/cubes stay distinguishable in the metadata.

All physical parameters come from configs/set2.yaml -> aruco.

    yolo/bin/python training/make_aruco_markers.py            # -> assets/aruco/
    yolo/bin/python training/make_aruco_markers.py --marker-mm 60   # override size

Output: assets/aruco/markers.pdf (+ per-page PNGs). Print at 100% scale ("actual
size") on MATTE paper, verify with the printed 100 mm ruler, cut along the gray
80 mm squares, glue one square flat onto each of 3 faces per cube.
"""

import argparse
import os

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "set2.yaml")
OUT_DIR = os.path.join(ROOT, "assets", "aruco")

DPI = 300
MM = DPI / 25.4                       # pixels per mm
A4_W, A4_H = int(210 * MM), int(297 * MM)
GRID_COLS, GRID_ROWS = 2, 3           # 80 mm cut squares per A4 page


def marker_image(dictionary, marker_id, side_px):
    """Crisp marker: generate at an exact multiple of the module count, then
    NEAREST-resize to the requested print size (keeps module edges sharp)."""
    n_modules = 4 + 2                 # DICT_4X4_* : 4 bits + 2 border modules
    gen_px = n_modules * max(1, side_px // n_modules)
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, gen_px)
    return cv2.resize(img, (side_px, side_px), interpolation=cv2.INTER_NEAREST)


def draw_ruler(page, x0, y0, length_mm=100):
    """100 mm calibration ruler so the user can verify the print is at 100% scale."""
    x1 = x0 + int(length_mm * MM)
    cv2.line(page, (x0, y0), (x1, y0), 0, 3)
    for mm10 in range(0, length_mm + 1, 10):
        x = x0 + int(mm10 * MM)
        cv2.line(page, (x, y0 - int(3 * MM)), (x, y0), 0, 3)
    cv2.putText(page, f"{length_mm} mm  (verify after printing at 100% / actual size)",
                (x0, y0 + int(6 * MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)


def make_pages(aruco_cfg, marker_mm):
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, aruco_cfg["dictionary"]))
    face_mm = aruco_cfg["face_size_m"] * 1000.0
    n_markers = aruco_cfg["n_cubes"] * aruco_cfg["faces_per_cube"]

    face_px = int(round(face_mm * MM))
    marker_px = int(round(marker_mm * MM))
    per_page = GRID_COLS * GRID_ROWS
    margin_x = (A4_W - GRID_COLS * face_px) // 2
    margin_y = int(18 * MM)

    pages = []
    for page_i in range((n_markers + per_page - 1) // per_page):
        page = np.full((A4_H, A4_W), 255, np.uint8)
        cv2.putText(page,
                    f"Set2 ArUco {aruco_cfg['dictionary']}  marker {marker_mm:.0f} mm"
                    f"  face {face_mm:.0f} mm  page {page_i + 1}",
                    (margin_x, int(10 * MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2,
                    cv2.LINE_AA)
        for slot in range(per_page):
            marker_id = page_i * per_page + slot
            if marker_id >= n_markers:
                break
            r, c = divmod(slot, GRID_COLS)
            x0 = margin_x + c * face_px
            y0 = margin_y + r * (face_px + int(8 * MM))
            # gray cut line = one full cube face (glued flat, covers the face)
            cv2.rectangle(page, (x0, y0), (x0 + face_px, y0 + face_px), 160, 2)
            off = (face_px - marker_px) // 2
            page[y0 + off:y0 + off + marker_px,
                 x0 + off:x0 + off + marker_px] = marker_image(
                     dictionary, marker_id, marker_px)
            cube, face = divmod(marker_id, aruco_cfg["faces_per_cube"])
            cv2.putText(page, f"id {marker_id}  (cube {cube + 1}, face {face + 1})",
                        (x0, y0 + face_px + int(5 * MM)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
        draw_ruler(page, margin_x, A4_H - int(12 * MM))
        pages.append(page)
    return pages, dictionary, n_markers


def self_test(pages, dictionary, n_markers):
    """Detect every marker back from the rendered pages before trusting the print."""
    det = cv2.aruco.ArucoDetector(dictionary)
    found = set()
    for page in pages:
        _, ids, _ = det.detectMarkers(page)
        if ids is not None:
            found.update(int(i) for i in ids.flatten())
    missing = set(range(n_markers)) - found
    if missing:
        raise RuntimeError(f"self-test failed, undetected ids: {sorted(missing)}")
    print(f"self-test OK: all {n_markers} markers detected on the rendered pages")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-mm", type=float, default=None,
                    help="override aruco.marker_size_m (in mm)")
    args = ap.parse_args()

    with open(CFG) as f:
        aruco_cfg = yaml.safe_load(f)["aruco"]
    marker_mm = args.marker_mm or aruco_cfg["marker_size_m"] * 1000.0

    pages, dictionary, n_markers = make_pages(aruco_cfg, marker_mm)
    self_test(pages, dictionary, n_markers)

    os.makedirs(OUT_DIR, exist_ok=True)
    pil_pages = []
    for i, page in enumerate(pages):
        png = os.path.join(OUT_DIR, f"markers_page{i + 1}.png")
        cv2.imwrite(png, page)
        # 1-bit pages: PIL's PDF writer needs no JPEG encoder for mode "1", and the
        # sheet is B/W anyway. Threshold at 200 keeps the gray cut lines (160) black.
        pil_pages.append(Image.fromarray(page).point(lambda v: 255 if v > 200 else 0)
                         .convert("1", dither=Image.Dither.NONE))
        print("wrote", os.path.relpath(png, ROOT))
    pdf = os.path.join(OUT_DIR, "markers.pdf")
    pil_pages[0].save(pdf, save_all=True, append_images=pil_pages[1:],
                      resolution=DPI)
    print("wrote", os.path.relpath(pdf, ROOT))
    print(f"\nPRINT at 100% scale (actual size), MATTE paper; verify the 100 mm ruler,"
          f"\ncut along gray {aruco_cfg['face_size_m'] * 1000:.0f} mm squares, glue one"
          f" per face on 3 faces of each cube.")


if __name__ == "__main__":
    main()
