#!/usr/bin/env python3
# process.py  ─────────────────────────────────────────────────────────────
"""
Stage A : derive rotated / axis-aligned boxes from raw frame annotations
Stage B : convert Stage-A CSVs to YOLO-txt files
Run one or both stages with  --stage {boxes|yolo|all}
"""

import argparse, sys, cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm                      # 🆕 progress bars
import bbox_functions                      # your helper module

# ─────────────────────────────── CLI & constants ─────────────────────────
SESSION_DICT = {1: "0800_0830", 2: "0830_0900", 3: "0900_0930",
                4: "0930_1000", 5: "1000_1030"}

CLASS_MAP = {
    "Car": 0, "Motorcycle": 1, "Bus": 2, "Taxi": 3,
    "Medium Vehicle": 4, "Heavy Vehicle": 5,
}

def parse_cli():
    p = argparse.ArgumentParser("Build boxes → (optional) YOLO export")
    p.add_argument("--base_dir", required=True)
    p.add_argument("--drone",    type=int, choices=range(1, 11), default=6)
    p.add_argument("--session",  type=int, choices=range(1, 6),  default=3)
    p.add_argument("--rotate_bbox", action="store_true")
    p.add_argument("--dont_show",   action="store_true")
    p.add_argument("--stage", choices=("all", "boxes", "yolo"), default="all")
    return p.parse_args()

# ─────────────────────────── helpers & I/O paths ─────────────────────────
def ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

def build_dirs(base: Path, drone: int, sess: int, rotated: bool):
    root = base / f"20181029_D{drone}_{SESSION_DICT[sess]}"
    return {
        "session"     : root,
        "frames"      : root / "Frames",
        "annots"      : root / "Annotations",
        "labels_box"  : root / ("Labels_rotated" if rotated else "Labels_axis_aligned"),
        "labels_yolo" : root / "Labels_YOLO_upright",      # 🆕 renamed folder
    }

# ──────────────────────────── Stage A : boxes ────────────────────────────
def stage_build_boxes(dirs, rotate, show):
    frames = sorted(dirs["frames"].glob("*.jpg"))
    ensure(dirs["labels_box"])
    print(f"[Stage A] writing CSVs → {dirs['labels_box'].name}")

    for f in tqdm(frames, desc="frames", unit="img"):
        stem = f.stem
        img  = cv2.imread(str(f))
        h, w = img.shape[:2]
        df   = pd.read_csv(dirs["annots"] / f"{stem}.csv")
        df   = df[~df["Type"].isin(("Bicycle", "Undefined", "Pedestrian"))]

        if rotate:
            df = df.assign(p1=None, p2=None, p3=None, p4=None)
        else:
            df = df.assign(cx=None, cy=None, box_w=None, box_h=None)

        for i, row in df.iterrows():
            rect = bbox_functions.create_bbox_for_vehicles(
                       row["Type"], (row["x_img [px]"], row["y_img [px]"]),
                       row["Angle_img [rad]"])
            rect = bbox_functions.adjust_bbox_for_crop(
                       rect, row["Type"], (0, 0), w, h, 0.2)
            if rect is None:
                continue

            if rotate:
                r = np.round(rect).astype(int)
                df.loc[i, ["p1", "p2", "p3", "p4"]] = [*r[:3], r[-1]]
                if show:
                    cv2.drawContours(img, [r], 0, (0, 255, 0), 2)
            else:
                x0, y0, bw, bh = np.round(
                    cv2.boundingRect(rect.astype(np.float32)))
                df.loc[i, ["cx", "cy", "box_w", "box_h"]] = [x0, y0, bw, bh]
                if show:
                    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh),
                                  (0, 255, 0), 2)

        df.to_csv(dirs["labels_box"] / f"{stem}_{'rotated' if rotate else 'upright'}.csv",
                  index=False)

        if show:
            cv2.imshow("boxes", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows(); sys.exit(0)

    print("✔ Stage A done")

# ──────────────────────────── Stage B : YOLO ─────────────────────────────
def csv_to_yolo(dirs, img_wh=(3840, 2100)):
    ensure(dirs["labels_yolo"])
    csvs = sorted(dirs["labels_box"].glob("*.csv"))
    print(f"[Stage B] converting {len(csvs)} CSVs → {dirs['labels_yolo'].name}")

    for csv_f in tqdm(csvs, desc="csv", unit="file"):
        df = pd.read_csv(csv_f)

        # remove optional '_upright' suffix from stem
        stem = csv_f.stem[:-8] if csv_f.stem.endswith("_upright") else csv_f.stem
        txt_f = dirs["labels_yolo"] / f"{stem}.txt"

        with open(txt_f, "w") as out:
            for _, r in df.iterrows():
                cid = CLASS_MAP.get(r["Type"])
                if cid is None:           # skip unknown classes
                    continue
                cx, cy, bw, bh = r[["cx", "cy", "box_w", "box_h"]]
                cx /= img_wh[0]; cy /= img_wh[1]; bw /= img_wh[0]; bh /= img_wh[1]
                out.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print("✔ Stage B done")

# ────────────────────────────────── main ─────────────────────────────────
if __name__ == "__main__":
    ARGS = parse_cli()
    dirs = build_dirs(Path(ARGS.base_dir), ARGS.drone, ARGS.session, ARGS.rotate_bbox)

    if ARGS.stage in ("all", "boxes"):
        stage_build_boxes(dirs, ARGS.rotate_bbox, not ARGS.dont_show)

    if ARGS.stage in ("all", "yolo"):
        if ARGS.rotate_bbox:
            print("[warn] YOLO export needs axis-aligned boxes "
                  "→ forcing rotate_bbox=False for Stage B")
            # ensure Stage B reads the AA folder
            dirs["labels_box"] = build_dirs(Path(ARGS.base_dir),
                                            ARGS.drone, ARGS.session,
                                            rotated=False)["labels_box"]
        csv_to_yolo(dirs)
