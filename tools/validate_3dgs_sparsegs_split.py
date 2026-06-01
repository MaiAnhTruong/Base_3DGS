import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.auto_split_3dgs import validate_split_report


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _count_images(path):
    if not os.path.isdir(path):
        return 0
    count = 0
    for _, _, files in os.walk(path):
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Validate a SparseGS-style 3DGS posed-view split.")
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--expected_train_views", type=int, default=0)
    parser.add_argument("--expected_hold", type=int, default=8)
    parser.add_argument("--require_sparsegs_triangulate", action="store_true")
    args = parser.parse_args()

    split_root = os.path.abspath(args.split_root)
    report_path = os.path.join(split_root, "reports", "split_report.json")
    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))

    check("split_report.json exists", os.path.isfile(report_path))
    if not os.path.isfile(report_path):
        for name, passed in checks:
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print("[SPLIT-VALIDATE] status=FAIL")
        return 1

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    full_view_count = int(report.get("full_view_count", 0))
    expected_test = len(range(0, full_view_count, int(args.expected_hold)))
    train_source_path = report.get("train_source_path", os.path.join(split_root, "train"))
    test_source_path = report.get("test_source_path", os.path.join(split_root, "test"))
    train_sparse = os.path.join(train_source_path, "sparse", "0")
    test_sparse = os.path.join(test_source_path, "sparse", "0")
    sparsegs = report.get("sparsegs_triangulate", {})
    contamination = report.get("contamination_guard", {})

    check("status=PASS", report.get("status") == "PASS")
    check("split_unit=COLMAP_posed_view", report.get("split_unit") == "COLMAP_posed_view")
    check("test_view_count == len(range(0, full_view_count, hold))", int(report.get("test_view_count", -1)) == expected_test)
    if args.expected_train_views > 0:
        check("train_view_count == expected_train_views", int(report.get("train_view_count", -1)) == int(args.expected_train_views))
    check("overlap_by_image_name_count == 0", int(report.get("overlap_by_image_name_count", -1)) == 0)
    check("overlap_by_colmap_image_id_count == 0", int(report.get("overlap_by_colmap_image_id_count", -1)) == 0)
    check("duplicate_train_name_count == 0", int(report.get("duplicate_train_name_count", -1)) == 0)
    check("duplicate_test_name_count == 0", int(report.get("duplicate_test_name_count", -1)) == 0)
    check("local_cluster_guard_pass == true", report.get("sequence_coverage", {}).get("local_cluster_guard_pass") is True)
    check("train/images count == train_view_count", _count_images(os.path.join(train_source_path, "images")) == int(report.get("train_view_count", -1)))
    check("test/images count == test_view_count", _count_images(os.path.join(test_source_path, "images")) == int(report.get("test_view_count", -1)))
    check("train/sparse/0/cameras.bin exists", os.path.isfile(os.path.join(train_sparse, "cameras.bin")))
    check("train/sparse/0/images.bin exists", os.path.isfile(os.path.join(train_sparse, "images.bin")))
    check("train/sparse/0/points3D.bin exists", os.path.isfile(os.path.join(train_sparse, "points3D.bin")))
    check("train/sparse/0/points3D.ply exists", os.path.isfile(os.path.join(train_sparse, "points3D.ply")))
    check("test/sparse/0/cameras.bin exists", os.path.isfile(os.path.join(test_sparse, "cameras.bin")))
    check("test/sparse/0/images.bin exists", os.path.isfile(os.path.join(test_sparse, "images.bin")))
    check("train_contact_sheet.jpg exists", os.path.isfile(os.path.join(split_root, "reports", "train_contact_sheet.jpg")))
    check("test_contact_sheet.jpg exists", os.path.isfile(os.path.join(split_root, "reports", "test_contact_sheet.jpg")))
    check("camera_split_plot.png exists", os.path.isfile(os.path.join(split_root, "reports", "camera_split_plot.png")))
    check("selected_views.csv exists", os.path.isfile(os.path.join(split_root, "reports", "selected_views.csv")))

    if args.require_sparsegs_triangulate:
        check("sparsegs_triangulate.status == PASS", sparsegs.get("status") == "PASS")
        check("full_points3D_used_for_train_initialization == false", sparsegs.get("full_points3D_used_for_train_initialization") is False)
        check(
            "triangulated_point_count >= split_min_triangulated_points",
            int(sparsegs.get("triangulated_point_count", 0)) >= int(report.get("split_min_triangulated_points", 100)),
        )
        check("contamination_guard.test_rgb_used_for_training == false", contamination.get("test_rgb_used_for_training") is False)
        check("contamination_guard.internal_llffhold_disabled == true", contamination.get("internal_llffhold_disabled") is True)

    library_validation = validate_split_report(report)
    check("library validation PASS", library_validation.get("status") == "PASS")

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"[SPLIT-VALIDATE] status={'PASS' if not failed else 'FAIL'}")
    if failed:
        print("[SPLIT-VALIDATE] failed_checks=" + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

