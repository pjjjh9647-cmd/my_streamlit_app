# -*- coding: utf-8 -*-
r"""
Apple Shape Grading (Final: L/D single criterion)
- Up/Side index pairing (C:\goldenball\up, C:\goldenball\side)
- Ruler px/mm estimation (top strip)
- 3-step segmentation (nonwhite → HSV green/yellow → Lab-a Otsu)
- PCA-based axes, aspect, circularity
- Symmetry (LR/UD): width-based stats + IoU + distance + composite score
- Area-based LR/UD difference (%)
- FINAL DECISION = ONLY L/D (side height / top diameter_max) with threshold 0.87

Usage:
    pip install opencv-python pillow numpy pandas scipy
    python analyze_apples.py
"""

import cv2, re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
from scipy.signal import find_peaks

# =========================
# Paths / Settings
# =========================
ROOT = Path(r"C:\goldenball")
DIR_UP   = ROOT / "up"
DIR_SIDE = ROOT / "side"
OUT_CSV  = ROOT / "apple_pairs_results.csv"
FAIL_CSV = ROOT / "apple_pairs_failures.csv"
DBG_DIR  = ROOT / "_debug_pairs"; DBG_DIR.mkdir(exist_ok=True)
PROF_DIR = DBG_DIR / "profiles"; PROF_DIR.mkdir(exist_ok=True)

RULER_STRIP_RATIO = 0.13
MIN_CONTOUR_AREA  = 5000

# ---- 자 눈금 설정 및 검증 한계 ----
RULER_MM_PER_TICK = 1.0   # 눈금 1칸의 실제 mm. 5mm 간격 자면 5.0으로 변경
PX_SPACING_MIN = 6        # 눈금 간 최소 픽셀 간격 하한
PX_SPACING_MAX = 60       # 눈금 간 최대 픽셀 간격 상한
CV_MAX = 0.35             # 간격 일관성(CV) 허용 상한

# 필요 시 상단 스트립을 조금 더 얇게
RULER_STRIP_RATIO = 0.08  # 기존 0.13에서 조정


# ---- L/D 단독 기준 ----
LD_THRESHOLD = 0.87  # L/D >= 0.87 → 정형과

# =========================
# Utilities
# =========================
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def naturalsort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images_sorted(folder: Path):
    files = [p for p in folder.iterdir() if p.suffix.lower() in EXTS and p.is_file()]
    files.sort(key=lambda p: naturalsort_key(p.name))
    return files

def imread_pillow_only(p: Path):
    with Image.open(p) as pil:
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _pxmm_from_strip(strip, axis=0):
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    if axis == 0:  # 수직 눈금(가로방향 미분)
        g = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        proj = np.abs(g).sum(axis=0)
    else:          # 수평 눈금(세로방향 미분)
        g = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        proj = np.abs(g).sum(axis=1)
    proj = (proj - proj.min()) / (np.ptp(proj) + 1e-6)

    peaks, _ = find_peaks(proj, prominence=0.10, distance=PX_SPACING_MIN)
    if len(peaks) < 8:
        return float("nan"), float("inf")

    spac = np.diff(peaks).astype(np.float32)
    spac = spac[(spac >= PX_SPACING_MIN) & (spac <= PX_SPACING_MAX)]
    if len(spac) < 6:
        return float("nan"), float("inf")

    med = float(np.median(spac))
    cv  = float(np.std(spac) / (med + 1e-6))
    # 눈금 1칸이 1mm가 아니면 그만큼 나눠서 px/mm 환산
    return med / max(RULER_MM_PER_TICK, 1e-6), cv

def estimate_px_per_mm(img, strip_ratio: float = RULER_STRIP_RATIO):
    try:
        h, w = img.shape[:2]
        sh = max(4, int(strip_ratio * h))
        sw = max(4, int(strip_ratio * w))
        top_strip  = img[:sh, :]
        left_strip = img[:, :sw]

        cand = []
        px1, cv1 = _pxmm_from_strip(top_strip, axis=0)   # 상단 스트립
        if not np.isnan(px1) and cv1 <= CV_MAX:
            cand.append((px1, cv1))
        px2, cv2 = _pxmm_from_strip(left_strip, axis=1)  # 좌측 스트립
        if not np.isnan(px2) and cv2 <= CV_MAX:
            cand.append((px2, cv2))

        if not cand:
            return float("nan")
        cand.sort(key=lambda t: t[1])  # CV가 작은 후보를 채택
        return cand[0][0]
    except Exception:
        return float("nan")


# -------------------------
# Segmentation (3-step)
# -------------------------
def seg_nonwhite(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]; v = hsv[:,:,2]
    mask = ((s > 20) & (v < 255)).astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask, "nonwhite"

def seg_hsv_green_yellow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([20, 40, 40]); upper1 = np.array([45,255,255])
    lower2 = np.array([35, 30, 30]); upper2 = np.array([85,255,255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask, "hsv_green_yellow"

def seg_lab_a(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a = cv2.GaussianBlur(lab[:,:,1], (5,5), 0)
    _, mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask, "lab_a_otsu"

def pick_apple_contour(mask, min_area=MIN_CONTOUR_AREA):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best, best_score = None, -1
    for c in cnts:
        A = cv2.contourArea(c)
        if A < min_area:
            continue
        P = cv2.arcLength(c, True)
        circ = 4*np.pi*A/(P**2) if P>0 else 0.0
        score = A * (0.5 + 0.5*circ)  # 면적+원형도
        if score > best_score:
            best_score, best = score, c
    return best

# -------------------------
# Geometry / Metrics
# -------------------------
def pca_axes(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = pts.mean(axis=0)
    X = pts - mean
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    proj_major = X @ eigvecs[:,0]
    proj_minor = X @ eigvecs[:,1]
    Lmaj = proj_major.max() - proj_major.min()
    Lmin = proj_minor.max() - proj_minor.min()
    return mean, eigvecs, float(Lmaj), float(Lmin)

def rotated_mask_from_contour(img, contour):
    h, w = img.shape[:2]
    mean, eigvecs, _, _ = pca_axes(contour)
    theta = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
    M = cv2.getRotationMatrix2D(tuple(mean), theta, 1.0)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return rot

def symmetry_metrics(mask_rot, axis="vertical", trim_ratio=0.2):
    """폭 기반 대칭: mean/max/p95 %, IoU, distance, composite score(0~1) + 프로파일."""
    H, W = mask_rot.shape
    m = mask_rot.copy()
    width_div = 20.0
    dist_div  = 5.0

    if axis == "vertical":
        top = int(H*trim_ratio); bottom = int(H*(1-trim_ratio))
        m = m[top:bottom, :]
        center = m.shape[1]//2
        left  = m[:, :center]
        right = m[:, center:]
        right_mirror = cv2.flip(right, 1)

        row_errs = []
        for y in range(m.shape[0]):
            row = m[y]
            xs = np.where(row>0)[0]
            if xs.size:
                lw = center - xs.min()
                rw = xs.max() - center
                avg = 0.5*(lw+rw) + 1e-6
                row_errs.append(abs(lw-rw)/avg)

        sym_width_pct = float(np.mean(row_errs)*100) if row_errs else np.nan
        width_mean_pct = sym_width_pct
        width_max_pct  = float(np.max(row_errs)*100) if row_errs else np.nan
        width_p95_pct  = float(np.percentile(row_errs, 95)*100) if row_errs else np.nan

        left_bin  = (left>0).astype(np.uint8)
        right_bin = (right_mirror>0).astype(np.uint8)
        inter = np.logical_and(left_bin, right_bin).sum()
        union = np.logical_or(left_bin, right_bin).sum() + 1e-6
        sym_iou = float(inter/union)

        dl = cv2.distanceTransform(255-left_bin*255, cv2.DIST_L2, 3)
        dr = cv2.distanceTransform(255-right_bin*255, cv2.DIST_L2, 3)
        sym_dist = float(np.mean(np.abs(dl - dr)))
        profile = row_errs

    else:
        center = m.shape[0]//2
        up   = m[:center, :]
        down = m[center:, :]
        down_mirror = cv2.flip(down, 0)

        col_errs = []
        for x in range(m.shape[1]):
            col = m[:, x]
            ys = np.where(col>0)[0]
            if ys.size:
                uw = center - ys.min()
                dw = ys.max() - center
                avg = 0.5*(uw+dw) + 1e-6
                col_errs.append(abs(uw-dw)/avg)

        sym_width_pct = float(np.mean(col_errs)*100) if col_errs else np.nan
        width_mean_pct = sym_width_pct
        width_max_pct  = float(np.max(col_errs)*100) if col_errs else np.nan
        width_p95_pct  = float(np.percentile(col_errs, 95)*100) if col_errs else np.nan

        up_bin   = (up>0).astype(np.uint8)
        down_bin = (down_mirror>0).astype(np.uint8)
        inter = np.logical_and(up_bin, down_bin).sum()
        union = np.logical_or(up_bin, down_bin).sum() + 1e-6
        sym_iou = float(inter/union)

        du = cv2.distanceTransform(255-up_bin*255, cv2.DIST_L2, 3)
        dd = cv2.distanceTransform(255-down_bin*255, cv2.DIST_L2, 3)
        sym_dist = float(np.mean(np.abs(du - dd)))
        profile = col_errs

    sym_width_norm = np.clip((sym_width_pct/width_div), 0, 1) if not np.isnan(sym_width_pct) else 1.0
    sym_iou_norm   = np.clip(1.0 - sym_iou, 0, 1)
    sym_dist_norm  = np.clip(sym_dist/dist_div, 0, 1)
    sym_score = 0.5*sym_width_norm + 0.3*sym_iou_norm + 0.2*sym_dist_norm

    return {
        "sym_width_pct": sym_width_pct,
        "sym_iou": sym_iou,
        "sym_dist": sym_dist,
        "sym_score": float(sym_score),
        "width_mean_pct": width_mean_pct,
        "width_max_pct": width_max_pct,
        "width_p95_pct": width_p95_pct,
        "_profile_pct": [float(x*100) for x in profile]
    }

def area_diff_percent(mask_rot, axis="vertical"):
    """반쪽 면적 차이 % (전체 균형). 0이면 완전 대칭."""
    m = mask_rot.copy()
    H, W = m.shape
    if axis=="vertical":
        c = W//2
        L = (m[:, :c] > 0).sum()
        R = (m[:, c:] > 0).sum()
    else:
        c = H//2
        U = (m[:c, :] > 0).sum()
        D = (m[c:, :] > 0).sum()
        L, R = U, D
    mean = 0.5*(L+R) + 1e-6
    return float(abs(L-R)/mean*100.0)

def compute_view_metrics(img, contour, px_per_mm):
    A = cv2.contourArea(contour)
    P = cv2.arcLength(contour, True)
    circularity = 4*np.pi*A/(P**2) if P>0 else np.nan

    mean, eigvecs, Lmaj, Lmin = pca_axes(contour)
    aspect = (Lmin / Lmaj) if Lmaj>0 else np.nan

    rot_mask = rotated_mask_from_contour(img, contour)
    sym_lr = symmetry_metrics(rot_mask, axis="vertical",   trim_ratio=0.2)
    sym_ud = symmetry_metrics(rot_mask, axis="horizontal", trim_ratio=0.2)

    area_lr_pct = area_diff_percent(rot_mask, axis="vertical")
    area_ud_pct = area_diff_percent(rot_mask, axis="horizontal")

    major_mm = float(Lmaj/px_per_mm) if not np.isnan(px_per_mm) else np.nan
    minor_mm = float(Lmin/px_per_mm) if not np.isnan(px_per_mm) else np.nan

    met = dict(
        major_axis_mm=None if np.isnan(major_mm) else major_mm,
        minor_axis_mm=None if np.isnan(minor_mm) else minor_mm,
        aspect_ratio=float(aspect),
        circularity=float(circularity),
        px_per_mm_est=None if np.isnan(px_per_mm) else float(px_per_mm),

        sym_width_pct_lr=sym_lr["sym_width_pct"],
        sym_iou_lr=sym_lr["sym_iou"],
        sym_dist_lr=sym_lr["sym_dist"],
        sym_score_lr=sym_lr["sym_score"],
        lr_width_mean_pct=sym_lr["width_mean_pct"],
        lr_width_max_pct=sym_lr["width_max_pct"],
        lr_width_p95_pct=sym_lr["width_p95_pct"],

        sym_width_pct_ud=sym_ud["sym_width_pct"],
        sym_iou_ud=sym_ud["sym_iou"],
        sym_dist_ud=sym_ud["sym_dist"],
        sym_score_ud=sym_ud["sym_score"],
        ud_width_mean_pct=sym_ud["width_mean_pct"],
        ud_width_max_pct=sym_ud["width_max_pct"],
        ud_width_p95_pct=sym_ud["width_p95_pct"],

        area_diff_lr_percent=area_lr_pct,
        area_diff_ud_percent=area_ud_pct,
    )
    return met, rot_mask

def analyze_one_view(img):
    """회전 보정 → px/mm → 분할 → 컨투어 → 지표"""
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 오른쪽으로 90도 돌아갔다 가정
    px_per_mm = estimate_px_per_mm(img, RULER_STRIP_RATIO)

    last_mask, last_stage = None, "none"
    for seg_fn in (seg_nonwhite, seg_hsv_green_yellow, seg_lab_a):
        mask, stage = seg_fn(img)
        last_mask, last_stage = mask, stage
        cnt = pick_apple_contour(mask, MIN_CONTOUR_AREA)
        if cnt is not None:
            met, rot_mask = compute_view_metrics(img, cnt, px_per_mm)
            return met, img, mask, cnt, stage, rot_mask
    return None, img, (last_mask if last_mask is not None else np.zeros(img.shape[:2], np.uint8)), None, last_stage, np.zeros(img.shape[:2], np.uint8)

def save_debug_pair(idx, view, img_rot, mask, contour, stage):
    out = DBG_DIR / f"pair{idx:04d}_{view}_{stage}.png"
    vis = img_rot.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0,0,255), 3)
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), 2)
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    H = max(vis.shape[0], m3.shape[0])
    pad_vis = np.zeros((H, vis.shape[1], 3), dtype=np.uint8)
    pad_m3  = np.zeros((H, m3.shape[1], 3), dtype=np.uint8)
    pad_vis[:vis.shape[0], :vis.shape[1]] = vis
    pad_m3[:m3.shape[0], :m3.shape[1]] = m3
    cat = np.concatenate([pad_vis, pad_m3], axis=1)
    cv2.putText(cat, f"{view} stage={stage}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
    cv2.imwrite(str(out), cat)

# =========================
# Main
# =========================
def main():
    ups   = list_images_sorted(DIR_UP)
    sides = list_images_sorted(DIR_SIDE)

    n = min(len(ups), len(sides))
    print(f"매칭 쌍 수: {n} (up={len(ups)}, side={len(sides)})")
    if n == 0:
        print("up/side 폴더를 확인하세요.")
        return

    rows, fails = [], []

    for i in range(n):
        up_path, side_path = ups[i], sides[i]
        try:
            up_img   = imread_pillow_only(up_path)
            side_img = imread_pillow_only(side_path)
        except Exception as e:
            fails.append((f"pair{i+1}", f"이미지 읽기 실패: {e}", str(up_path), str(side_path)))
            continue

        # 분석
        up_met, up_rot, up_mask, up_cnt, up_stage, _ = analyze_one_view(up_img)
        save_debug_pair(i+1, "up", up_rot, up_mask, up_cnt, up_stage)
        side_met, side_rot, side_mask, side_cnt, side_stage, _ = analyze_one_view(side_img)
        save_debug_pair(i+1, "side", side_rot, side_mask, side_cnt, side_stage)

        if (up_met is None) or (side_met is None):
            reason = []
            if up_met is None: reason.append("up 분할 실패")
            if side_met is None: reason.append("side 분할 실패")
            fails.append((f"pair{i+1}", "; ".join(reason), str(up_path), str(side_path)))
            continue

        # --- L/D 계산 ---
        top_major_mm = up_met["major_axis_mm"]
        top_minor_mm = up_met["minor_axis_mm"]
        side_major_mm = side_met["major_axis_mm"]  # 옆뷰 높이 = L

        LD_ratio = None
        ld_reason = None
        if (top_major_mm is None) or (top_minor_mm is None) or (side_major_mm is None):
            ld_reason = "L/D 계산 불가(px/mm 추정 실패)"
        else:
            D = max(top_major_mm, top_minor_mm)  # 위뷰 큰 지름
            L = side_major_mm                    # 옆뷰 높이
            if D > 0:
                LD_ratio = float(L / D)
            else:
                ld_reason = "D가 0 이하"

        # --- 최종 판정: L/D 단독 기준 ---
        if LD_ratio is None:
            final_label = "미판정"
        else:
            final_label = "정형과" if (LD_ratio >= LD_THRESHOLD) else "비정형"

        # --- row 작성 ---
        row = {
            "pair_id": i+1,
            "up_file":   up_path.name,
            "side_file": side_path.name,

            # Top
            "top_major_mm": top_major_mm,
            "top_minor_mm": top_minor_mm,
            "top_aspect":   up_met["aspect_ratio"],
            "top_circularity": up_met["circularity"],
            "top_px_per_mm_est": up_met["px_per_mm_est"],
            "top_sym_score_lr":  up_met["sym_score_lr"],
            "top_sym_score_ud":  up_met["sym_score_ud"],
            "top_lr_width_mean_pct": up_met["lr_width_mean_pct"],
            "top_lr_width_max_pct":  up_met["lr_width_max_pct"],
            "top_lr_width_p95_pct":  up_met["lr_width_p95_pct"],
            "top_ud_width_mean_pct": up_met["ud_width_mean_pct"],
            "top_ud_width_max_pct":  up_met["ud_width_max_pct"],
            "top_ud_width_p95_pct":  up_met["ud_width_p95_pct"],
            "top_area_diff_lr_percent":  up_met["area_diff_lr_percent"],
            "top_area_diff_ud_percent":  up_met["area_diff_ud_percent"],

            # Side
            "side_major_mm": side_major_mm,
            "side_minor_mm": side_met["minor_axis_mm"],
            "side_shape_index(aspect)": side_met["aspect_ratio"],
            "side_circularity": side_met["circularity"],
            "side_px_per_mm_est": side_met["px_per_mm_est"],
            "side_sym_score_lr": side_met["sym_score_lr"],
            "side_sym_score_ud": side_met["sym_score_ud"],
            "side_lr_width_mean_pct": side_met["lr_width_mean_pct"],
            "side_lr_width_max_pct":  side_met["lr_width_max_pct"],
            "side_lr_width_p95_pct":  side_met["lr_width_p95_pct"],
            "side_ud_width_mean_pct": side_met["ud_width_mean_pct"],
            "side_ud_width_max_pct":  side_met["ud_width_max_pct"],
            "side_ud_width_p95_pct":  side_met["ud_width_p95_pct"],
            "side_area_diff_lr_percent":  side_met["area_diff_lr_percent"],
            "side_area_diff_ud_percent":  side_met["area_diff_ud_percent"],

            # L/D 단독 기준
            "LD_ratio": LD_ratio,
            "최종_판정": final_label
        }

        # 참고: 옆뷰 좌우 대칭률(%) = 100 - 좌우 폭차 평균(%)
        if side_met["lr_width_mean_pct"] is not None:
            row["side_lr_symmetry_percent"] = max(0.0, 100.0 - float(side_met["lr_width_mean_pct"]))
        else:
            row["side_lr_symmetry_percent"] = None

        # rows 적재
        rows.append(row)

        # L/D 계산 불가 케이스는 실패 로그에도 남김
        if (LD_ratio is None) and ld_reason:
            fails.append((f"pair{i+1}", ld_reason, str(up_path), str(side_path)))

    # 저장
    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print("결과 저장:", OUT_CSV)
    else:
        print("성공한 매칭 결과가 없습니다.")

    if fails:
        pd.DataFrame(fails, columns=["pair_id","reason","up_file","side_file"]).to_csv(FAIL_CSV, index=False, encoding="utf-8-sig")
        print("실패 로그 저장:", FAIL_CSV)

    print("디버그 이미지는 여기 저장됨:", DBG_DIR)

if __name__ == "__main__":
    main()
