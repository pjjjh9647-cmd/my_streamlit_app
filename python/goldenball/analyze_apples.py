# -*- coding: utf-8 -*-
"""
Apple Shape Grading (Top + Side paired by index)
- Pair rule: up[i] ↔ side[i]  (자연정렬)
- Robust image loading (Pillow-only)  / NumPy 2.0 대응 (np.ptp)
- Ruler px/mm estimation (top strip)
- 3-step segmentation (nonwhite → HSV green/yellow → Lab-a Otsu)
- PCA-based axes, aspect ratio, circularity
- Strong symmetry metrics (LR/UD) with composite score (sym_score)
- View-wise pass → Final pass, CSV + failure log + debug images

Run:
    pip install opencv-python pillow numpy pandas scipy
    python analyze_apples.py
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
from scipy.signal import find_peaks
import re

# =========================
# Paths (수정 지점)
# =========================
ROOT = Path(r"C:\goldenball")
DIR_UP   = ROOT / "up"
DIR_SIDE = ROOT / "side"
OUT_CSV  = ROOT / "apple_pairs_results.csv"
FAIL_CSV = ROOT / "apple_pairs_failures.csv"
DBG_DIR  = ROOT / "_debug_pairs"; DBG_DIR.mkdir(exist_ok=True)

# =========================
# Thresholds (현장에 맞게 조정 가능)
# =========================
# Top view (정원형/대칭/원형도)
TOP_ASPECT_MIN, TOP_ASPECT_MAX = 0.95, 1.05
TOP_CIRC_MIN = 0.80
TOP_SYM_SCORE_MAX = 0.45   # 좌우 대칭 종합 스코어 (0=대칭, 1=비대칭)

# Side view (높이/직경 비율 + 좌우/상하 대칭)
SIDE_SHAPE_MIN, SIDE_SHAPE_MAX = 0.90, 1.10
SIDE_SYM_SCORE_LR_MAX = 0.55   # 좌우
SIDE_SYM_SCORE_UD_MAX = 0.55   # 상하

# Segmentation / contour sanity
MIN_CONTOUR_AREA = 5000

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

def estimate_px_per_mm(img):
    """상단 스트립에서 눈금자 피크 간격으로 px/mm 추정. 실패 시 NaN."""
    try:
        h, w = img.shape[:2]
        strip_h = int(0.13*h)
        strip = img[:strip_h, :]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        proj = np.abs(sobelx).sum(axis=0)
        proj_norm = (proj - proj.min()) / (np.ptp(proj) + 1e-6)  # NumPy 2.0
        peaks, _ = find_peaks(proj_norm, prominence=0.05, distance=3)
        if len(peaks) >= 10:
            spacings = np.diff(peaks)
            return float(np.median(spacings))
    except Exception:
        pass
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

def pick_apple_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best, best_score = None, -1
    for c in cnts:
        A = cv2.contourArea(c)
        if A < MIN_CONTOUR_AREA:
            continue
        P = cv2.arcLength(c, True)
        circ = 4*np.pi*A/(P**2) if P>0 else 0.0
        score = A * (0.5 + 0.5*circ)  # 면적+원형도 (길쭉한 물체 억제)
        if score > best_score:
            best_score, best = score, c
    return best

# -------------------------
# Metrics
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
    """
    mask_rot: PCA 주축 정렬된 이진 마스크(255)
    axis: "vertical"(좌우), "horizontal"(상하)
    return: sym_width_pct, sym_iou, sym_dist, sym_score (0=대칭, 1=비대칭)
    """
    H, W = mask_rot.shape
    m = mask_rot.copy()

    if axis == "vertical":
        top = int(H*trim_ratio); bottom = int(H*(1-trim_ratio))
        m = m[top:bottom, :]
        center = m.shape[1]//2
        left  = m[:, :center]
        right = m[:, center:]
        right_mirror = cv2.flip(right, 1)  # 좌우 반사

        diffs = []
        for y in range(m.shape[0]):
            row = m[y]
            xs = np.where(row>0)[0]
            if xs.size:
                lw = center - xs.min()
                rw = xs.max() - center
                avg = 0.5*(lw+rw) + 1e-6
                diffs.append(abs(lw-rw)/avg)
        sym_width_pct = float(np.mean(diffs)*100) if diffs else np.nan

        left_bin  = (left>0).astype(np.uint8)
        right_bin = (right_mirror>0).astype(np.uint8)
        inter = np.logical_and(left_bin, right_bin).sum()
        union = np.logical_or(left_bin, right_bin).sum() + 1e-6
        sym_iou = float(inter/union)

        dl = cv2.distanceTransform(255-left_bin*255, cv2.DIST_L2, 3)
        dr = cv2.distanceTransform(255-right_bin*255, cv2.DIST_L2, 3)
        sym_dist = float(np.mean(np.abs(dl - dr)))

    else:  # horizontal 상하대칭
        center = m.shape[0]//2
        up   = m[:center, :]
        down = m[center:, :]
        down_mirror = cv2.flip(down, 0)  # 상하 반사

        diffs = []
        for x in range(m.shape[1]):
            col = m[:, x]
            ys = np.where(col>0)[0]
            if ys.size:
                uw = center - ys.min()
                dw = ys.max() - center
                avg = 0.5*(uw+dw) + 1e-6
                diffs.append(abs(uw-dw)/avg)
        sym_width_pct = float(np.mean(diffs)*100) if diffs else np.nan

        up_bin   = (up>0).astype(np.uint8)
        down_bin = (down_mirror>0).astype(np.uint8)
        inter = np.logical_and(up_bin, down_bin).sum()
        union = np.logical_or(up_bin, down_bin).sum() + 1e-6
        sym_iou = float(inter/union)

        du = cv2.distanceTransform(255-up_bin*255, cv2.DIST_L2, 3)
        dd = cv2.distanceTransform(255-down_bin*255, cv2.DIST_L2, 3)
        sym_dist = float(np.mean(np.abs(du - dd)))

    # 정규화 → 종합 스코어
    sym_width_norm = np.clip((sym_width_pct/20.0), 0, 1) if not np.isnan(sym_width_pct) else 1.0
    sym_iou_norm   = np.clip(1.0 - sym_iou, 0, 1)
    sym_dist_norm  = np.clip(sym_dist/5.0, 0, 1)  # 데이터에 맞게 조정 가능
    sym_score = 0.5*sym_width_norm + 0.3*sym_iou_norm + 0.2*sym_dist_norm

    return {
        "sym_width_pct": sym_width_pct,
        "sym_iou": sym_iou,
        "sym_dist": sym_dist,
        "sym_score": float(sym_score)
    }

def compute_view_metrics(img, contour, px_per_mm):
    # 면적/둘레/원형도 + PCA 축 길이
    A = cv2.contourArea(contour)
    P = cv2.arcLength(contour, True)
    circularity = 4*np.pi*A/(P**2) if P>0 else np.nan

    mean, eigvecs, Lmaj, Lmin = pca_axes(contour)
    aspect = (Lmin / Lmaj) if Lmaj>0 else np.nan

    # 주축 정렬 마스크로 대칭 지표 (LR/UD)
    rot_mask = rotated_mask_from_contour(img, contour)
    sym_lr = symmetry_metrics(rot_mask, axis="vertical",   trim_ratio=0.2)
    sym_ud = symmetry_metrics(rot_mask, axis="horizontal", trim_ratio=0.2)

    # mm 환산
    major_mm = float(Lmaj/px_per_mm) if not np.isnan(px_per_mm) else np.nan
    minor_mm = float(Lmin/px_per_mm) if not np.isnan(px_per_mm) else np.nan

    met = dict(
        major_axis_mm=None if np.isnan(major_mm) else major_mm,
        minor_axis_mm=None if np.isnan(minor_mm) else minor_mm,
        aspect_ratio=float(aspect),
        circularity=float(circularity),
        px_per_mm_est=None if np.isnan(px_per_mm) else float(px_per_mm),
        # 대칭 상세(LR/UD)
        sym_width_pct_lr=sym_lr["sym_width_pct"],
        sym_iou_lr=sym_lr["sym_iou"],
        sym_dist_lr=sym_lr["sym_dist"],
        sym_score_lr=sym_lr["sym_score"],
        sym_width_pct_ud=sym_ud["sym_width_pct"],
        sym_iou_ud=sym_ud["sym_iou"],
        sym_dist_ud=sym_ud["sym_dist"],
        sym_score_ud=sym_ud["sym_score"],
    )
    return met

def analyze_one_view(img):
    """단일 뷰 분석: 회전 보정 → px/mm → 3-스텝 분할 → 컨투어 → 지표"""
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 오른쪽으로 90도 돌아갔다 가정
    px_per_mm = estimate_px_per_mm(img)

    last_mask, last_stage = None, "none"
    for seg_fn in (seg_nonwhite, seg_hsv_green_yellow, seg_lab_a):
        mask, stage = seg_fn(img)
        last_mask, last_stage = mask, stage
        cnt = pick_apple_contour(mask)
        if cnt is not None:
            met = compute_view_metrics(img, cnt, px_per_mm)
            return met, img, mask, cnt, stage
    # 실패
    return None, img, (last_mask if last_mask is not None else np.zeros(img.shape[:2], np.uint8)), None, last_stage

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
# Pairing & Main
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

        # 각 뷰 분석
        up_met, up_rot, up_mask, up_cnt, up_stage = analyze_one_view(up_img)
        save_debug_pair(i+1, "up", up_rot, up_mask, up_cnt, up_stage)
        side_met, side_rot, side_mask, side_cnt, side_stage = analyze_one_view(side_img)
        save_debug_pair(i+1, "side", side_rot, side_mask, side_cnt, side_stage)

        if (up_met is None) or (side_met is None):
            reason = []
            if up_met is None: reason.append("up 분할 실패")
            if side_met is None: reason.append("side 분할 실패")
            fails.append((f"pair{i+1}", "; ".join(reason), str(up_path), str(side_path)))
            continue

        # --- 뷰별 판정 ---
        up_ok = (
            (TOP_ASPECT_MIN <= up_met["aspect_ratio"] <= TOP_ASPECT_MAX) and
            (up_met["circularity"] >= TOP_CIRC_MIN) and
            (up_met["sym_score_lr"] <= TOP_SYM_SCORE_MAX)   # 좌우 대칭(Top)
        )

        # 옆뷰: shape_index ≒ aspect_ratio (높이/직경)
        side_ok = (
            (SIDE_SHAPE_MIN <= side_met["aspect_ratio"] <= SIDE_SHAPE_MAX) and
            (side_met["sym_score_lr"] <= SIDE_SYM_SCORE_LR_MAX) and   # 좌우
            (side_met["sym_score_ud"] <= SIDE_SYM_SCORE_UD_MAX)       # 상하
        )

        final_ok = up_ok and side_ok
        final_label = "정형과" if final_ok else "비정형"

        rows.append({
            "pair_id": i+1,
            "up_file":   up_path.name,
            "side_file": side_path.name,

            # Top metrics
            "top_major_mm": up_met["major_axis_mm"],
            "top_minor_mm": up_met["minor_axis_mm"],
            "top_aspect":   up_met["aspect_ratio"],
            "top_circularity": up_met["circularity"],
            "top_px_per_mm_est": up_met["px_per_mm_est"],
            "top_sym_score_lr":  up_met["sym_score_lr"],
            "top_sym_score_ud":  up_met["sym_score_ud"],

            # Side metrics
            "side_major_mm": side_met["major_axis_mm"],
            "side_minor_mm": side_met["minor_axis_mm"],
            "side_shape_index(aspect)": side_met["aspect_ratio"],
            "side_circularity": side_met["circularity"],
            "side_px_per_mm_est": side_met["px_per_mm_est"],
            "side_sym_score_lr": side_met["sym_score_lr"],
            "side_sym_score_ud": side_met["sym_score_ud"],

            # Decision
            "최종_판정": final_label
        })

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
