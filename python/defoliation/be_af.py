
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import hashlib

def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path: Path, img, ext_hint=".jpg"):
    ext = path.suffix.lower() if path.suffix else ext_hint
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in [".jpg", ".jpeg"] else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"imencode 실패: {path}")
    # 한글 경로 대응
    buf.tofile(str(path))

def ascii_name(stem: str, suffix: str) -> str:
    # 한글 파일명이 실패할 때를 대비한 ASCII 대체 이름
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{suffix}_overlay_{h}.jpg".encode("ascii", "ignore").decode() or f"id_{h}_{suffix}.jpg"

def imwrite_unicode(path: Path, img, ext_hint=".jpg"):
    # 확장자 결정
    ext = path.suffix.lower() if path.suffix else ext_hint
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        ext = ".jpg"
    # 인코드 후 tofile 사용 (유니코드 경로 안전)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in [".jpg", ".jpeg"] else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"imencode 실패: {path}")
    buf.tofile(str(path))

def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)   # Unicode 경로 안전
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

# -------------------------
# 1) 경로 설정 (필요 시 수정)
# -------------------------
BASE_DIR   = Path(r"C:\Users\User\Desktop\mba\defoliation")
BEFORE_DIR = BASE_DIR / "적엽전"
AFTER_DIR  = BASE_DIR / "적엽후"
OUT_DIR    = BASE_DIR / "_out"
OVL_DIR    = OUT_DIR / "overlays"

OUT_DIR.mkdir(exist_ok=True, parents=True)
OVL_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------
# 2) 유틸: 잎 세그먼트
#    - 색상기반(HSV + Lab) 규칙을 혼합
#    - 과실/배경(노란 과실, 흰 방풍막, 하늘) 억제
# -------------------------
def segment_leaves(img_bgr: np.ndarray) -> np.ndarray:

    # 리사이즈(너무 큰 이미지라면 2000px 상한)
    h, w = img_bgr.shape[:2]
    max_side = 2000
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # 블러로 노이즈 약화
    blur = cv2.GaussianBlur(img_bgr, (5,5), 0)

    # HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Lab (a*가 낮을수록 green)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
    Lc, ac, bc = cv2.split(lab)

    # 2-1) 그린 후보 (HSV 범위)
    # OpenCV H: 0~179 (대략 35~85가 녹색권)
    green_h1 = cv2.inRange(H, 35, 85)
    green_s  = cv2.inRange(S, 50, 255)   # 충분한 채도
    green_v  = cv2.inRange(V, 40, 255)   # 너무 어둡지 않게
    green_hsv = cv2.bitwise_and(green_h1, cv2.bitwise_and(green_s, green_v))

    # 2-2) Lab에서 green 강화 (a* 작을수록 green)
    # a*의 중앙값(=128) 기준으로 여유를 두고 임계
    lab_green = cv2.inRange(ac, 0, 135)  # 135 이하인 픽셀을 green 후보로

    # 2-3) 노란 과실 억제 (HSV에서 황색: H~20~35)
    yellow = cv2.inRange(H, 20, 34)
    # 과실은 S와 V가 비교적 높음
    yellow = cv2.bitwise_and(yellow, cv2.inRange(S, 40, 255))
    yellow = cv2.bitwise_and(yellow, cv2.inRange(V, 80, 255))

    # 2-4) 하늘/흰막 억제 (파란 하늘: H~90~130, 흰막: S 낮고 V/L 밝음)
    sky = cv2.inRange(H, 90, 130)
    white_bg = cv2.bitwise_and(cv2.inRange(S, 0, 60), cv2.inRange(V, 180, 255))
    # 밝은 배경 완화: L*도 높으면 배경으로 본다
    bright_L = cv2.inRange(Lc, 200, 255)
    bg = cv2.bitwise_or(sky, cv2.bitwise_or(white_bg, bright_L))

    # 2-5) 통합: green - (yellow + bg)
    green_raw = cv2.bitwise_and(green_hsv, lab_green)
    not_yellow = cv2.bitwise_not(yellow)
    not_bg = cv2.bitwise_not(bg)
    leaf0 = cv2.bitwise_and(green_raw, cv2.bitwise_and(not_yellow, not_bg))

    # 2-6) 형태학적 보정
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    leaf = cv2.morphologyEx(leaf0, cv2.MORPH_OPEN, k, iterations=1)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, k, iterations=2)

    # 작은 잡영 제거
    cnts, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(leaf.shape, np.uint8)
    for c in cnts:
        if cv2.contourArea(c) >= 150:  # 픽셀 기준, 필요 시 조정
            cv2.drawContours(mask, [c], -1, 255, -1)

    return mask.astype(bool), img_bgr

# -------------------------
# 3) ROI 생성: 전/후 잎 마스크의 합집합을 팽창
#    동일 ROI에서의 비율 비교 → 카메라 위치 변화 영향 완화
# -------------------------
def make_roi(mask_before: np.ndarray, mask_after: np.ndarray) -> np.ndarray:
    union = np.logical_or(mask_before, mask_after).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41,41))
    roi = cv2.dilate(union, k, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k, iterations=1)
    # 구멍 채우기
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(roi)
    for c in cnts:
        if cv2.contourArea(c) >= 2000:
            cv2.drawContours(canvas, [c], -1, 255, -1)
    return canvas.astype(bool)

# -------------------------
# 4) 오버레이 저장
# -------------------------
def save_overlay(stem: str, img_bgr: np.ndarray, mask: np.ndarray, roi: np.ndarray, suffix: str):
    vis = img_bgr.copy()
    roi_cnts, _ = cv2.findContours(roi.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, roi_cnts, -1, (200,200,200), 2)

    leaf_cnts, _ = cv2.findContours(mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, leaf_cnts, -1, (0,180,0), 2)

    out_path = OVL_DIR / f"{stem}_{suffix}_overlay.jpg"
    try:
        imwrite_unicode(out_path, vis, ext_hint=".jpg")
        print(f"[saved] {out_path}")
    except Exception as e:
        alt = OVL_DIR / ascii_name(stem, suffix)
        imwrite_unicode(alt, vis, ext_hint=".jpg")
        print(f"[saved-alt] {alt} (원래이름 저장 실패: {e})")


# -------------------------
# 5) 페어링 및 계산
# -------------------------
def get_pairs():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    before = {p.stem.lower(): p for p in BEFORE_DIR.iterdir() if p.suffix.lower() in exts}
    after  = {p.stem.lower(): p for p in AFTER_DIR.iterdir()  if p.suffix.lower() in exts}
    keys = sorted(set(before.keys()) & set(after.keys()))
    return [(before[k], after[k]) for k in keys]

def main():
    pairs = get_pairs()
    if not pairs:
        print("매칭되는 전/후 파일이 없습니다. 폴더와 확장자를 확인하세요.")
        return

    rows = []
    for b_path, a_path in pairs:
        stem = b_path.stem
        img_b = imread_unicode(b_path)
        img_a = imread_unicode(a_path)
        if img_b is None or img_a is None:
            print(f"[경고] 읽기 실패: {stem}")
            continue

        mask_b, img_b_resized = segment_leaves(img_b)
        mask_a, img_a_resized = segment_leaves(img_a)

        # ROI(전/후 공통)
        roi = make_roi(mask_b, mask_a)

        leaf_px_b = int(mask_b.sum())
        leaf_px_a = int(mask_a.sum())
        roi_px    = int(roi.sum()) if roi.sum() > 0 else img_b_resized.shape[0]*img_b_resized.shape[1]

        cov_b = leaf_px_b / roi_px if roi_px else 0.0
        cov_a = leaf_px_a / roi_px if roi_px else 0.0

        delta_px = leaf_px_a - leaf_px_b
        # 감소율(전 기준): 음수면 감소, 양수면 증가
        change_pct = (leaf_px_a - leaf_px_b) / max(1, leaf_px_b) * 100.0
        change_cov_pct = (cov_a - cov_b) / max(1e-9, cov_b) * 100.0 if cov_b>0 else np.nan

        rows.append({
            "file": stem,
            "leaf_px_before": leaf_px_b,
            "leaf_px_after":  leaf_px_a,
            "delta_px":       delta_px,
            "change_pct_vs_before(%)": round(change_pct, 2),
            "roi_px":         roi_px,
            "coverage_before(leaf/roi)": round(cov_b, 6),
            "coverage_after(leaf/roi)":  round(cov_a, 6),
            "coverage_change_pct(%)":    round(change_cov_pct, 2) if not np.isnan(change_cov_pct) else np.nan
        })

        # 오버레이 저장
        save_overlay(stem, img_b_resized, mask_b, roi, "before")
        save_overlay(stem, img_a_resized, mask_a, roi, "after")

    df = pd.DataFrame(rows).sort_values("file")
    out_csv = OUT_DIR / "result.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"완료: {out_csv}")
    print(f"오버레이: {OVL_DIR}")

if __name__ == "__main__":
    main()
