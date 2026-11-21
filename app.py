"""
Aplikasi Streamlit — Deteksi Kendaraan & Deteksi Garis Jalan (Lane Detection)
Menggunakan OpenCV (classic computer vision pipeline, tanpa deep learning).

Fitur:
- Upload video jalan raya
- Deteksi garis jalan: Grayscale -> Gaussian Blur -> Canny -> ROI mask -> Hough Transform
- Deteksi kendaraan: pilihan antara Background Subtraction (MOG2) atau Haar Cascade
- Mode "Uji Frame" untuk tuning parameter secara instan sebelum proses full video
- Proses full video dengan progress bar, lalu hasil bisa ditonton & diunduh

Jalankan dengan:
    streamlit run app.py
"""

import os
import time
import tempfile
import subprocess
import urllib.request

import cv2
import numpy as np
import streamlit as st


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Deteksi Kendaraan & Garis Jalan",
    page_icon="🚗",
    layout="wide",
)

CASCADE_URL = "https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml"
CASCADE_DIR = os.path.join(tempfile.gettempdir(), "vehicle_lane_app")
CASCADE_PATH = os.path.join(CASCADE_DIR, "cars.xml")


# =========================================================
# UTIL: DOWNLOAD HAAR CASCADE (dengan cache lokal)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_haar_cascade():
    """Unduh & muat cascade classifier untuk deteksi mobil (di-cache)."""
    os.makedirs(CASCADE_DIR, exist_ok=True)
    try:
        if not os.path.exists(CASCADE_PATH):
            urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if cascade.empty():
            return None
        return cascade
    except Exception:
        return None


# =========================================================
# LANE DETECTION PIPELINE
# =========================================================
def build_roi_mask(shape, bottom_width_pct, top_width_pct, top_height_pct):
    """Buat mask trapesium (region of interest) di area bawah frame (jalan)."""
    h, w = shape[0], shape[1]
    top_y = int(h * (1 - top_height_pct))

    bottom_left = (int(w * (0.5 - bottom_width_pct / 2)), h)
    bottom_right = (int(w * (0.5 + bottom_width_pct / 2)), h)
    top_left = (int(w * (0.5 - top_width_pct / 2)), top_y)
    top_right = (int(w * (0.5 + top_width_pct / 2)), top_y)

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    return mask, vertices


def average_lane_lines(lines, shape, top_height_pct):
    """Pisahkan garis kiri/kanan berdasarkan slope, lalu rata-ratakan & ekstrapolasi."""
    h, w = shape[0], shape[1]
    left_fit, right_fit = [], []

    if lines is None:
        return []

    for line in lines:
        coords = np.asarray(line).reshape(-1)
        if coords.size != 4:
            continue
        x1, y1, x2, y2 = coords.astype(float)
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if abs(slope) < 0.3:  # buang garis nyaris horizontal (bukan lane)
            continue
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    y1 = h
    y2 = int(h * (1 - top_height_pct))
    result = []

    for fit_group in (left_fit, right_fit):
        if len(fit_group) == 0:
            continue
        slope, intercept = np.mean(fit_group, axis=0)
        if slope == 0:
            continue
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        result.append((x1, y1, x2, y2))

    return result


def detect_lanes(frame, params):
    """Jalankan pipeline deteksi garis jalan dan kembalikan frame beranotasi."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, params["canny_low"], params["canny_high"])

    mask, roi_vertices = build_roi_mask(
        frame.shape, params["bottom_width_pct"], params["top_width_pct"], params["top_height_pct"]
    )
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=params["hough_threshold"],
        minLineLength=params["min_line_length"],
        maxLineGap=params["max_line_gap"],
    )

    overlay = np.zeros_like(frame)

    if params["smooth_lanes"]:
        avg_lines = average_lane_lines(lines, frame.shape, params["top_height_pct"])
        for (x1, y1, x2, y2) in avg_lines:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 10)
    else:
        if lines is not None:
            for line in lines:
                coords = np.asarray(line).reshape(-1)
                if coords.size != 4:
                    continue
                x1, y1, x2, y2 = coords.astype(float)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 5)

    result = cv2.addWeighted(frame, 1.0, overlay, 0.9, 0)

    if params["show_roi"]:
        cv2.polylines(result, roi_vertices, isClosed=True, color=(255, 255, 0), thickness=2)

    if params["show_edges_debug"]:
        edges_bgr = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        small = cv2.resize(edges_bgr, (frame.shape[1] // 4, frame.shape[0] // 4))
        result[10:10 + small.shape[0], 10:10 + small.shape[1]] = small

    return result


# =========================================================
# VEHICLE DETECTION — BACKGROUND SUBTRACTION (MOG2)
# =========================================================
def create_background_subtractor(history, var_threshold):
    return cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=True
    )


def detect_vehicles_bgsub(frame, back_sub, params, roi_mask=None):
    fg_mask = back_sub.apply(frame)
    # buang bayangan (nilai 127) dan noise
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    if roi_mask is not None:
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = frame.copy()
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < params["min_area"] or area > params["max_area"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio < params["min_aspect"] or aspect_ratio > params["max_aspect"]:
            continue
        count += 1
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, f"Kendaraan {count}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return result, count


# =========================================================
# VEHICLE DETECTION — HAAR CASCADE
# =========================================================
def remove_overlapping_boxes(boxes, iou_threshold=0.3):
    """Buang kotak yang berpotongan berat sehingga satu mobil tidak dihitung berkali-kali."""
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

    order = np.argsort(y2)
    keep = []
    while len(order) > 0:
        current = order[-1]
        keep.append(current)
        order = order[:-1]

        if len(order) == 0:
            break

        xx1 = np.maximum(x1[current], x1[order])
        yy1 = np.maximum(y1[current], y1[order])
        xx2 = np.minimum(x2[current], x2[order])
        yy2 = np.minimum(y2[current], y2[order])

        intersection_width = np.maximum(0, xx2 - xx1)
        intersection_height = np.maximum(0, yy2 - yy1)
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[order] - intersection
        iou = intersection / np.maximum(union, 1)

        order = order[iou <= iou_threshold]

    return boxes[keep].tolist()


def detect_vehicles_haar(frame, cascade, params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cars = cascade.detectMultiScale(
        gray,
        scaleFactor=params["scale_factor"],
        minNeighbors=params["min_neighbors"],
        minSize=(params["min_size"], params["min_size"]),
    )

    frame_h, frame_w = frame.shape[:2]
    boundary_margin = params.get("boundary_margin", 0)

    filtered_cars = []
    for x, y, w, h in cars:
        if x <= boundary_margin or y <= boundary_margin:
            continue
        if x + w >= frame_w - boundary_margin or y + h >= frame_h - boundary_margin:
            continue

        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 4.5:
            continue

        filtered_cars.append((x, y, w, h))

    filtered_cars = remove_overlapping_boxes(filtered_cars, iou_threshold=0.3)

    result = frame.copy()
    for i, (x, y, w, h) in enumerate(filtered_cars, start=1):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(result, f"Mobil {i}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

    return result, len(filtered_cars)


# =========================================================
# GABUNGAN PIPELINE UNTUK SATU FRAME
# =========================================================
def process_frame(frame, mode, lane_params, vehicle_method, vehicle_params, back_sub=None, cascade=None):
    output = frame.copy()
    vehicle_count = None

    if mode in ("Deteksi Garis Jalan", "Keduanya"):
        output = detect_lanes(output, lane_params)

    if mode in ("Deteksi Kendaraan", "Keduanya"):
        if vehicle_method == "Background Subtraction (MOG2)" and back_sub is not None:
            output, vehicle_count = detect_vehicles_bgsub(output, back_sub, vehicle_params)
        elif vehicle_method == "Haar Cascade" and cascade is not None:
            output, vehicle_count = detect_vehicles_haar(output, cascade, vehicle_params)

    return output, vehicle_count


def reencode_for_browser(input_path, output_path):
    """Re-encode video ke H.264 supaya bisa diputar di st.video / browser."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


# =========================================================
# SIDEBAR — PENGATURAN
# =========================================================
def main():
    st.title("🚗 Deteksi Kendaraan & Garis Jalan (OpenCV + Streamlit)")
    st.caption("Pipeline klasik computer vision: Canny + Hough Transform untuk garis jalan, "
               "Background Subtraction / Haar Cascade untuk kendaraan.")

    with st.sidebar:
        st.header("⚙️ Pengaturan")

        uploaded_file = st.file_uploader("Upload video jalan raya", type=["mp4", "avi", "mov", "mkv"])

        mode = st.radio(
            "Mode Deteksi",
            ["Keduanya", "Deteksi Garis Jalan", "Deteksi Kendaraan"],
            index=0,
        )

        st.divider()
        st.subheader("🛣️ Parameter Garis Jalan")
        with st.expander("Canny Edge Detection", expanded=False):
            canny_low = st.slider("Canny Threshold Rendah", 0, 255, 50)
            canny_high = st.slider("Canny Threshold Tinggi", 0, 255, 150)

        with st.expander("Region of Interest (ROI)", expanded=False):
            bottom_width_pct = st.slider("Lebar bawah ROI (%)", 0.1, 1.0, 0.85)
            top_width_pct = st.slider("Lebar atas ROI (%)", 0.05, 1.0, 0.25)
            top_height_pct = st.slider("Tinggi ROI dari bawah (%)", 0.1, 0.9, 0.4)
            show_roi = st.checkbox("Tampilkan garis ROI", value=True)

        with st.expander("Hough Transform", expanded=False):
            hough_threshold = st.slider("Hough Threshold", 5, 200, 20)
            min_line_length = st.slider("Panjang Garis Minimum", 5, 200, 20)
            max_line_gap = st.slider("Jarak Celah Maksimum", 1, 200, 30)
            smooth_lanes = st.checkbox("Rata-ratakan garis kiri/kanan (smoothing)", value=True)

        show_edges_debug = st.checkbox("Tampilkan preview edge (debug)", value=False)

        lane_params = {
            "canny_low": canny_low,
            "canny_high": canny_high,
            "bottom_width_pct": bottom_width_pct,
            "top_width_pct": top_width_pct,
            "top_height_pct": top_height_pct,
            "show_roi": show_roi,
            "hough_threshold": hough_threshold,
            "min_line_length": min_line_length,
            "max_line_gap": max_line_gap,
            "smooth_lanes": smooth_lanes,
            "show_edges_debug": show_edges_debug,
        }

        st.divider()
        st.subheader("🚙 Parameter Deteksi Kendaraan")
        vehicle_method = st.selectbox(
            "Metode Deteksi Kendaraan",
            ["Background Subtraction (MOG2)", "Haar Cascade"],
            index=0,
            help=(
                "Background Subtraction: cocok untuk kamera statis (CCTV), mendeteksi objek bergerak.\n\n"
                "Haar Cascade: mendeteksi bentuk mobil langsung per-frame, cocok untuk kamera bergerak (dashcam)."
            ),
        )

        if vehicle_method == "Background Subtraction (MOG2)":
            with st.expander("Parameter MOG2", expanded=False):
                mog_history = st.slider("History (jumlah frame background)", 50, 1000, 300)
                mog_var_threshold = st.slider("Var Threshold (sensitivitas)", 4, 100, 16)
                min_area = st.slider("Luas Kontur Minimum (px²)", 100, 20000, 1200)
                max_area = st.slider("Luas Kontur Maksimum (px²)", 1000, 200000, 60000)
                min_aspect = st.slider("Aspect Ratio Minimum", 0.1, 3.0, 0.5)
                max_aspect = st.slider("Aspect Ratio Maksimum", 0.5, 5.0, 3.0)
            vehicle_params = {
                "min_area": min_area, "max_area": max_area,
                "min_aspect": min_aspect, "max_aspect": max_aspect,
            }
        else:
            with st.expander("Parameter Haar Cascade", expanded=False):
                scale_factor = st.slider("Scale Factor", 1.01, 1.5, 1.1, step=0.01)
                min_neighbors = st.slider("Min Neighbors", 1, 20, 4)
                min_size = st.slider("Ukuran Objek Minimum (px)", 10, 200, 40)
                boundary_margin = st.slider("Margin tepi kotak deteksi (px)", 0, 100, 0)
            vehicle_params = {
                "scale_factor": scale_factor,
                "min_neighbors": min_neighbors,
                "min_size": min_size,
                "boundary_margin": boundary_margin,
            }
            mog_history, mog_var_threshold = 300, 16  # default tak dipakai

        st.divider()
        st.subheader("⚡ Performa")
        resize_width = st.select_slider(
            "Resize lebar frame (px) — makin kecil makin cepat",
            options=[320, 480, 640, 854, 960, 1280, "Asli"],
            value=960,
        )
        frame_skip = st.slider("Proses setiap-n frame (skip frame)", 1, 5, 1)


    # =========================================================
    # AREA UTAMA
    # =========================================================
    if uploaded_file is None:
        st.info("⬆️ Silakan upload video terlebih dahulu dari sidebar untuk memulai.")
        st.markdown(
            """
            **Cara kerja aplikasi ini:**
            1. **Deteksi Garis Jalan** — Grayscale → Gaussian Blur → Canny Edge Detection →
               masking Region of Interest (ROI) → Hough Line Transform → rata-ratakan garis kiri/kanan.
            2. **Deteksi Kendaraan** — dua pilihan metode:
               - *Background Subtraction (MOG2)*: memisahkan objek bergerak dari background,
                 cocok untuk video kamera statis/CCTV.
               - *Haar Cascade*: classifier klasik yang mendeteksi bentuk mobil langsung per-frame,
                 lebih cocok untuk video dashcam (kamera ikut bergerak).
            3. Gunakan tab **"Uji 1 Frame"** untuk mengatur parameter secara instan sebelum memproses
               seluruh video (supaya tidak menunggu lama baru tahu hasilnya kurang pas).
            """
        )
        st.stop()

    # simpan upload ke file temp agar bisa dibuka cv2.VideoCapture
    if "video_path" not in st.session_state or st.session_state.get("video_name") != uploaded_file.name:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        tfile.close()
        st.session_state["video_path"] = tfile.name
        st.session_state["video_name"] = uploaded_file.name

    video_path = st.session_state["video_path"]

    cap_info = cv2.VideoCapture(video_path)
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_info.get(cv2.CAP_PROP_FPS) or 25
    orig_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps else 0
    cap_info.release()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Resolusi Asli", f"{orig_w}x{orig_h}")
    col_b.metric("Total Frame", f"{total_frames}")
    col_c.metric("FPS", f"{fps:.1f}")
    col_d.metric("Durasi", f"{duration:.1f} s")

    # siapkan cascade sekali (di-cache)
    cascade = load_haar_cascade() if vehicle_method == "Haar Cascade" else None
    if vehicle_method == "Haar Cascade" and cascade is None:
        st.warning(
            "⚠️ Gagal mengunduh Haar Cascade untuk deteksi mobil (kemungkinan tidak ada akses internet "
            "di server). Silakan gunakan metode 'Background Subtraction (MOG2)' sebagai alternatif."
        )

    tab_test, tab_process = st.tabs(["🔍 Uji 1 Frame (tuning parameter)", "🎬 Proses Seluruh Video"])

    # ---------------------------------------------------------
    # TAB 1 — UJI 1 FRAME
    # ---------------------------------------------------------
    with tab_test:
        st.write("Geser slider di bawah untuk memilih posisi frame, lalu atur parameter di sidebar "
                 "sampai hasil deteksi terlihat pas. Untuk metode *Background Subtraction*, model background "
                 "dibangun dari beberapa frame sebelum posisi yang dipilih.")

        test_frame_idx = st.slider("Posisi Frame", 0, max(total_frames - 1, 0), min(int(fps * 2), total_frames - 1))

        if st.button("▶️ Jalankan Uji pada Frame Ini", type="primary"):
            cap = cv2.VideoCapture(video_path)

            test_back_sub = None
            if vehicle_method == "Background Subtraction (MOG2)":
                test_back_sub = create_background_subtractor(mog_history, mog_var_threshold)
                # warming up background model dari beberapa frame sebelumnya
                warmup_start = max(0, test_frame_idx - 40)
                cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
                for _ in range(test_frame_idx - warmup_start):
                    ret, f = cap.read()
                    if not ret:
                        break
                    if resize_width != "Asli":
                        f = cv2.resize(f, (resize_width, int(resize_width * orig_h / orig_w)))
                    test_back_sub.apply(f)

            cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                st.error("Tidak bisa membaca frame tersebut.")
            else:
                if resize_width != "Asli":
                    frame = cv2.resize(frame, (resize_width, int(resize_width * orig_h / orig_w)))

                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Frame Asli", width="stretch")

                output, vcount = process_frame(
                    frame, mode, lane_params, vehicle_method, vehicle_params,
                    back_sub=test_back_sub, cascade=cascade,
                )
                with col2:
                    caption = "Hasil Deteksi"
                    if vcount is not None:
                        caption += f" — {vcount} kendaraan terdeteksi"
                    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption=caption, width="stretch")

    # ---------------------------------------------------------
    # TAB 2 — PROSES SELURUH VIDEO
    # ---------------------------------------------------------
    with tab_process:
        st.write("Proses seluruh video dengan pengaturan saat ini. Untuk video panjang, "
                 "gunakan opsi *resize* dan *skip frame* di sidebar agar lebih cepat.")

        max_seconds = duration if duration > 0 else 60.0
        time_range = st.slider(
            "Rentang waktu yang diproses (detik)",
            0.0, float(max_seconds), (0.0, float(max_seconds)), step=0.5,
        )

        start_frame = int(time_range[0] * fps)
        end_frame = int(time_range[1] * fps)

        if st.button("🚀 Mulai Proses Video", type="primary"):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            out_w = resize_width if resize_width != "Asli" else orig_w
            out_h = int(out_w * orig_h / orig_w)

            raw_out_path = os.path.join(tempfile.gettempdir(), "processed_raw.mp4")
            final_out_path = os.path.join(tempfile.gettempdir(), "processed_final.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(raw_out_path, fourcc, fps / frame_skip, (out_w, out_h))

            back_sub = create_background_subtractor(mog_history, mog_var_threshold) \
                if vehicle_method == "Background Subtraction (MOG2)" else None

            progress_bar = st.progress(0, text="Memproses video...")
            preview_slot = st.empty()

            n_total = max(end_frame - start_frame, 1)
            frame_idx = 0
            vehicle_counts = []
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_pos = start_frame + frame_idx
                if current_pos >= end_frame:
                    break

                if frame_idx % frame_skip == 0:
                    frame_resized = cv2.resize(frame, (out_w, out_h))
                    output, vcount = process_frame(
                        frame_resized, mode, lane_params, vehicle_method, vehicle_params,
                        back_sub=back_sub, cascade=cascade,
                    )
                    if vcount is not None:
                        vehicle_counts.append(vcount)
                    writer.write(output)

                    if frame_idx % (frame_skip * 10) == 0:
                        preview_slot.image(
                            cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                            caption=f"Preview — frame {current_pos}",
                            width="stretch",
                        )

                frame_idx += 1
                progress_bar.progress(
                    min(frame_idx / n_total, 1.0),
                    text=f"Memproses frame {frame_idx}/{n_total}...",
                )

            cap.release()
            writer.release()
            elapsed = time.time() - start_time

            st.success(f"✅ Selesai memproses {frame_idx} frame dalam {elapsed:.1f} detik.")
            if vehicle_counts:
                st.write(
                    f"📊 Rata-rata kendaraan terdeteksi per frame: **{np.mean(vehicle_counts):.1f}**, "
                    f"maksimum: **{int(np.max(vehicle_counts))}**"
                )

            with st.spinner("Mengonversi video agar bisa diputar di browser..."):
                ok = reencode_for_browser(raw_out_path, final_out_path)

            display_path = final_out_path if ok else raw_out_path
            if os.path.exists(display_path):
                st.video(display_path)
                with open(display_path, "rb") as f:
                    st.download_button(
                        "⬇️ Unduh Video Hasil",
                        data=f.read(),
                        file_name="hasil_deteksi.mp4",
                        mime="video/mp4",
                    )
            else:
                st.error("Gagal menghasilkan file video output.")


if __name__ == "__main__":
    main()
