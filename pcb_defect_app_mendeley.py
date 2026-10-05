# ============================================================
# AUTOMATED PCB DEFECT DETECTION
# Digital Image Processing Based Project  (Python / Streamlit)
#
# Dataset: "PCB-Defect: An Annotated Dataset for Surface Defect
#          Detection in Printed Circuit Boards" (Mendeley Data,
#          DOI 10.17632/vdj74sngvn.1)
#          230 colour photos (800x600 .. 6000x4000), COCO JSON boxes,
#          6 classes: missing pad, mouse bite, open circuit,
#          short circuit, spur, spurious copper
#
# This dataset has NO defect-free reference image per board, so the
# "reference" is generated from the test image itself (grayscale
# opening + closing sized to the trace width). All other steps of
# the original pipeline are unchanged.
#
# Run:   streamlit run pcb_defect_app_mendeley.py
# Put the dataset images (and the COCO .json) inside ./PCBData
# ============================================================

import os
import glob
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# RESEARCH PAPER REFERENCE
# ============================================================
# Methodological reference:
# S. H. Indera Putera and Zuwairie Ibrahim,
# "Printed Circuit Board Defect Detection Using Mathematical
# Morphology and MATLAB Image Processing Tools," IEEE ICETC 2010.
# DOI: 10.1109/ICETC.2010.5530052
#
# In this project, the classical morphology + binary comparison
# idea is adapted to the RGB PCB-Defect dataset. Because this
# dataset does not provide a defect-free template for every board,
# an expected clean reference is generated from the selected PCB.
# The IEEE-inspired mode binarizes the generated reference and the
# test image and compares them using XOR before morphology and
# connected-component analysis.
# ============================================================

# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCB Defect Inspection",
    page_icon="🔧",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔧 Automated PCB Defect Detection and Localization")

st.success(
    "Research methodology: IEEE ICETC 2010 — Indera Putera & Zuwairie Ibrahim. "
    "The implementation adapts the classical mathematical-morphology and binary "
    "reference/test comparison concept to the RGB PCB-Defect dataset."
)

with st.expander("📚 IEEE Research Paper Used in This Project", expanded=True):
    st.markdown(
        "**S. H. Indera Putera & Zuwairie Ibrahim (IEEE ICETC 2010)**\n\n"
        "*Printed Circuit Board Defect Detection Using Mathematical Morphology "
        "and MATLAB Image Processing Tools*\n\n"
        "**DOI:** `10.1109/ICETC.2010.5530052`\n\n"
        "**Methodological elements adapted:** mathematical morphology, binary "
        "reference/test comparison, XOR-based difference, and region localization.\n\n"
        "**Dataset adaptation:** the PCB-Defect dataset is RGB and does not "
        "provide a clean reference for each board, so the expected reference "
        "is generated from the selected image before the binary comparison."
    )

st.write(
    """
    This system detects possible PCB defects in a single colour PCB
    photograph using Digital Image Processing. Because no defect-free
    reference image exists for each board, an expected clean PCB is
    generated from the image itself and compared with the actual PCB.
    """
)

# ============================================================
# RESEARCH PAPER HIGHLIGHT ON MAIN PAGE
# ============================================================

st.divider()
st.header("📄 Research Paper Used in This Project")

research_col1, research_col2 = st.columns([1.15, 2.85])

with research_col1:
    st.metric("Reference", "IEEE ICETC 2010")
    st.caption("Primary methodology reference")

with research_col2:
    st.markdown(
        "**S. H. Indera Putera & Zuwairie Ibrahim**  \n"
        "*Printed Circuit Board Defect Detection Using Mathematical Morphology "
        "and MATLAB Image Processing Tools*"
    )
    st.markdown("**DOI:** `10.1109/ICETC.2010.5530052`")

st.info(
    "**IEEE methodology adapted:** Mathematical morphology + binary "
    "reference/test comparison + XOR-based defect extraction + region "
    "localization.\n\n"
    "**Our RGB adaptation:** PCB-Defect does not provide a defect-free "
    "reference for every board, so an expected clean reference is generated "
    "from the selected RGB PCB before the binary comparison."
)

flow_col1, flow_col2, flow_col3, flow_col4, flow_col5 = st.columns(5)
flow_col1.markdown("**1. Reference\nGeneration**")
flow_col2.markdown("**2. Binary\nComparison**")
flow_col3.markdown("**3. XOR\nDefect Mask**")
flow_col4.markdown("**4. Morphology**")
flow_col5.markdown("**5. Localization**")

st.caption(
    "This section is the explicit research-paper connection used in the "
    "implementation and presentation. Numeric parameter values are "
    "implementation choices unless the paper explicitly specifies them."
)


# ============================================================
# DATASET LOCATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PCB_DATA_DIR = os.path.join(
    BASE_DIR,
    "PCBData"
)


# ============================================================
# FIND ALL IMAGES
# ============================================================

def find_images(folder):

    images = []

    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp",
                    "*.JPG", "*.JPEG", "*.PNG"):

        images += glob.glob(
            os.path.join(folder, "**", pattern),
            recursive=True
        )

    return sorted(set(images))


test_images = find_images(PCB_DATA_DIR)


if len(test_images) == 0:

    st.error(
        "No PCB images were found. "
        "Please check that the PCBData folder is present."
    )

    st.stop()


# ============================================================
# GROUND TRUTH (COCO JSON) - OPTIONAL
# ============================================================

@st.cache_data(show_spinner=False)
def load_coco_index(folder):

    index = {}

    for path in glob.glob(
        os.path.join(folder, "**", "*.json"),
        recursive=True
    ):

        try:

            with open(path, "r", encoding="utf-8") as f:

                data = json.load(f)

        except Exception:

            continue

        if (
            not isinstance(data, dict)
            or "images" not in data
            or "annotations" not in data
        ):

            continue

        categories = {
            c["id"]: c["name"]
            for c in data.get("categories", [])
        }

        id_to_name = {
            im["id"]: os.path.basename(
                str(im["file_name"]).replace("\\", "/")
            ).lower()
            for im in data["images"]
        }

        for ann in data["annotations"]:

            name = id_to_name.get(ann["image_id"])

            if name is None:

                continue

            index.setdefault(name, []).append(
                (
                    ann["bbox"],
                    categories.get(ann["category_id"], "defect")
                )
            )

    return index


coco_index = load_coco_index(PCB_DATA_DIR)


# ============================================================
# SIDEBAR
# ============================================================

# Prominent research-paper section so it is always easy to identify
# the methodology used in the project during demonstration/viva.
st.sidebar.markdown(
    """
    <div style="padding:14px; border-radius:12px; background:#111827;
         border:1px solid #4b5563; margin-bottom:12px;">
        <div style="font-size:20px; font-weight:700; margin-bottom:8px;">
            📄 IEEE RESEARCH PAPER
        </div>
        <div style="font-size:14px; line-height:1.45;">
            <b>IEEE ICETC 2010</b><br>
            S. H. Indera Putera &amp; Zuwairie Ibrahim<br><br>
            <i>Printed Circuit Board Defect Detection Using Mathematical
            Morphology and MATLAB Image Processing Tools</i><br><br>
            <b>DOI:</b> 10.1109/ICETC.2010.5530052
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.success(
    "IEEE methodology used in this implementation:\n\n"
    "1. Mathematical morphology\n"
    "2. Binary reference/test comparison\n"
    "3. XOR defect mask\n"
    "4. Connected-component localization"
)

st.sidebar.header("🔬 Research Method")
comparison_mode = st.sidebar.selectbox(
    "Select comparison method",
    [
        "IEEE-inspired Binary XOR (Recommended)",
        "Absolute Difference (Baseline)"
    ],
    help=(
        "IEEE-inspired mode uses binary reference/test comparison "
        "with XOR before morphological cleanup. Baseline uses "
        "absolute grayscale difference."
    )
)

st.sidebar.header("⚙️ PCB Selection")

image_names = [
    os.path.relpath(image, PCB_DATA_DIR)
    for image in test_images
]


selected_name = st.sidebar.selectbox(
    "Select PCB Image",
    image_names,
    index=0
)


selected_index = image_names.index(
    selected_name
)


selected_test_path = test_images[
    selected_index
]


# ============================================================
# TUNING PARAMETERS
# ============================================================

st.sidebar.header("🎛️ Parameters")

comparison_kernel_size = st.sidebar.selectbox(
    "Morphology kernel size",
    [3, 5, 7],
    index=0
)

merge_radius = st.sidebar.slider(
    "Defect-region merge radius (px)",
    0, 5, 2
)

max_dim = st.sidebar.select_slider(
    "Processing size (longest side, px)",
    options=[800, 1000, 1200, 1600, 2000, 2400],
    value=1600
)

feature_mode = st.sidebar.selectbox(
    "Colour to gray method",
    ["Best colour channel (PCA on Lab)", "Plain grayscale"]
)

polarity = st.sidebar.selectbox(
    "Copper brightness",
    ["auto", "bright", "dark"]
)

width_factor = st.sidebar.slider(
    "Structuring element size (x trace half-width)",
    0.3, 1.5, 0.7, 0.05
)

min_diff_floor = st.sidebar.slider(
    "Minimum threshold floor (0-255)",
    0, 80, 25
)

min_area = st.sidebar.slider(
    "Minimum defect area (px)",
    5, 200, 20
)

border_margin = st.sidebar.slider(
    "Ignore image border (px)",
    0, 60, 15
)


# ============================================================
# LOAD IMAGE
# ============================================================

original_image = cv2.imread(
    selected_test_path,
    cv2.IMREAD_COLOR
)


if original_image is None:

    st.error(
        "Unable to read the selected image."
    )

    st.stop()


orig_h, orig_w = original_image.shape[:2]

scale = min(
    1.0,
    max_dim / max(orig_h, orig_w)
)


if scale < 1.0:

    test_image = cv2.resize(
        original_image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

else:

    test_image = original_image.copy()


# ============================================================
# DISPLAY SELECTED IMAGE
# ============================================================

st.sidebar.write(
    "**Selected:**"
)

st.sidebar.info(
    selected_name
)

st.sidebar.caption(
    f"Original {orig_w}x{orig_h} → processed "
    f"{test_image.shape[1]}x{test_image.shape[0]}"
)


run_button = st.sidebar.button(
    "🔍 RUN INSPECTION",
    type="primary",
    width="stretch"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def stretch01(img):

    lo, hi = np.percentile(img, (0.5, 99.5))

    if hi <= lo:

        hi = lo + 1

    return np.clip((img - lo) / (hi - lo), 0, 1)


def ellipse(radius):

    size = 2 * int(radius) + 1

    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size)
    )


def box_overlap(a, b):

    # intersection area / area of the smaller box, boxes are (x, y, w, h)

    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))

    return (ix * iy) / max(1e-9, min(a[2] * a[3], b[2] * b[3]))


# ============================================================
# IMAGE PROCESSING FUNCTION
# ============================================================

def process_pcb(test, use_pca, polarity, width_factor,
                min_diff_floor, min_area, border_margin,
                comparison_mode, comparison_kernel_size, merge_radius):

    height, width = test.shape[:2]

    # --------------------------------------------------------
    # 1. GRAYSCALE CONVERSION
    #    plain grayscale + best colour channel for copper
    # --------------------------------------------------------

    test_gray = cv2.cvtColor(
        test,
        cv2.COLOR_BGR2GRAY
    )

    lab = cv2.cvtColor(
        test.astype(np.float32) / 255.0,
        cv2.COLOR_BGR2Lab
    )

    L = lab[:, :, 0]

    # remove uneven lighting from the luminance channel
    bg_sigma = max(15, 0.03 * max(height, width))

    L_flat = (
        L
        - cv2.GaussianBlur(L, (0, 0), bg_sigma)
        + L.mean()
    )

    if use_pca:

        # first principal axis of (L, a, b): the colour direction
        # that best separates copper from the board

        samples = np.stack(
            [L_flat.ravel(), lab[:, :, 1].ravel(), lab[:, :, 2].ravel()],
            axis=1
        )

        step = max(1, samples.shape[0] // 50000)

        sub = samples[::step]

        mean = sub.mean(axis=0)

        eigvals, eigvecs = np.linalg.eigh(
            np.cov((sub - mean).T)
        )

        axis = eigvecs[:, np.argmax(eigvals)]

        projection = ((samples - mean) @ axis).reshape(height, width)

        feature = (stretch01(projection) * 255).astype(np.uint8)

    else:

        feature = test_gray.copy()


    # --------------------------------------------------------
    # 2. GAUSSIAN FILTERING
    # --------------------------------------------------------

    test_gray_smooth = cv2.GaussianBlur(
        test_gray,
        (5, 5),
        0
    )

    feature_smooth = cv2.GaussianBlur(
        feature,
        (5, 5),
        0
    )


    # make copper the bright class (auto = copper is the minority class)

    otsu_level, feature_binary = cv2.threshold(
        feature_smooth,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if polarity == "dark":

        invert = True

    elif polarity == "bright":

        invert = False

    else:

        invert = (feature_binary > 0).mean() > 0.5

    if invert:

        feature = 255 - feature

        feature_smooth = 255 - feature_smooth

        _, feature_binary = cv2.threshold(
            feature_smooth,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    copper_mask = feature_binary > 0


    # --------------------------------------------------------
    # 3. GENERATE THE "REFERENCE" (expected defect-free PCB)
    # --------------------------------------------------------

    # estimate trace width from the copper mask

    prelim = copper_mask.astype(np.uint8)

    n_lab, lab_img, st_, _ = cv2.connectedComponentsWithStats(
        prelim,
        connectivity=8
    )

    keep_labels = np.where(st_[:, cv2.CC_STAT_AREA] >= 50)[0]

    keep_labels = keep_labels[keep_labels != 0]

    prelim = np.isin(lab_img, keep_labels).astype(np.uint8)

    dist = cv2.distanceTransform(
        prelim,
        cv2.DIST_L2,
        5
    )

    ridge = (
        (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
        & (dist >= 1)
    )

    half_width = float(np.median(dist[ridge])) if ridge.any() else 3.0

    radius = int(np.clip(round(width_factor * half_width), 1, 20))

    se_ref = ellipse(radius)

    # opening removes extra copper (spur, spurious copper, short bridge)
    # closing fills missing copper (mouse bite, open circuit, missing pad)

    reference_smooth = cv2.morphologyEx(
        cv2.morphologyEx(feature_smooth, cv2.MORPH_CLOSE, se_ref),
        cv2.MORPH_OPEN,
        se_ref
    )

    test_smooth = feature_smooth


    # --------------------------------------------------------
    # 4. IEEE-INSPIRED BINARY COMPARISON / XOR
    # --------------------------------------------------------

    # The selected IEEE methodology uses classical morphology and
    # binary comparison of corresponding PCB regions. Because the
    # RGB PCB-Defect dataset has no separate clean template, the
    # reference above is generated from the same input image.
    # We then binarize both images and compare them with XOR.

    reference_otsu, reference_binary = cv2.threshold(
        reference_smooth,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    test_otsu, test_binary = cv2.threshold(
        test_smooth,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    xor_difference = cv2.bitwise_xor(
        reference_binary,
        test_binary
    )

    # Baseline continuous-valued difference is kept for comparison.
    absolute_difference = cv2.absdiff(
        reference_smooth,
        test_smooth
    )

    if comparison_mode.startswith("IEEE"):
        difference = xor_difference.copy()
        comparison_method = "IEEE-inspired binary XOR"
        otsu_value = float((reference_otsu + test_otsu) / 2.0)
    else:
        difference = absolute_difference.copy()
        comparison_method = "Absolute difference baseline"
        otsu_value = float(0)

    # Signed intensity difference is still retained for the rule-based
    # defect-type classification (extra copper vs missing copper).
    signed_diff = (
        test_smooth.astype(np.int16)
        - reference_smooth.astype(np.int16)
    )


    # --------------------------------------------------------
    # 5. HISTOGRAM
    # --------------------------------------------------------

    histogram = cv2.calcHist(
        [difference],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.flatten()


    # --------------------------------------------------------
    # 6. THRESHOLDING / DEFECT MASK
    # --------------------------------------------------------

    if comparison_mode.startswith("IEEE"):
        # XOR is already binary: 0 = same, 255 = different.
        threshold_value = 0.0
        binary = xor_difference.copy()
    else:
        baseline_otsu, _ = cv2.threshold(
            difference,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_value = float(baseline_otsu)
        threshold_value = max(float(baseline_otsu), float(min_diff_floor))
        binary = (
            (difference > threshold_value).astype(np.uint8) * 255
        )

    if border_margin > 0:

        m = int(border_margin)

        binary[:m, :] = 0
        binary[-m:, :] = 0
        binary[:, :m] = 0
        binary[:, -m:] = 0


    # --------------------------------------------------------
    # 7. MORPHOLOGICAL OPERATIONS
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (comparison_kernel_size, comparison_kernel_size)
    )


    # Erosion
    erosion = cv2.erode(
        binary,
        kernel,
        iterations=1
    )


    # Dilation
    dilation = cv2.dilate(
        binary,
        kernel,
        iterations=1
    )


    # Opening
    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )


    # Closing
    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )


    # --------------------------------------------------------
    # 8. CONNECTED COMPONENT ANALYSIS
    # --------------------------------------------------------

    # merge fragments of one defect before labelling

    if merge_radius > 0:
        merged = cv2.dilate(
            closing,
            ellipse(merge_radius)
        )
    else:
        merged = closing.copy()

    number_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            merged,
            connectivity=8
        )
    )


    ring_radius = max(2, radius) + 1

    pad = 2 * radius + 4

    copper_u8 = copper_mask.astype(np.uint8)

    defect_regions = []


    for label in range(
        1,
        number_labels
    ):

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])


        # area of the real (un-merged) defect pixels

        area = int(
            np.count_nonzero(
                (labels[y:y + h, x:x + w] == label)
                & (closing[y:y + h, x:x + w] > 0)
            )
        )


        # Remove very small noise
        if area < min_area:

            continue


        # ----------------------------------------------------
        # 9. CLASSIFY THE REGION (rule based)
        # ----------------------------------------------------

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)

        reg = labels[y1:y2, x1:x2] == label

        copper_crop = copper_u8[y1:y2, x1:x2]

        signed_crop = signed_diff[y1:y2, x1:x2]

        core = reg & (closing[y1:y2, x1:x2] > 0)

        if not core.any():

            core = reg

        sign = float(signed_crop[core].mean())

        ring = (
            cv2.dilate(reg.astype(np.uint8), ellipse(ring_radius)) > 0
        ) & ~reg

        if sign > 0:

            # extra copper: remove it and count copper pieces around it
            _, lab2 = cv2.connectedComponents(
                ((copper_crop > 0) & ~reg).astype(np.uint8),
                connectivity=8
            )

        else:

            # missing copper: count copper pieces on the two sides
            _, lab2 = cv2.connectedComponents(
                copper_crop,
                connectivity=8
            )

        touching = lab2[ring & (lab2 > 0)]

        if touching.size == 0:

            n_touch = 0

        else:

            n_touch = int(
                np.count_nonzero(np.bincount(touching) >= 3)
            )

        if sign > 0:

            if n_touch == 0:

                defect_type = "Spurious copper"

            elif n_touch == 1:

                defect_type = "Spur"

            else:

                defect_type = "Short circuit"

        else:

            ring_copper = (
                np.count_nonzero(ring & (copper_crop > 0))
                / max(1, np.count_nonzero(ring))
            )

            if n_touch >= 2:

                defect_type = "Open circuit"

            elif ring_copper > 0.85:

                defect_type = "Missing pad"

            else:

                defect_type = "Mouse bite"


        defect_regions.append(
            {
                "Defect": len(defect_regions) + 1,
                "Type": defect_type,
                "X": x,
                "Y": y,
                "Width": w,
                "Height": h,
                "Area": area,
                "X_full": int(round(x / scale_factor_global)),
                "Y_full": int(round(y / scale_factor_global)),
                "W_full": int(round(w / scale_factor_global)),
                "H_full": int(round(h / scale_factor_global))
            }
        )


    # --------------------------------------------------------
    # 10. DRAW BOUNDING BOXES
    # --------------------------------------------------------

    result_image = test.copy()

    font_scale = max(0.5, max(height, width) / 1600 * 0.6)

    thickness = max(2, int(round(max(height, width) / 700)))


    for defect in defect_regions:

        x = defect["X"]
        y = defect["Y"]

        width_box = defect["Width"]
        height_box = defect["Height"]


        cv2.rectangle(
            result_image,
            (x, y),
            (x + width_box, y + height_box),
            (0, 0, 255),
            thickness
        )


        cv2.putText(
            result_image,
            defect["Type"],
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )


    return {

        "test_gray":
            test_gray,

        "feature":
            feature,

        "test_gray_smooth":
            test_gray_smooth,

        "feature_smooth":
            feature_smooth,

        "reference_smooth":
            reference_smooth,

        "test_smooth":
            test_smooth,

        "reference_binary":
            reference_binary,

        "test_binary":
            test_binary,

        "xor_difference":
            xor_difference,

        "comparison_method":
            comparison_method,

        "reference_otsu":
            float(reference_otsu),

        "test_otsu":
            float(test_otsu),

        "copper_mask":
            copper_u8 * 255,

        "half_width":
            half_width,

        "radius":
            radius,

        "difference":
            difference,

        "histogram":
            histogram,

        "otsu_value":
            float(otsu_value),

        "threshold":
            threshold_value,

        "binary":
            binary,

        "erosion":
            erosion,

        "dilation":
            dilation,

        "opening":
            opening,

        "closing":
            closing,

        "defect_regions":
            defect_regions,

        "result_image":
            result_image
    }


# the full-resolution columns need the resize factor
scale_factor_global = scale


# ============================================================
# INITIAL SCREEN
# ============================================================

if not run_button:

    st.subheader(
        "📷 Selected PCB"
    )


    st.image(
        cv2.cvtColor(
            test_image,
            cv2.COLOR_BGR2RGB
        ),
        caption=selected_name,
        width="stretch"
    )


    st.info(
        "Select any PCB from the sidebar and click "
        "'RUN INSPECTION'."
    )


# ============================================================
# RUN INSPECTION
# ============================================================

if run_button:

    with st.spinner(
        "Processing PCB using Digital Image Processing..."
    ):

        result = process_pcb(
            test_image,
            feature_mode.startswith("Best"),
            polarity,
            width_factor,
            min_diff_floor,
            min_area,
            border_margin,
            comparison_mode,
            comparison_kernel_size,
            merge_radius
        )


    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    st.success(
        "PCB inspection completed."
    )


    st.header(
        "📊 Inspection Summary"
    )


    number_defects = len(
        result["defect_regions"]
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Selected PCB",
            os.path.basename(selected_name)
        )


    with col2:

        st.metric(
            "Detected Regions",
            number_defects
        )


    with col3:

        st.metric(
            "Threshold Used",
            f"{result['threshold']:.2f}",
            help=f"Otsu value was {result['otsu_value']:.2f}; "
                 f"a floor of {min_diff_floor} is applied."
        )


    # ========================================================
    # INPUT IMAGE
    # ========================================================

    st.divider()

    st.header(
        "1️⃣ Input Image"
    )


    st.image(
        cv2.cvtColor(
            test_image,
            cv2.COLOR_BGR2RGB
        ),
        caption="Selected Test PCB (colour)",
        width="stretch"
    )


    # ========================================================
    # GRAYSCALE
    # ========================================================

    st.divider()

    st.header(
        "2️⃣ Grayscale Conversion"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            result["test_gray"],
            caption="Plain Grayscale",
            width="stretch"
        )


    with col2:

        st.image(
            result["feature"],
            caption="Best colour channel (copper bright)",
            width="stretch"
        )


    # ========================================================
    # FILTERING
    # ========================================================

    st.divider()

    st.header(
        "3️⃣ Gaussian Filtering"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            result["test_gray_smooth"],
            caption="Filtered Grayscale",
            width="stretch"
        )


    with col2:

        st.image(
            result["feature_smooth"],
            caption="Filtered Colour Channel",
            width="stretch"
        )


    # ========================================================
    # REFERENCE GENERATION + SUBTRACTION
    # ========================================================

    st.divider()

    st.header(
        "4️⃣ IEEE-Inspired Reference Generation and Binary Comparison"
    )


    st.write(
        f"""
        **Research methodology reference:** S. H. Indera Putera and
        Zuwairie Ibrahim, IEEE ICETC 2010.

        This implementation adapts the paper's classical
        morphology + binary comparison idea to the RGB PCB-Defect
        dataset. A separate defect-free template is not available,
        so the expected clean PCB is generated from the selected image.

        Estimated trace half-width: **{result['half_width']:.1f} px**
        → reference structuring-element radius: **{result['radius']} px**.

        Current comparison: **{result['comparison_method']}**
        """
    )


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            result["reference_smooth"],
            caption="Generated Reference (opening + closing)",
            width="stretch"
        )

    with col2:

        st.image(
            result["test_smooth"],
            caption="Test Feature (Gaussian filtered)",
            width="stretch"
        )

    col3, col4 = st.columns(2)

    with col3:

        st.image(
            result["reference_binary"],
            caption=f"Binary Reference (Otsu = {result['reference_otsu']:.1f})",
            width="stretch"
        )

    with col4:

        st.image(
            result["test_binary"],
            caption=f"Binary Test (Otsu = {result['test_otsu']:.1f})",
            width="stretch"
        )

    st.image(
        result["xor_difference"],
        caption="IEEE-inspired XOR Difference",
        width="stretch"
    )


    # ========================================================
    # HISTOGRAM
    # ========================================================

    st.divider()

    st.header(
        "5️⃣ Histogram Analysis"
    )


    histogram_df = pd.DataFrame(
        {
            "Intensity":
                np.arange(256),

            "Pixel Count":
                result["histogram"]
        }
    )


    st.bar_chart(
        histogram_df,
        x="Intensity",
        y="Pixel Count"
    )


    # ========================================================
    # THRESHOLDING
    # ========================================================

    st.divider()

    st.header(
        "6️⃣ Otsu Thresholding"
    )


    if comparison_mode.startswith("IEEE"):
        st.write(
            f"""
            Otsu is applied separately to the generated reference
            and the test feature before the binary XOR comparison.

            Reference Otsu: **{result['reference_otsu']:.2f}**  
            Test Otsu: **{result['test_otsu']:.2f}**
            """
        )
    else:
        st.write(
            f"""
            Automatically calculated Otsu value:
            **{result['otsu_value']:.2f}**
            — threshold used (with floor): **{result['threshold']:.2f}**
            """
        )


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            result["copper_mask"],
            caption="Copper mask (check that copper is white)",
            width="stretch"
        )


    with col2:

        st.image(
            result["binary"],
            caption="Binary Image after Otsu Thresholding",
            width="stretch"
        )


    # ========================================================
    # MORPHOLOGY
    # ========================================================

    st.divider()

    st.header(
        "7️⃣ Morphological Processing"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            result["erosion"],
            caption="Erosion",
            width="stretch"
        )


    with col2:

        st.image(
            result["dilation"],
            caption="Dilation",
            width="stretch"
        )


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            result["opening"],
            caption="Opening",
            width="stretch"
        )


    with col2:

        st.image(
            result["closing"],
            caption="Closing",
            width="stretch"
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    st.divider()

    st.header(
        "8️⃣ Final Defect Localization"
    )


    st.image(
        cv2.cvtColor(
            result["result_image"],
            cv2.COLOR_BGR2RGB
        ),
        caption="Detected Defect Regions",
        width="stretch"
    )


    # ========================================================
    # DEFECT TABLE
    # ========================================================

    st.subheader(
        "Detected Region Details"
    )


    if number_defects > 0:

        defect_df = pd.DataFrame(
            result["defect_regions"]
        )


        st.dataframe(
            defect_df,
            width="stretch",
            hide_index=True
        )


        st.warning(
            f"{number_defects} possible defect region(s) "
            "were detected."
        )


    else:

        st.success(
            "No significant defect regions detected."
        )


    # ========================================================
    # GROUND TRUTH COMPARISON
    # ========================================================

    st.divider()

    st.header(
        "9️⃣ Ground Truth Comparison"
    )


    gt_entries = coco_index.get(
        os.path.basename(selected_name).lower(),
        []
    )


    if len(gt_entries) == 0:

        st.info(
            "No COCO annotation was found for this image "
            "(put the dataset .json inside PCBData to enable this)."
        )

    else:

        gt_boxes = [
            (
                b[0] * scale + 0,
                b[1] * scale + 0,
                b[2] * scale,
                b[3] * scale
            )
            for b, _ in gt_entries
        ]

        gt_labels = [name for _, name in gt_entries]

        det_boxes = [
            (d["X"], d["Y"], d["Width"], d["Height"])
            for d in result["defect_regions"]
        ]

        gt_hit = [False] * len(gt_boxes)

        det_hit = [False] * len(det_boxes)

        for i, g in enumerate(gt_boxes):

            for j, d in enumerate(det_boxes):

                if box_overlap(g, d) >= 0.3:

                    gt_hit[i] = True

                    det_hit[j] = True

        recall = sum(gt_hit) / len(gt_boxes)

        precision = (
            sum(det_hit) / len(det_boxes) if det_boxes else 0.0
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric("Annotated defects", len(gt_boxes))

        with col2:

            st.metric("Recall", f"{recall * 100:.1f}%")

        with col3:

            st.metric("Precision", f"{precision * 100:.1f}%")

        gt_image = test_image.copy()

        gt_thickness = max(2, int(round(max(gt_image.shape[:2]) / 700)))

        for (gx, gy, gw, gh) in gt_boxes:

            cv2.rectangle(
                gt_image,
                (int(gx), int(gy)),
                (int(gx + gw), int(gy + gh)),
                (0, 200, 0),
                gt_thickness
            )

        for (dx, dy, dw, dh) in det_boxes:

            cv2.rectangle(
                gt_image,
                (dx, dy),
                (dx + dw, dy + dh),
                (0, 0, 255),
                1
            )

        st.image(
            cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB),
            caption="Green = annotated defects, red = detected regions",
            width="stretch"
        )

        gt_df = pd.DataFrame(
            {
                "Annotated class": gt_labels,
                "Found": gt_hit
            }
        )

        summary_df = (
            gt_df.groupby("Annotated class")["Found"]
            .agg(Found="sum", Total="count")
            .reset_index()
        )

        st.dataframe(
            summary_df,
            width="stretch",
            hide_index=True
        )


# ============================================================
# RESEARCH PAPER CONNECTION
# ============================================================

st.divider()
st.header("📄 Research Paper Connection")

st.info(
    """
    **Selected methodology paper:**
    S. H. Indera Putera and Zuwairie Ibrahim,
    *Printed Circuit Board Defect Detection Using Mathematical
    Morphology and MATLAB Image Processing Tools*, IEEE ICETC 2010.
    DOI: 10.1109/ICETC.2010.5530052

    **How this project uses the paper:** the implementation follows
    the classical idea of mathematical morphology and binary PCB
    comparison. For the RGB PCB-Defect dataset, a clean reference is
    first estimated from the selected image because a separate
    defect-free template is not supplied. The generated reference and
    test image are binarized, then XOR is used to highlight structural
    differences before morphological cleanup and connected-component
    analysis.

    **Adaptation note:** this is an implementation inspired by the
    paper, not a claim of exact reproduction of the original dataset
    or experimental setup.
    """
)

# ============================================================
# THEORY SECTION
# ============================================================

st.divider()

st.header(
    "📚 Theory — How Does the System Work?"
)


st.write(
    """
    The system works on a single colour PCB photograph. Since no
    defect-free reference image is available, an expected clean PCB
    is generated from the image itself. Digital Image Processing
    operations then highlight and localize differences that may
    correspond to PCB defects.
    """
)


# ============================================================
# THEORY 1
# ============================================================

with st.expander(
    "1️⃣ Test Image and Generated Reference"
):

    st.write(
        """
        The test image is the PCB photograph selected by the user.
        The reference image is not available, so the expected
        defect-free PCB is generated from the test image itself
        (see step 4).

        We compare the two images because a defect is a place where
        the actual PCB differs from the clean, regular copper pattern.
        """
    )


# ============================================================
# THEORY 2
# ============================================================

with st.expander(
    "2️⃣ Grayscale Conversion"
):

    st.write(
        """
        The colour image is converted into a grayscale image.

        A grayscale image contains intensity values instead
        of three separate colour channels.

        Because the images are colour photographs, we also project
        the Lab colour values onto their main axis of variation
        (principal component). This gives one channel in which
        copper and board differ the most, after removing uneven
        lighting.
        """
    )


# ============================================================
# THEORY 3
# ============================================================

with st.expander(
    "3️⃣ Gaussian Filtering"
):

    st.write(
        """
        Gaussian filtering smooths the image and reduces
        small unwanted variations such as camera noise and
        surface texture.

        This prevents very small noise from being treated as
        a PCB defect.
        """
    )


# ============================================================
# THEORY 4
# ============================================================

with st.expander(
    "4️⃣ Reference Generation and Image Subtraction"
):

    st.write(
        """
        The trace width is estimated from the copper mask using the
        distance transform. A structuring element slightly smaller
        than the trace width is then used for:

        Opening: removes copper features thinner than the element
        (spurs, spurious copper, thin short-circuit bridges).

        Closing: fills dark gaps smaller than the element
        (mouse bites, open circuits, missing pads).

        The result of opening followed by closing is the expected
        clean PCB.

        Difference = |Reference − Test|

        Where the PCB is regular the difference is small. Where a
        defect exists the local difference becomes larger.
        """
    )


# ============================================================
# THEORY 5
# ============================================================

with st.expander(
    "5️⃣ Histogram"
):

    st.write(
        """
        A histogram shows the distribution of intensity values
        in the difference image.

        It helps us understand how the pixel intensities are
        distributed before thresholding. Most pixels are close to
        zero, and defects form a small tail at higher values.
        """
    )


# ============================================================
# THEORY 6
# ============================================================

with st.expander(
    "6️⃣ Otsu Thresholding"
):

    st.write(
        """
        Otsu thresholding converts the difference image into
        a binary image.

        Pixels above the selected threshold become foreground,
        while pixels below the threshold become background.

        This separates possible defect regions from the
        background automatically. A small minimum threshold is
        applied so that nearly flawless boards do not produce
        false detections.
        """
    )


# ============================================================
# THEORY 7
# ============================================================

with st.expander(
    "7️⃣ Morphological Operations"
):

    st.write(
        """
        Morphological operations improve the binary image.

        Erosion:
        Removes or shrinks foreground regions.

        Dilation:
        Expands foreground regions.

        Opening:
        Helps remove small unwanted regions.

        Closing:
        Helps fill small gaps and connect nearby regions.
        """
    )


# ============================================================
# THEORY 8
# ============================================================

with st.expander(
    "8️⃣ Connected Component Analysis"
):

    st.write(
        """
        Connected component analysis identifies separate
        connected regions in the processed binary image.

        Each sufficiently large region is considered a
        possible defect.

        For each region we calculate its position, width,
        height and area.
        """
    )


# ============================================================
# THEORY 9
# ============================================================

with st.expander(
    "9️⃣ Defect Type and Final Localization"
):

    st.write(
        """
        The sign of the difference tells whether copper is extra
        (test brighter than reference) or missing (test darker).

        Extra copper:
        no copper neighbour = spurious copper,
        one copper neighbour = spur,
        two or more = short circuit.

        Missing copper:
        copper pieces on both sides = open circuit,
        completely surrounded by copper = missing pad,
        otherwise = mouse bite.

        Finally, bounding boxes are drawn around the detected
        regions so the user can see where the possible PCB
        defects are located.
        """
    )


# ============================================================
# COMPLETE FLOW
# ============================================================

st.subheader(
    "🔄 Complete Processing Flow"
)


st.code(
    """
Colour PCB photograph (no reference image)
      ↓
Grayscale + Best Colour Channel
      ↓
Gaussian Filtering
      ↓
Self-Generated Reference (Opening + Closing)
      ↓
Otsu on Reference + Test
      ↓
Binary XOR Comparison
      ↓
Histogram Analysis
      ↓
Morphological Operations
      ↓
Connected Component Analysis
      ↓
Defect Localization and Type
""",
    language="text"
)


# ============================================================
# VIVA EXPLANATION
# ============================================================

st.subheader(
    "🎓 Simple Viva Explanation"
)


st.info(
    """
    The dataset contains single colour PCB photographs without a
    defect-free reference, so we generate the expected clean PCB
    from the image itself. First the image is converted to
    grayscale and a colour channel that separates copper from the
    board, then smoothed with Gaussian filtering. We estimate the
    trace width and apply opening and closing with a structuring
    element of that size, which removes extra copper and fills
    missing copper. Subtracting this reference from the test image
    makes the defects visible. We analyse the histogram of the
    difference image and use Otsu thresholding to create a binary
    image. Morphological operations remove small unwanted regions.
    Finally, connected component analysis identifies separate
    regions, bounding boxes are drawn around the possible PCB
    defects, and each region is labelled by the type of defect.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Automated PCB Defect Detection and Localization "
    "Using Digital Image Processing"
)
