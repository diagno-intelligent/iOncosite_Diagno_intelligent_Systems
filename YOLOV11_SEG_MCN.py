import shutil
import os
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# Load best model
# Class mapping
class_names = {0: "Mass", 1: "COPD", 2: "Normal"}
# Fixed colors (BGR for OpenCV)
class_colors = {
    "Mass": (0, 0, 255),       # Red
    "COPD": (0, 165, 255),     # Orange
    "Normal": (0, 255, 0)      # Green
}

# Transparency factor
alpha = 0.4
results=[]
# Load best model
model = YOLO("./yolov11_seg_MCN_best.pt")
# Input/output
#image_path = "./images/File_004991_7204_nn.png" #"E:/project_new/new_5_MCNTS_multiclass_data/Mass_COPD_Normal_dataset/images/00000830_000.png"
#  "E:\project_new\downloads\chest_xray_14_Multiclass_data\images_002\images\00001362_015.png"
#"E:\"E:\project_new\downloads\chest_xray_14_Multiclass_data\images_005\images\00010815_006.png"
# "E:\project_new\downloads\chest_xray_14_Multiclass_data\images_005\images\00010125_004.png"
# "E:\project_new\downloads\chest_xray_14_Multiclass_data\"E:\project_new\downloads\chest_xray_14_Multiclass_data\images_010\images\00023075_033.png"
# \images\00023075_033.png"
# "E:\project_new\downloads\chest_xray_14_Multiclass_data\images_011\images\00026196_001.png"
#"E:\project_new\downloads\chest_xray_14_Multiclass_data\images_011\images\00027833_022.png"
#"E:\project_new\downloads\chest_xray_14_Multiclass_data\images_012\images\00028265_007.png" #


image_path ="E:/project_new/downloads/chest_xray_14_Multiclass_data/images_012/images/00028265_007.png" # dataset chest14
#image_path ="E:/project_new/LC_Normal/LC_Normal_resized_data/combined_data_LC_normal/File_008795_12749.png" # dataset indian lungcancer
#image_path ="E:/project_new/new_5_MCNTS_multiclass_data/all_resized_dicom_png_1024_NN_label_TSCN/File_000288_410.png"
#####################33
#img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#hist_eq = cv2.equalizeHist(img)
#cv2.imwrite("./delect/V11_input.png",  cv2.cvtColor(hist_eq, cv2.COLOR_RGB2BGR))

#image_path ="./delect/V11_input.png"
############################
output_path = "./images_YOLOV11/V11_input.png"

# Make sure the output folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Copy image
shutil.copy(image_path, output_path)

output_path = "./output_YOLOV11/V11_SEG_PRED.png"

# Run inference
results = model(image_path, conf=0.5, iou=0.5, imgsz=1024)
result = results[0]

# Read original image
img = cv2.imread(image_path)

# Run inference
if result.masks is not None:   # ✅ check before using
    results = model(image_path, conf=0.3, iou=0.5, imgsz=1024)
    result = results[0]
if result.masks is not None:   # ✅ check before using
    results = model(image_path, conf=0.05, iou=0.5, imgsz=1024)
    result = results[0]


# ################3 real confidence score
# # -------- Step 1: Apply segmentation masks (without darkening background) --------
# if result.masks is not None:   # ✅ check before using
#     for mask, cls_id in zip(result.masks.data, result.boxes.cls):
#         cls_id = int(cls_id)
#         cls_name = class_names[cls_id]
#         color = class_colors[cls_name]
#
#         mask = mask.cpu().numpy().astype(np.uint8)
#         mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
#
#         # Extract the region of interest (ROI) where mask==1
#         roi = img[mask == 1]
#
#         # Create same-shape color array
#         color_arr = np.full_like(roi, color, dtype=np.uint8)
#
#         # Blend only masked region
#         blended = cv2.addWeighted(roi, 1 - alpha, color_arr, alpha, 0)
#
#         # Put back blended pixels
#         img[mask == 1] = blended
#
#
#     # -------- Step 2: Draw bounding boxes + labels (with white background) --------
#     for box, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
#         cls_id = int(cls_id)
#         cls_name = class_names[cls_id]
#         color = class_colors[cls_name]
#
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#
#         # Build label with confidence
#         conf=conf*100
#         label = f"{cls_name} {conf:.0f}%"
#
#         # Get text size
#         (font_w, font_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#
#         # White background rectangle
#         cv2.rectangle(img,
#                       (x1, y1 - font_h - baseline),
#                       (x1 + font_w, y1),
#                       (255, 255, 255), -1)
#
#         # Text on top of white background
#         cv2.putText(img, label,
#                     (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9,
#                     color, 2)
#
# # -------- Step 3: Save and show result --------
# cv2.imwrite(output_path, img)
# print(f"Saved to {output_path}")
#
# # Show with matplotlib (correct colors)
# plt.figure(figsize=(7, 7))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title("Detected regions")
# plt.axis("off")
# plt.tight_layout()
# plt.show()

###### feature extraction
import os
import torch
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# === Load trained YOLOv10 model ===
model.eval()

# === Hook to extract backbone features ===
features_dict = {}

def hook_fn(module, input, output):
    pooled = torch.mean(output[0], dim=(1, 2))  # Global Average Pooling
    features_dict['feat'] = pooled.detach().cpu().numpy()

# You might need to adjust this index based on your model structure
hook = model.model.model[10].register_forward_hook(hook_fn)
def extract_features_from_txt(image_folder, save_csv_path):
    data = []
    all_images = sorted(os.listdir(image_folder))

    for filename in tqdm(all_images, desc=f"Extracting from {os.path.basename(image_folder)}"):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue

        img_path = os.path.join(image_folder, filename)
        #label_path = os.path.join(label_folder, filename.replace('.png', '.txt').replace('.jpg', '.txt'))
        main_class=10

        try:
            _ = model(img_path)
            feat = features_dict.get('feat')
            if feat is not None:
                row = [filename, main_class] + feat.tolist()
                data.append(row)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # Save CSV
    if data:
        columns = ['filename', 'label'] + [f'feat_{i}' for i in range(len(data[0]) - 2)]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(save_csv_path, index=False)
        print(f"✅ Saved features to {save_csv_path}")
    else:
        print("⚠️ No data was extracted.")
extract_features_from_txt(
    image_folder='./images_YOLOV11',
    save_csv_path='./yolov11_MCN_whole_features_test.csv'
)
######### ML results
import ens_modelling_MCN_test_fn

ens_ML_MCN_output,predicted_proba = ens_modelling_MCN_test_fn.ens_ML_MCN()
print("Ens ML results:", ens_ML_MCN_output)
predicted_proba=predicted_proba[0]
conf_ML=predicted_proba[ens_ML_MCN_output]*100
print('conf_ML',conf_ML)

################3 changing label confidence score
# -------- Step 1: Apply segmentation masks (without darkening background) --------
if result.masks is not None:   # ✅ check before using
    for mask, cls_id in zip(result.masks.data, result.boxes.cls):
        cls_id = int(cls_id)
        cls_name = class_names[cls_id]
        color = class_colors[cls_name]

        mask = mask.cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Extract the region of interest (ROI) where mask==1
        roi = img[mask == 1]

        # Create same-shape color array
        color_arr = np.full_like(roi, color, dtype=np.uint8)

        # Blend only masked region
        blended = cv2.addWeighted(roi, 1 - alpha, color_arr, alpha, 0)

        # Put back blended pixels
        img[mask == 1] = blended


    # -------- Step 2: Draw bounding boxes + labels (with white background) --------
    for box, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        cls_id = int(cls_id)
        cls_name = class_names[cls_id]
        color = class_colors[cls_name]

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Build label with confidence
        conf=conf_ML
        label = f"{cls_name} {conf:.0f}%"

        # Get text size
        (font_w, font_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

        # White background rectangle
        cv2.rectangle(img,
                      (x1, y1 - font_h - baseline),
                      (x1 + font_w, y1),
                      (255, 255, 255), -1)

        # Text on top of white background
        if ens_ML_MCN_output==2:
            color1=(0,0,0)
            cv2.putText(img, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        color1, 2)
        else:
            cv2.putText(img, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        color, 2)

# -------- Step 3: Save and show result --------
cv2.imwrite(output_path, img)
print(f"Saved to {output_path}")

# Show with matplotlib (correct colors)
plt.figure(figsize=(7, 7))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detected regions")
plt.axis("off")
plt.tight_layout()
plt.show()

############## segmented region
if ens_ML_MCN_output==0: #
    import os
    import cv2
    from PIL import Image

    # ----------------------------
    # CLASS NAMES
    # ----------------------------
    class_names = {0: "Mass", 1: "COPD", 2:"Normal"}


    # ==================== helpers ====================

    def feret_diameters_from_contour(cnt):
        if cnt.ndim == 3 and cnt.shape[1] == 1:
            cnt = cnt[:, 0, :]
        cnt = cnt.astype(np.float32)

        area = float(cv2.contourArea(cnt))
        if len(cnt) < 3:
            return dict(area=area, major_len=0.0, minor_len=0.0,
                        major_p1=(0, 0), major_p2=(0, 0), major_angle_deg=0.0,
                        minor_p1=(0, 0), minor_p2=(0, 0), minor_angle_deg=0.0)

        hull = cv2.convexHull(cnt)
        if hull.ndim == 3:
            hull = hull[:, 0, :]
        P = hull.astype(np.float32)
        M = len(P)

        # --- Max Feret ---
        if M > 600:
            step = int(M / 600) + 1
            P_major = P[::step]
        else:
            P_major = P

        A = P_major[:, None, :]
        B = P_major[None, :, :]
        diff = A - B
        D2 = (diff ** 2).sum(-1)
        i, j = np.unravel_index(np.argmax(D2), D2.shape)
        p1 = tuple(P_major[i].astype(float))
        p2 = tuple(P_major[j].astype(float))
        major_len = float(np.sqrt(D2[i, j]))
        major_angle_deg = float((np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) + 180) % 180)

        # --- Min Feret (rotating calipers) ---
        rect = cv2.minAreaRect(P)
        (cx, cy), (w, h), angle = rect
        if w < h:
            min_len = w
            min_angle_deg = angle
        else:
            min_len = h
            min_angle_deg = angle + 90
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        dists = [np.linalg.norm(box[(k + 1) % 4] - box[k]) for k in range(4)]
        kmin = int(np.argmin(dists))
        q1 = tuple(box[kmin])
        q2 = tuple(box[(kmin + 1) % 4])

        return dict(
            area=area,
            major_len=major_len, major_p1=p1, major_p2=p2, major_angle_deg=major_angle_deg,
            minor_len=float(min_len), minor_p1=q1, minor_p2=q2, minor_angle_deg=min_angle_deg
        )


    # ==================== main ====================

    def process_segmentation(image_path, results):
        orig_img = np.array(Image.open(image_path).convert("RGB"))
        H, W = orig_img.shape[:2]

        if results[0].masks is None:
            print("No segmentation detected.")
            return

        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        region_rows = []
        overlay = orig_img.copy()
        red = np.array([255, 0, 0], dtype=np.uint8)
        alpha = 0.35

        os.makedirs("./output_poly_feret", exist_ok=True)

        for idx, m in enumerate(masks, start=0):  # keep index aligned with boxes
            m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_bin = (m_resized > 0.5).astype(np.uint8)
            if mask_bin.sum() == 0:
                continue

            m_idx = mask_bin.astype(bool)
            overlay[m_idx] = (alpha * red + (1 - alpha) * overlay[m_idx]).astype(np.uint8)

            cnts, _ = cv2.findContours((mask_bin * 255).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)

            stats = feret_diameters_from_contour(cnt)

            # --- Class + Confidence ---
            cls_id = class_ids[idx]
            conf = confidences[idx]
            label = class_names.get(cls_id, str(cls_id))

            # --- Draw major axis line ---
            p1 = tuple(map(int, stats['major_p1']))
            p2 = tuple(map(int, stats['major_p2']))
            cv2.line(overlay, p1, p2, (0, 255, 0), 2)

            # --- Label ---
            def draw_text_with_background(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale=0.8, text_color=(255, 0, 0),  # Red text
                                          bg_color=(255, 0, 255), thickness=2, padding=4):  # Yellow background

                # Get text size
                (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # Coordinates for background rectangle
                x, y = org
                cv2.rectangle(img, (x, y - h - baseline - padding),
                              (x + w + padding * 2, y + baseline + padding),
                              bg_color, -1)

                # Put text over the background
                cv2.putText(img, text, (x + padding, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

                return img
            x, y, w, h = cv2.boundingRect(cnt)
            # --- Place text near region instead of top-left ---
            txt = [
                f"{label} ({conf_ML:.0f}%)",
                f"Area: {stats['area']:.0f} (pix)",
                f"Length: {stats['major_len']:.0f} (pix)",
                # f"MinFeret: {stats['minor_len']:.0f} (pix)"
            ]

            # Start y above the bounding box (or inside if too close to top)
            y0 = max(y - 10, 20)
            dy = 30

            for i, t in enumerate(txt):
                yy = y0 + i * dy
                if i == 0:  # label + confidence → red text on yellow background
                    draw_text_with_background(
                        overlay, t, (x+w, yy),
                        text_color=(255, 0, 0),  # red
                        bg_color=(255, 255, 0)  # yellow
                    )
                else:  # white plain text for stats
                    cv2.putText(
                        overlay, t, (x+w, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA
                    )

            # --- Save row ---
            region_rows.append({
                "Region": idx + 1,
                "Class": label,
                "Confidence": float(conf),
                "Area_px2": stats['area'],
                "MaxFeret_px": stats['major_len'],
                "MinFeret_px": stats['minor_len'],
                "MaxFeret_angle_deg": stats['major_angle_deg'],
                "MinFeret_angle_deg": stats['minor_angle_deg'],
            })

        plt.figure(figsize=(7, 7))
        plt.imshow(overlay)
        plt.title("Overlay + Polygon Feret Diameters + Class")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        #cv2.imwrite("./output_poly_feret/overlay_with_class.png",
                    #cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./output_YOLOV11/V11_SEG_PRED.png",  cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        df = pd.DataFrame(region_rows)
        df.to_csv("./output_poly_feret/region_stats_with_class.csv", index=False)
        print(df)


    process_segmentation(image_path, results)