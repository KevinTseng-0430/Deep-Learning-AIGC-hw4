import streamlit as st
from pathlib import Path
import os
from app_utils import (
    list_images_in_folder,
    load_pil_image,
    predict_image_stub,
    gather_dataset_stats,
    extract_image_metadata,
    compute_detailed_stats,
    try_load_torch_model,
    prepare_predictions_dataframe,
    calibration_stats,
    is_streamlit_cloud,
    is_local_deployment,
)
import io
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from PIL import Image


def inject_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def image_download_button(img, filename="image.png", button_text="Download image"):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)


@st.cache_data
def load_images_from_folder(folder_path: str):
    """Load images from folder with error handling."""
    try:
        data_dir = Path(folder_path)
        if not data_dir.exists():
            return None, f"❌ Folder not found: {folder_path}"
        
        img_paths = list_images_in_folder(data_dir)
        if not img_paths:
            return None, f"⚠️ No images found in: {folder_path}"
        
        images = []
        errors = []
        for p in img_paths:
            try:
                img = load_pil_image(p)
                images.append((p, img))
            except Exception as e:
                errors.append(f"Failed to load {p.name}: {str(e)}")
        
        return images, None
    except Exception as e:
        return None, f"❌ Error loading folder: {str(e)}"


def main():
    st.set_page_config(page_title="Crested Myna Recognizer", layout="wide")
    inject_css()

    st.markdown("<h1 class='title'>Crested Myna (Acridotheres cristatellus) Recognizer</h1>", unsafe_allow_html=True)
    st.markdown("A polished Streamlit app to upload images or read from the project's `data/` folder. Includes dataset analytics, previews, predictions, and downloads.")

    workspace_root = Path(__file__).parent
    default_data_dir = workspace_root / "data"

    with st.sidebar:
        st.header("Upload images")
        st.markdown("Upload multiple images in the main area to run predictions and see analytics.")
        st.markdown("---")
        st.header("Model")
        model_info = try_load_torch_model(Path(__file__).parent / "models")
        if model_info[0]:
            st.success(f"✅ PyTorch model: {model_info[1]}")
        else:
            st.info("📊 Using heuristic demo predictor")
        
        st.markdown("---")
        st.subheader("❓ Help & Support")
        st.markdown("""
        Use the main Upload area to drag & drop images.
        
        **Note:** This app no longer supports reading a local `data/` folder directly — upload images instead (works both locally and on Streamlit Cloud).
        """)

    # Main layout with tabs: Analytics and Gallery
    tab_analytics, tab_gallery = st.tabs(["Analytics 📊", "Gallery 🖼️"])

    # Upload-only flow: load images from uploaded files
    uploaded_files = st.file_uploader(
        "📤 Upload images (support multiple files)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    images = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                img = load_pil_image(uploaded_file)
                images.append((uploaded_file.name, img))
            except Exception as e:
                st.warning(f"Failed to load {uploaded_file.name}: {str(e)}")

    # Analytics tab (now computed from uploaded images)
    with tab_analytics:
        st.subheader("📊 Batch overview (from uploaded images)")
        st.markdown("Image data => https://github.com/KevinTseng-0430/Deep-Learning-AIGC-hw4/tree/main/data")
        # Build lightweight stats from uploaded images
        stats = {"total_images": 0, "class_counts": {}, "sizes": [], "formats": {}, "aspect_ratios": [], "file_sizes_mb": []}
        if images:
            stats["total_images"] = len(images)
            for name, img in images:
                md = extract_image_metadata(img)
                stats["sizes"].append((md.get("width", 0), md.get("height", 0)))
                stats["aspect_ratios"].append(md.get("aspect_ratio", 0))
                fmt = md.get("format", "UNKNOWN")
                stats["formats"][fmt] = stats["formats"].get(fmt, 0) + 1
        
        if stats and stats.get("total_images", 0) > 0:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total images", stats.get("total_images", 0))
            with col_m2:
                sizes = stats.get("sizes", [])
                if sizes:
                    avg_width = sum(s[0] for s in sizes) / len(sizes)
                    st.metric("Avg width (px)", f"{avg_width:.0f}")
            with col_m3:
                if sizes:
                    avg_height = sum(s[1] for s in sizes) / len(sizes)
                    st.metric("Avg height (px)", f"{avg_height:.0f}")
            with col_m4:
                file_sizes = stats.get("file_sizes_mb", [])
                if file_sizes:
                    avg_sz = sum(file_sizes) / len(file_sizes)
                    st.metric("Avg file size (MB)", f"{avg_sz:.2f}")

            st.markdown("### Class Distribution")
            cc = stats.get("class_counts", {})
            if cc:
                df_cc = pd.DataFrame(list(cc.items()), columns=["class", "count"])
                fig = px.bar(df_cc, x="class", y="count", title="Class count by category", 
                            color="class", text="count")
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

            # Image dimensions (streamlined)
            st.markdown("### Image Dimensions")
            sizes = stats.get("sizes", [])
            if sizes:
                widths = [s[0] for s in sizes]
                heights = [s[1] for s in sizes]
                df_sz = pd.DataFrame({"width": widths, "height": heights})
                fig_2d = px.scatter(df_sz, x="width", y="height", 
                                   title="Image dimensions (Width vs Height)",
                                   opacity=0.6, size_max=10)
                st.plotly_chart(fig_2d, use_container_width=True)

            # Image formats
            st.markdown("### Image Formats")
            fm = stats.get("formats", {})
            if fm:
                df_fm = pd.DataFrame(list(fm.items()), columns=["format", "count"])
                fig_fmt = px.pie(df_fm, names="format", values="count", title="Image format breakdown")
                st.plotly_chart(fig_fmt, use_container_width=True)

            # Model Predictions (expanded)
            st.markdown("### 🤖 Crested Myna Detection Confidence")
            if images:
                # Build a predictions DataFrame using helper (infers true labels when possible)
                df_preds = prepare_predictions_dataframe(images, model_info[2] if model_info[0] else None)
                df_myna = df_preds["confidence"] if not df_preds.empty else pd.Series(dtype=float)
                # Normalize to a DataFrame for backwards compatibility with older visual code
                if isinstance(df_myna, pd.Series):
                    df_myna = pd.DataFrame({"confidence": df_myna.values})
                
                if len(df_myna) > 0:
                    # Row 1: Main confidence distribution for Crested Myna
                    st.markdown("#### Crested Myna Detection Confidence Distribution")
                    fig_conf = px.histogram(df_myna, x="confidence", nbins=20, 
                                           title="Crested Myna detection confidence",
                                           marginal="box", color_discrete_sequence=['#2ca02c'])
                    fig_conf.update_xaxes(range=[0, 1])
                    st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Row 2: Crested Myna prediction metrics
                    col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)
                    mean_conf = df_myna["confidence"].mean()
                    std_conf = df_myna["confidence"].std()
                    median_conf = df_myna["confidence"].median()
                    max_conf = df_myna["confidence"].max()
                    
                    with col_pred1:
                        st.metric("Mean Crested Myna confidence", f"{mean_conf:.3f}")
                    with col_pred2:
                        st.metric("Median confidence", f"{median_conf:.3f}")
                    with col_pred3:
                        st.metric("Std deviation", f"{std_conf:.3f}")
                    with col_pred4:
                        st.metric("Max confidence", f"{max_conf:.3f}")
                    
                    # Row 3: Confidence percentiles for Crested Myna
                    st.markdown("#### Crested Myna Confidence Percentiles")
                    percentiles = [10, 25, 50, 75, 90, 95, 99]
                    percentile_vals = [df_myna["confidence"].quantile(p/100) for p in percentiles]
                    df_percentile = pd.DataFrame({
                        "Percentile": [f"{p}%" for p in percentiles],
                        "Confidence": percentile_vals
                    })
                    fig_percentile = px.bar(df_percentile, x="Percentile", y="Confidence",
                                           title="Crested Myna detection at different percentiles",
                                           text="Confidence", color="Confidence",
                                           color_continuous_scale="Greens")
                    fig_percentile.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig_percentile, use_container_width=True)
                    
                    # Row 4: High/Low confidence Crested Myna samples
                    st.markdown("#### Detection Confidence Extremes")
                    col_extreme1, col_extreme2 = st.columns(2)
                    
                    with col_extreme1:
                        # High confidence threshold
                        high_conf_threshold = df_myna["confidence"].quantile(0.75)
                        high_conf_count = len(df_myna[df_myna["confidence"] >= high_conf_threshold])
                        st.metric("High confidence detections (≥75th percentile)", high_conf_count)
                        if high_conf_count > 0:
                            high_samples = df_myna[df_myna["confidence"] >= high_conf_threshold]["confidence"].describe()
                            st.caption(f"High conf: min={high_samples['min']:.3f}, max={high_samples['max']:.3f}, mean={high_samples['mean']:.3f}")
                    
                    with col_extreme2:
                        # Low confidence threshold
                        low_conf_threshold = df_myna["confidence"].quantile(0.25)
                        low_conf_count = len(df_myna[df_myna["confidence"] <= low_conf_threshold])
                        st.metric("Low confidence detections (≤25th percentile)", low_conf_count)
                        if low_conf_count > 0:
                            low_samples = df_myna[df_myna["confidence"] <= low_conf_threshold]["confidence"].describe()
                            st.caption(f"Low conf: min={low_samples['min']:.3f}, max={low_samples['max']:.3f}, mean={low_samples['mean']:.3f}")
                    
                    # Row 5: Confidence range breakdown for Crested Myna
                    st.markdown("#### Crested Myna Detection Ranges")
                    ranges = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
                    range_counts = [
                        len(df_myna[(df_myna["confidence"] >= 0.0) & (df_myna["confidence"] < 0.2)]),
                        len(df_myna[(df_myna["confidence"] >= 0.2) & (df_myna["confidence"] < 0.4)]),
                        len(df_myna[(df_myna["confidence"] >= 0.4) & (df_myna["confidence"] < 0.6)]),
                        len(df_myna[(df_myna["confidence"] >= 0.6) & (df_myna["confidence"] < 0.8)]),
                        len(df_myna[(df_myna["confidence"] >= 0.8) & (df_myna["confidence"] <= 1.0)]),
                    ]
                    df_ranges = pd.DataFrame({"Confidence Range": ranges, "Count": range_counts})
                    fig_ranges = px.bar(df_ranges, x="Confidence Range", y="Count",
                                       title="Crested Myna detections by confidence range",
                                       text="Count", color="Count",
                                       color_continuous_scale="Greens")
                    st.plotly_chart(fig_ranges, use_container_width=True)
                    
                    # Row 6: Detection summary table
                    st.markdown("#### Crested Myna Detection Summary")
                    summary_stats = {
                        "Metric": [
                            "Total detections",
                            "Mean confidence",
                            "Std deviation",
                            "Min confidence",
                            "25th percentile",
                            "Median (50th percentile)",
                            "75th percentile",
                            "Max confidence"
                        ],
                        "Value": [
                            len(df_myna),
                            f"{mean_conf:.4f}",
                            f"{std_conf:.4f}",
                            f"{df_myna['confidence'].min():.4f}",
                            f"{df_myna['confidence'].quantile(0.25):.4f}",
                            f"{median_conf:.4f}",
                            f"{df_myna['confidence'].quantile(0.75):.4f}",
                            f"{max_conf:.4f}"
                        ]
                    }
                    df_summary_pred = pd.DataFrame(summary_stats)
                    st.dataframe(df_summary_pred, use_container_width=True)
                    # Additional prediction charts
                    st.markdown("### 🔍 Calibration & Confidence Analysis")

                    # Calibration curve (requires inferred labels)
                    cal = calibration_stats(df_preds if not df_preds.empty else None, n_bins=10)
                    if cal["bin_mid"]:
                        fig_cal = go.Figure()
                        fig_cal.add_trace(go.Line(x=cal["bin_mid"], y=cal["avg_conf"], name="Avg predicted confidence", line=dict(color="#2ca02c")))
                        if any(v is not None for v in cal["accuracy"]):
                            fig_cal.add_trace(go.Scatter(x=cal["bin_mid"], y=cal["accuracy"], name="Empirical accuracy", mode="markers+lines", marker=dict(color="#636efa")))
                        fig_cal.update_layout(title="Calibration: predicted confidence vs empirical accuracy", xaxis_title="Confidence bin midpoint", yaxis_title="Fraction / Confidence", yaxis=dict(range=[0, 1]))
                        st.plotly_chart(fig_cal, use_container_width=True)
                    else:
                        st.info("Calibration plot requires predictions with inferred ground-truth labels (place images in class-labeled subfolders).")

                    # Confidence CDF (cumulative distribution)
                    st.markdown("#### Confidence CDF")
                    if not df_preds.empty:
                        sorted_conf = df_preds["confidence"].sort_values().reset_index(drop=True)
                        cdf = pd.DataFrame({"confidence": sorted_conf, "cdf": (sorted_conf.rank(method='first') / len(sorted_conf))})
                        fig_cdf = px.line(cdf, x="confidence", y="cdf", title="Cumulative distribution of predicted confidence")
                        fig_cdf.update_xaxes(range=[0, 1])
                        st.plotly_chart(fig_cdf, use_container_width=True)

                    # Cumulative gain / lift if true labels are available
                    st.markdown("#### Cumulative gain (requires inferred labels)")
                    if not df_preds.empty and df_preds["true_label"].notnull().any():
                        # treat true positives as those where true_label indicates myna
                        df_temp = df_preds.copy()
                        df_temp["is_true_myna"] = df_temp["true_label"].apply(lambda x: True if x and ("myna" in str(x) or "crested" in str(x)) else False)
                        df_temp = df_temp.sort_values("confidence", ascending=False).reset_index(drop=True)
                        df_temp["cum_true_positives"] = df_temp["is_true_myna"].cumsum()
                        df_temp["pct_data"] = (df_temp.index + 1) / len(df_temp)
                        fig_gain = px.line(df_temp, x="pct_data", y="cum_true_positives", title="Cumulative true positives by top-scoring fraction", labels={"pct_data":"Fraction of dataset (top-scoring)", "cum_true_positives":"Cumulative true positives"})
                        st.plotly_chart(fig_gain, use_container_width=True)
                    else:
                        st.info("Cumulative gain requires class-labeled data in subfolders to infer true labels.")

                    # Top / bottom examples
                    st.markdown("### 🔝 Top & 🔚 Bottom predictions")
                    if not df_preds.empty:
                        topk = df_preds.sort_values("confidence", ascending=False).head(5)
                        botk = df_preds.sort_values("confidence", ascending=True).head(5)
                        col_top, col_bot = st.columns(2)
                        with col_top:
                            st.markdown("**Top 5 by confidence**")
                            for _, r in topk.iterrows():
                                st.write(f"{r['image']} — {r['confidence']:.3f}")
                        with col_bot:
                            st.markdown("**Bottom 5 by confidence**")
                            for _, r in botk.iterrows():
                                st.write(f"{r['image']} — {r['confidence']:.3f}")
                    else:
                        st.info("No prediction table available for top/bottom examples.")
                    
                    # Row 7: Resolution vs Confidence Analysis
                    st.markdown("#### Resolution vs Detection Confidence")
                    sizes = stats.get("sizes", [])
                    if sizes and not df_myna.empty and len(df_myna) == len(sizes):
                        # Build dataframe with image dimensions and confidence
                        widths = [s[0] for s in sizes]
                        heights = [s[1] for s in sizes]
                        df_res_conf = pd.DataFrame({
                            "width": widths,
                            "height": heights,
                            "confidence": df_myna["confidence"].values,
                            "aspect_ratio": [w/h if h > 0 else 0 for w, h in zip(widths, heights)]
                        })
                        
                        col_res1, col_res2 = st.columns(2)
                        
                        with col_res1:
                            # Width vs Confidence scatter
                            fig_w_conf = px.scatter(df_res_conf, x="width", y="confidence",
                                                   title="Image Width vs Detection Confidence",
                                                   size_max=8,
                                                   color="confidence", color_continuous_scale="Greens")
                            fig_w_conf.update_yaxes(range=[0, 1])
                            st.plotly_chart(fig_w_conf, use_container_width=True)
                        
                        with col_res2:
                            # Height vs Confidence scatter
                            fig_h_conf = px.scatter(df_res_conf, x="height", y="confidence",
                                                   title="Image Height vs Detection Confidence",
                                                   size_max=8,
                                                   color="confidence", color_continuous_scale="Greens")
                            fig_h_conf.update_yaxes(range=[0, 1])
                            st.plotly_chart(fig_h_conf, use_container_width=True)
                        
                        # Aspect ratio vs Confidence
                        col_ar = st.columns(1)[0]
                        fig_ar_conf = px.scatter(df_res_conf, x="aspect_ratio", y="confidence",
                                               title="Aspect Ratio (Width/Height) vs Detection Confidence",
                                               size_max=8,
                                               color="confidence", color_continuous_scale="Greens")
                        fig_ar_conf.update_yaxes(range=[0, 1])
                        st.plotly_chart(fig_ar_conf, use_container_width=True)
                        
                        # Resolution categories vs Confidence
                        st.markdown("#### Detection Confidence by Resolution Category")
                        
                        # Categorize by resolution
                        df_res_conf["resolution_category"] = df_res_conf.apply(
                            lambda row: "Low (< 480p)" if row["height"] < 480 
                            else "Medium (480-720p)" if row["height"] < 720
                            else "High (720-1080p)" if row["height"] < 1080
                            else "Very High (> 1080p)", axis=1
                        )
                        
                        fig_res_cat = px.box(df_res_conf, x="resolution_category", y="confidence",
                                            title="Detection Confidence Distribution by Resolution Category",
                                            color="resolution_category",
                                            category_orders={"resolution_category": ["Low (< 480p)", "Medium (480-720p)", 
                                                                                     "High (720-1080p)", "Very High (> 1080p)"]})
                        fig_res_cat.update_yaxes(range=[0, 1])
                        st.plotly_chart(fig_res_cat, use_container_width=True)
                        
                        # Resolution summary stats
                        st.markdown("#### Resolution Statistics")
                        res_stats = {
                            "Metric": ["Avg width (px)", "Avg height (px)", "Min width (px)", "Max width (px)", 
                                      "Min height (px)", "Max height (px)", "Avg aspect ratio"],
                            "Value": [
                                f"{df_res_conf['width'].mean():.0f}",
                                f"{df_res_conf['height'].mean():.0f}",
                                f"{df_res_conf['width'].min():.0f}",
                                f"{df_res_conf['width'].max():.0f}",
                                f"{df_res_conf['height'].min():.0f}",
                                f"{df_res_conf['height'].max():.0f}",
                                f"{df_res_conf['aspect_ratio'].mean():.2f}"
                            ]
                        }
                        df_res_stats = pd.DataFrame(res_stats)
                        st.dataframe(df_res_stats, use_container_width=True)
                else:
                    st.info("No predictions available.")
            
            # Simple dataset summary table
            st.markdown("### Dataset Summary")
            cc = stats.get("class_counts", {})
            sizes = stats.get("sizes", [])
            aspect_ratios = stats.get("aspect_ratios", [])
            file_sizes_mb = stats.get("file_sizes_mb", [])
            
            summary_table = {
                "Metric": ["Total images", "Num classes", "Avg image width", "Avg image height", 
                          "Avg aspect ratio", "Avg file size (MB)"],
                "Value": [
                    stats.get("total_images", 0),
                    len(cc) if cc else 0,
                    f"{sum(s[0] for s in sizes) / len(sizes):.0f}" if sizes else "N/A",
                    f"{sum(s[1] for s in sizes) / len(sizes):.0f}" if sizes else "N/A",
                    f"{sum(aspect_ratios) / len(aspect_ratios):.2f}" if aspect_ratios else "N/A",
                    f"{sum(file_sizes_mb) / len(file_sizes_mb):.2f}" if file_sizes_mb else "N/A"
                ]
            }
            df_summary = pd.DataFrame(summary_table)
            st.dataframe(df_summary, use_container_width=True)
        else:
            st.info("📤 No images uploaded yet — use the Upload area above to add images and run analytics.")
            st.markdown("""
            ### How to use

            - Drag & drop multiple images in the Upload area (top of the page)
            - After upload, the Analytics tab will show confidence distributions, calibration (if class labels inferred), CDF, and resolution analysis
            - The Gallery tab shows per-image predictions and download links

            **Note:** This app no longer reads a local `data/` folder; please upload images instead (works on Streamlit Cloud and locally).
            """)

    # Gallery tab (upload-only)
    with tab_gallery:
        if not images:
            st.info("📤 No images to show. Upload images in the main area to see predictions and the gallery.")
        else:
            st.subheader(f"🤖 Model Predictions ({len(images)} images)")

            # Create prediction results (and a DataFrame) using helper
            df_preds = prepare_predictions_dataframe(images, model_info[2] if model_info[0] else None)

            # Grid of images with badges
            cols = st.columns(3)
            for idx, (path, img) in enumerate(images):
                row = df_preds.iloc[idx] if not df_preds.empty and idx < len(df_preds) else None
                with cols[idx % 3]:
                    st.image(img, use_container_width=True)
                    if row is not None:
                        conf_score = float(row["confidence"])
                        if conf_score > 0.7:
                            badge_color = "🟢"
                        elif conf_score > 0.4:
                            badge_color = "🟡"
                        else:
                            badge_color = "🔴"
                        st.markdown(f"**{badge_color} {row['label']}**")
                        st.markdown(f"Confidence: **{conf_score*100:.1f}%**")
                    else:
                        st.markdown("Prediction unavailable")
                    image_download_button(img, filename=f"{Path(path).stem}.png", button_text="⬇️ Download")

            # Summary statistics for uploaded batch
            st.markdown("---")
            st.subheader("📊 Batch Summary")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                avg_conf = df_preds["confidence"].mean() if not df_preds.empty else 0.0
                st.metric("Average Confidence", f"{avg_conf*100:.1f}%")
            with col_s2:
                high_conf = int((df_preds["confidence"] > 0.7).sum()) if not df_preds.empty else 0
                st.metric("High Confidence (>70%)", high_conf)
            with col_s3:
                low_conf = int((df_preds["confidence"] < 0.4).sum()) if not df_preds.empty else 0
                st.metric("Low Confidence (<40%)", low_conf)

            # Detailed results table
            st.markdown("### Detailed Results")
            if not df_preds.empty:
                results_df = df_preds[["image", "label", "confidence"]].rename(columns={"image": "Image", "label": "Label", "confidence": "Score"})
                results_df["Confidence"] = (results_df["Score"] * 100).round(1).astype(str) + "%"
                st.dataframe(results_df, use_container_width=True)
            else:
                st.info("No prediction data available for the uploaded images.")

    st.markdown("---")
    st.caption("This app is a UI wrapper; to enable real predictions, place a PyTorch model in `models/` or implement a TensorFlow loader in `app_utils.py`.")


if __name__ == "__main__":
    main()
