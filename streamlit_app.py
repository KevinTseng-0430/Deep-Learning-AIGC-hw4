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
    is_streamlit_cloud,
    is_local_deployment,
)
import io
import base64
import pandas as pd
import plotly.express as px
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
        st.header("Data source")
        source = st.radio("Choose image source:", ("Upload image", "Project data folder"))
        
        data_dir = None
        show_data_info = False
        
        if source == "Project data folder":
            st.markdown("### Cloud Deployment Info 🌐")
            if is_streamlit_cloud():
                st.warning("""
                ⚠️ **Running on Streamlit Cloud**
                
                The local `data/` folder is **not available** in cloud unless explicitly pushed to GitHub.
                
                **Recommended:** Use **"Upload image"** mode instead! ✨
                """)
                st.markdown("[📖 Full setup guide](https://github.com/KevinTseng-0430/Deep-Learning-AIGC-hw4/blob/main/DEPLOYMENT_GUIDE.md)")
            else:
                st.info("💻 Running locally — folder access should work fine!")
            
            custom_path = st.text_input("Enter data folder path:", "./data")
            data_dir = Path(custom_path)
            show_data_info = True

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
        **Issues with project data folder?**
        
        See [DEPLOYMENT_GUIDE.md](https://github.com/KevinTseng-0430/Deep-Learning-AIGC-hw4/blob/main/DEPLOYMENT_GUIDE.md) for:
        - Cloud deployment setup
        - Data folder troubleshooting  
        - Cloud storage integration
        
        **Quick tip:** Use "Upload image" mode on Streamlit Cloud for best results! 🚀
        """)

    # Main layout with tabs: Analytics and Gallery
    tab_analytics, tab_gallery = st.tabs(["Analytics 📊", "Gallery 🖼️"])

    # Load images depending on source
    if source == "Upload image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        images = []
        if uploaded:
            img = load_pil_image(uploaded)
            images = [("uploaded", img)]
    else:
        images = []
        if show_data_info and data_dir:
            # Try to load from folder with better error handling
            loaded_images, error_msg = load_images_from_folder(str(data_dir))
            
            if error_msg:
                st.warning(error_msg)
                
                # Provide helpful context
                if not data_dir.exists():
                    st.markdown("""
                    ### 💡 Troubleshooting
                    - **Local deployment**: Make sure the `data/` folder exists with images
                    - **Streamlit Cloud**: Upload images using the "Upload image" option instead
                    - **GitHub integration**: Push your `data/` folder to your repository
                    """)
            else:
                images = loaded_images or []

    # Analytics tab
    with tab_analytics:
        st.subheader("📊 Dataset overview")
        if source == "Project data folder" and data_dir:
            if data_dir.exists():
                stats = compute_detailed_stats(data_dir)
            else:
                stats = None
                st.error(f"Cannot access data folder: {data_dir}")
        else:
            stats = None
        
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
                                   opacity=0.6, trendline="ols", size_max=10)
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
                confs = []
                for p, img in images:
                    _, c = predict_image_stub(img, model_info[2] if model_info[0] else None)
                    confs.append(c)
                
                df_myna = pd.DataFrame({"confidence": confs})
                
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
                    
                    # Row 7: Resolution vs Confidence Analysis
                    st.markdown("#### Resolution vs Detection Confidence")
                    sizes = stats.get("sizes", [])
                    if sizes and len(df_myna) == len(sizes):
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
                                                   trendline="ols", size_max=8,
                                                   color="confidence", color_continuous_scale="Greens")
                            fig_w_conf.update_yaxes(range=[0, 1])
                            st.plotly_chart(fig_w_conf, use_container_width=True)
                        
                        with col_res2:
                            # Height vs Confidence scatter
                            fig_h_conf = px.scatter(df_res_conf, x="height", y="confidence",
                                                   title="Image Height vs Detection Confidence",
                                                   trendline="ols", size_max=8,
                                                   color="confidence", color_continuous_scale="Greens")
                            fig_h_conf.update_yaxes(range=[0, 1])
                            st.plotly_chart(fig_h_conf, use_container_width=True)
                        
                        # Aspect ratio vs Confidence
                        col_ar = st.columns(1)[0]
                        fig_ar_conf = px.scatter(df_res_conf, x="aspect_ratio", y="confidence",
                                               title="Aspect Ratio (Width/Height) vs Detection Confidence",
                                               trendline="ols", size_max=8,
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
            st.info("📤 **No data available**")
            st.markdown("""
            ### How to use the Analytics tab:
            
            **Option 1: Upload Images** (Recommended for Streamlit Cloud)
            - Switch to "Upload image" in the sidebar
            - Drag & drop your images
            - Analytics will show predictions
            
            **Option 2: Project Data Folder** (Requires local setup)
            - Ensure you have a `data/` folder with images
            - **On Streamlit Cloud**: This won't work unless data is committed to GitHub
            - **Locally**: Place images in `./data/crested_myna/` or `./data/other/`
            - Then select "Project data folder" in the sidebar
            
            ### ❓ Why can't I see my data folder on Streamlit Cloud?
            
            Streamlit Cloud deploys from GitHub. The `data/` folder:
            - May not be committed to your repository
            - May be listed in `.gitignore`
            - May be too large for GitHub
            
            ### ✅ Solution
            
            1. **For demo**: Use the image upload feature
            2. **For production**: 
               - Store images in cloud storage (Google Cloud Storage, AWS S3, etc.)
               - Or commit a sample dataset to GitHub (max ~50-100 images recommended)
            """)
            
            # Show sample setup instructions
            with st.expander("📝 See how to set up local data folder"):
                st.code("""
# Create folder structure
mkdir -p data/crested_myna
mkdir -p data/other

# Add your images
# Copy Crested Myna images to data/crested_myna/
# Copy other bird images to data/other/

# Run locally
streamlit run streamlit_app.py
                """, language="bash")

    # Gallery tab
    with tab_gallery:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Image gallery")
            if not images:
                st.info("No images to show. Upload an image or provide a valid data folder path on the sidebar.")
            else:
                # optional class filter
                class_filter = st.selectbox("Filter by class (if available)", options=["All"] + list(stats.get("class_counts", {}).keys()) if data_dir and data_dir.exists() else ["All"])
                cols = st.columns(3)
                for i, (path, img) in enumerate(images):
                    # apply simple filter if classes are by parent folder name
                    if class_filter != "All" and isinstance(path, Path) and path.parent.name != class_filter:
                        continue
                    with cols[i % 3]:
                        caption = Path(path).name if path != "uploaded" else "uploaded"
                        if st.button(f"Select: {caption}", key=f"sel_{i}"):
                            st.session_state.selected = i
                        st.image(img, use_container_width=True, caption=caption)

        with col2:
            st.subheader("Selected image")
            sel = st.session_state.get("selected", 0) if images else None
            # Validate index is within bounds
            if sel is not None and images and sel < len(images):
                path, img = images[sel]
                st.image(img, caption=str(path), use_container_width=True)
                st.markdown("**Prediction**")
                if model_info[0]:
                    pred_text, conf = predict_image_stub(img, model_info[2])
                else:
                    pred_text, conf = predict_image_stub(img, None)
                st.metric(label=pred_text, value=f"{conf*100:.1f}%")
                st.markdown("**Metadata**")
                md = extract_image_metadata(img, path if isinstance(path, Path) else None)
                st.json(md)
                st.markdown("**Actions**")
                image_download_button(img, filename=f"{Path(path).stem}.png")
                if st.button("View raw path"):
                    st.write(str(path))
            else:
                st.info("Select an image from the gallery or upload one to see predictions and actions.")

    st.markdown("---")
    st.caption("This app is a UI wrapper; to enable real predictions, place a PyTorch model in `models/` or implement a TensorFlow loader in `app_utils.py`.")


if __name__ == "__main__":
    main()
