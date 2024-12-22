import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import cv2
import io
import plotly.graph_objects as go
import plotly.express as px
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
from datetime import datetime
import json
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison


class StyleTransferApp:
    def __init__(self):
        self.model = hub.load(
            "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
        )
        self.setup_database()
        self.history = self.load_history()

    def setup_database(self):
        conn = sqlite3.connect("style_transfer.db")
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS transfers
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     metrics TEXT,
                     settings TEXT)"""
        )
        conn.commit()
        conn.close()

    def load_history(self):
        conn = sqlite3.connect("style_transfer.db")
        df = pd.read_sql_query("SELECT * FROM transfers", conn)
        conn.close()
        if not df.empty:
            df["metrics"] = df["metrics"].apply(json.loads)
            df["settings"] = df["settings"].apply(json.loads)
        return df

    def save_transfer(self, metrics, settings):
        conn = sqlite3.connect("style_transfer.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO transfers (timestamp, metrics, settings) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), json.dumps(metrics), json.dumps(settings)),
        )
        conn.commit()
        conn.close()
        self.history = self.load_history()

    def load_image(self, img):
        img_array = np.array(img)
        img = tf.convert_to_tensor(img_array)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [256, 256])
        img = img[tf.newaxis, :]
        return img

    def process_image(self, content_image, style_image, style_intensity=1.0):
        content = self.load_image(content_image)
        style = self.load_image(style_image)
        outputs = self.model(content, style)
        stylized_image = outputs[0]
        stylized_image = tf.squeeze(stylized_image)
        stylized_image = tf.image.convert_image_dtype(stylized_image, tf.uint8)
        return Image.fromarray(stylized_image.numpy()).resize((256, 256))

    def apply_filters(self, img, filters):
        img_array = np.array(img)

        if filters["grayscale"]:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        if filters["blur"] > 0:
            img_array = cv2.GaussianBlur(img_array, (5, 5), filters["blur"])

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= filters["saturation"]
        hsv[..., 2] *= filters["brightness"]
        hsv[..., 0] += filters["hue"]
        img_array = cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB
        )

        if filters["sharpen"]:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            img_array = cv2.filter2D(img_array, -1, kernel)

        if filters["emboss"]:
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            img_array = cv2.filter2D(img_array, -1, kernel)

        return Image.fromarray(img_array)

    def calculate_metrics(self, original, stylized):
        original = original.resize((256, 256))
        stylized = stylized.resize((256, 256))

        original_gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
        stylized_gray = cv2.cvtColor(np.array(stylized), cv2.COLOR_RGB2GRAY)

        similarity_index = ssim(original_gray, stylized_gray)
        psnr_value = psnr(original_gray, stylized_gray)

        hist_original = cv2.calcHist([np.array(original)], [0], None, [256], [0, 256])
        hist_stylized = cv2.calcHist([np.array(stylized)], [0], None, [256], [0, 256])
        hist_correlation = cv2.compareHist(
            hist_original, hist_stylized, cv2.HISTCMP_CORREL
        )

        return {
            "ssim": similarity_index,
            "psnr": psnr_value,
            "hist_correlation": hist_correlation,
            "mean_pixel_diff": np.mean(np.abs(np.array(original) - np.array(stylized))),
        }

    def plot_metrics_history(self, metrics_df):
        fig = go.Figure()

        metrics = ["ssim", "psnr", "hist_correlation", "mean_pixel_diff"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for metric, color in zip(metrics, colors):
            fig.add_trace(
                go.Scatter(
                    y=[m[metric] for m in metrics_df["metrics"]],
                    name=metric.upper(),
                    line=dict(color=color, width=2),
                    mode="lines+markers",
                )
            )

        fig.update_layout(
            title="Metrics History Over Time",
            xaxis_title="Transfer ID",
            yaxis_title="Metric Value",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig

    def plot_settings_distribution(self, settings_df):
        fig = go.Figure()

        numeric_settings = [
            "style_intensity",
            "blur",
            "brightness",
            "saturation",
            "hue",
        ]

        for setting in numeric_settings:
            values = [s[setting] for s in settings_df["settings"]]
            fig.add_trace(
                go.Violin(
                    y=values, name=setting, box_visible=True, meanline_visible=True
                )
            )

        fig.update_layout(
            title="Distribution of Style Settings",
            yaxis_title="Value",
            template="plotly_white",
            showlegend=False,
        )
        return fig

    def plot_correlation_matrix(self, metrics_df):
        metrics_data = pd.DataFrame([m for m in metrics_df["metrics"]])
        correlation_matrix = metrics_data.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                text=np.round(correlation_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title="Metric Correlation Matrix",
            template="plotly_white",
            width=600,
            height=600,
        )
        return fig


def main():
    st.set_page_config(layout="wide")
    st.title("Style Transfer Lab")

    app = StyleTransferApp()

    tab1, tab2, tab3 = st.tabs(["Style Transfer", "Analytics", "History"])

    with tab1:
        st.caption(
            "Note: The style image will be the image whose distinct features will be applied to the content image (for example, consider uploading an image of Starry Night and see its effect on a photograph of a tree.)"
        )
        content_file = st.file_uploader(
            "Upload content image", type=["png", "jpg", "jpeg"]
        )
        style_file = st.file_uploader("Upload style image", type=["png", "jpg", "jpeg"])

        if content_file and style_file:
            col1, col2 = st.columns([1, 2])

            with col1:
                content_image = Image.open(content_file)
                content_image = content_image.resize((256, 256))
                st.image(content_image, caption="Content Image")

                style_image = Image.open(style_file)
                style_image = style_image.resize((256, 256))
                st.image(style_image, caption="Style Image")

                st.subheader("Style Controls")
                style_intensity = st.slider("Style Intensity", 0.0, 1.0, 0.8)

                st.subheader("Image Filters")
                grayscale = st.checkbox("Grayscale")
                blur = st.slider("Blur", 0.0, 5.0, 0.0)
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
                saturation = st.slider("Saturation", 0.0, 2.0, 1.0)
                hue = st.slider("Hue Shift", -180, 180, 0)
                sharpen = st.checkbox("Sharpen")
                emboss = st.checkbox("Emboss")

                if st.button("Random Style"):
                    st.session_state.update(
                        {
                            "style_intensity": np.random.uniform(0.3, 1.0),
                            "brightness": np.random.uniform(0.7, 1.3),
                            "saturation": np.random.uniform(0.7, 1.3),
                            "hue": np.random.randint(-180, 180),
                        }
                    )
                    st.rerun()

            with col2:
                if st.button("Generate"):
                    with st.spinner("Processing..."):
                        settings = {
                            "style_intensity": style_intensity,
                            "grayscale": grayscale,
                            "blur": blur,
                            "brightness": brightness,
                            "saturation": saturation,
                            "hue": hue,
                            "sharpen": sharpen,
                            "emboss": emboss,
                        }

                        result = app.process_image(
                            content_image, style_image, style_intensity
                        )
                        final_result = app.apply_filters(result, settings)

                        st.subheader("Result Comparison")
                        image_comparison(
                            img1=content_image,
                            img2=final_result,
                            label1="Original",
                            label2="Stylized",
                            width=700,
                        )

                        metrics = app.calculate_metrics(content_image, final_result)
                        app.save_transfer(metrics, settings)

                        st.subheader("Image Metrics")
                        cols = st.columns(4)
                        cols[0].metric("SSIM", f"{metrics['ssim']:.4f}")
                        cols[1].metric("PSNR", f"{metrics['psnr']:.2f}")
                        cols[2].metric(
                            "Histogram Correlation",
                            f"{metrics['hist_correlation']:.4f}",
                        )
                        cols[3].metric(
                            "Mean Pixel Difference", f"{metrics['mean_pixel_diff']:.2f}"
                        )

                        buf = io.BytesIO()
                        final_result.save(buf, format="PNG")
                        st.download_button(
                            "Download Result",
                            buf.getvalue(),
                            "stylized_image.png",
                            "image/png",
                        )

    with tab2:
        if not app.history.empty:
            st.subheader("Metrics Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    app.plot_metrics_history(app.history), use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    app.plot_correlation_matrix(app.history), use_container_width=True
                )

            st.subheader("Settings Analysis")
            st.plotly_chart(
                app.plot_settings_distribution(app.history), use_container_width=True
            )

    with tab3:
        if not app.history.empty:
            st.subheader("Transfer History")

            history_data = []
            for _, row in app.history.iterrows():
                history_data.append(
                    {
                        "Timestamp": row["timestamp"],
                        "SSIM": row["metrics"]["ssim"],
                        "PSNR": row["metrics"]["psnr"],
                        "Style Intensity": row["settings"]["style_intensity"],
                        "Filters Applied": ", ".join(
                            k
                            for k, v in row["settings"].items()
                            if v and k != "style_intensity"
                        ),
                    }
                )

            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)


if __name__ == "__main__":
    main()
