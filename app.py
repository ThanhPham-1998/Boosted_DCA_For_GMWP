import streamlit as st
import numpy as np
import random

from algorithms import BDCA, DCA, ConstrainedBDCA, ConstrainedDCA, ConstrainedBDCAV2
from visualization import plot_clusters, plot_convergence, generate_solution_movie, generate_random_colors
from utils import project_onto_linf_ball, project_onto_l1_ball

np.random.seed(1111)
random.seed(42)


st.set_page_config(page_title="Boosted DCA Algorithms", layout="wide")
st.title("🎯 Minh hoạ thuật toán giải  bài toán GMWP")

st.sidebar.header("⚙️ Cấu hình chạy")
algorithm_choice = st.sidebar.selectbox(
    "Chọn thuật toán",
    ["BDCA", "DCA", "ConstrainedBDCA", "ConstrainedDCA", "ConstrainedBDCAV2"]
)

project_function = st.sidebar.selectbox(
    "Chọn projected function",
    [
        "project_onto_linf_ball",
        "project_onto_l1_ball"
    ]
)

norm_type = st.sidebar.selectbox(
    "Chọn norm type",
    [
        "1",
        "2",
        "inf"
    ]
)

def get_parameters_ui():
    st.sidebar.header("⚙️ Tham số thuật toán")
    params = {
        'mu': st.sidebar.number_input('mu', value=0.5),
        'muf': st.sidebar.number_input('muf', value=0.01),
        'delta': st.sidebar.number_input('delta', value=0.5),
        'inner_norm': st.sidebar.number_input('inner_norm', value=1e-5, format="%.1e"),
        'outer_norm': st.sidebar.number_input('outer_norm', value=1e-5, format="%.1e"),
        'alpha': st.sidebar.number_input('alpha', value=0.01),
        'beta': st.sidebar.number_input('beta', value=0.5),
        'lambda_start': st.sidebar.number_input('lambda_start', value=1.0),
        'gamma': st.sidebar.number_input('gamma', value=1.5),
        'lambda_history': st.sidebar.number_input('lambda_history', value=3),
        'max_search': st.sidebar.number_input('max_search', value=5),
        'lambda_min': st.sidebar.number_input('lambda_min', value=0.1),
        'lambda_skip': st.sidebar.number_input('lambda_skip', value=2),
        'q': st.sidebar.number_input('q', value=0.1)
    }
    return params

num_clusters = st.sidebar.number_input("Số cụm", min_value=2, max_value=100, value=4)
num_points = st.sidebar.number_input("Số điểm", min_value=2, max_value=10000, value=100)
fps = st.sidebar.number_input("FPS cho video", 1, 200, 50)

parameters = get_parameters_ui()

st.sidebar.markdown("### 🧭 Centers")
center_mode = st.sidebar.radio("Cách chọn tâm", ["Random", "Nhập thủ công"])

if center_mode == "Random":
    centers = np.random.rand(num_clusters, 2)

else:
    st.sidebar.markdown("Nhập tọa độ từng cụm:")
    centers = []
    for i in range(num_clusters):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x_val = st.number_input(f"Cụm {i+1} - X", key=f"x_{i}")
        with col2:
            y_val = st.number_input(f"Cụm {i+1} - Y", key=f"y_{i}")
        centers.append([x_val, y_val])
    centers = np.array(centers)

st.sidebar.markdown("### 📊 Dữ liệu (Points)")
data_mode = st.sidebar.radio("Cách tạo dữ liệu", ["Random", "Nhập thủ công"])
if data_mode == "Random":
    a = 2 + 2.5 * np.random.randn(100, 2)
else:
    st.sidebar.markdown("Nhập tọa độ từng điểm (x, y):")
    a = []
    for i in range(num_points):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x_val = st.number_input(f"Điểm {i+1} - X", key=f"px_{i}")
        with col2:
            y_val = st.number_input(f"Điểm {i+1} - Y", key=f"py_{i}")
        a.append([x_val, y_val])
    a = np.array(a)

# Map tên → class
algo_map = {
    "BDCA": BDCA,
    "DCA": DCA,
    "ConstrainedBDCA": ConstrainedBDCA,
    "ConstrainedDCA": ConstrainedDCA,
    "ConstrainedBDCAV2": ConstrainedBDCAV2
}

proj_func_map = {
    "project_onto_l1_ball": project_onto_l1_ball,
    "project_onto_linf_ball": project_onto_linf_ball
}

norm_type_map = {
    "1": 1,
    "2": 2,
    "inf": np.inf
}
colors = generate_random_colors(num_clusters)

if st.button("🚀 Chạy thuật toán"):
    st.subheader(f"Thuật toán: {algorithm_choice}")

    AlgorithmClass = algo_map[algorithm_choice]
    alg = AlgorithmClass(save_to_file=True, parameters=parameters)

    with st.spinner("Đang chạy thuật toán..."):
        (x, iter_count, iter_logs), total_time = alg.run(
            a, centers, proj_func_map[project_function], norm_type_map[norm_type])

    # Hiển thị kết quả
    st.success(f"✅ Hoàn tất trong khoảng thời gian {total_time} (s).")
    fig = plot_clusters(
        a=a,
        x=x,
        indices=iter_logs[iter_count]["solution"][-1]["indices"],
        colors=colors
    )
    st.pyplot(fig, use_container_width=True)
    st.caption(f"📌 Hình 1: Kết quả phân cụm bằng thuật toán {algorithm_choice}")

    fig = plot_convergence(
        iter_logs=iter_logs,
        norm_type=norm_type_map[norm_type],
    )

    st.pyplot(fig, use_container_width=True)
    st.caption(f"📌 Hình 2: Các thông số của thuật toán {algorithm_choice}")
    # Sinh video
    video_path = f"{algorithm_choice}_result.mp4"

    generate_solution_movie(
        iter_logs, a, video_path,
        fps=fps, num_clusters=num_clusters,
        colors=colors
    )
    st.video(video_path)
    st.caption(f"📌 Video 1: Minh họa quá trình hội tụ của {algorithm_choice}")
st.markdown("---")
st.caption("App demo cho các thuật toán DCA • Made with ❤️ by Tieu Thanh")