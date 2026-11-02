import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="분류 모델",
    page_icon="🧠",
    layout="wide",
)

# --------------------------------------------------
# 데이터
# --------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/ghdwlswn814/data/main/stroke.csv"

KOREAN_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)


df = load_data()

# --------------------------------------------------
# 제목
# --------------------------------------------------

st.title("분류 모델")

st.write(
    "나이, 평균 혈당, 체질량지수, 고혈압, 심장병 정보를 이용해 "
    "뇌졸중 여부를 분류합니다."
)

# --------------------------------------------------
# 입력 속성 선택
# --------------------------------------------------

selected_features = st.multiselect(
    "입력으로 사용할 속성을 고르세요.",
    options=FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
    ],
    format_func=lambda x: KOREAN_NAMES[x],
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()

# --------------------------------------------------
# 번호 순으로 정렬
# --------------------------------------------------

work = df.copy()

if "id" in work.columns:
    work = work.sort_values("id").reset_index(drop=True)
else:
    work.insert(0, "id", range(1, len(work) + 1))
    work = work.sort_values("id").reset_index(drop=True)

# --------------------------------------------------
# 숫자형 변환
# --------------------------------------------------

for column in FEATURES:
    work[column] = pd.to_numeric(
        work[column],
        errors="coerce",
    )

work["stroke"] = pd.to_numeric(
    work["stroke"],
    errors="coerce",
)

work = work.dropna(
    subset=["stroke"]
).reset_index(drop=True)

work["stroke"] = work["stroke"].astype(int)

# --------------------------------------------------
# 10명씩 묶어서 앞 3명을 테스트용으로 고정
# --------------------------------------------------

work["묶음"] = np.arange(len(work)) // 10
work["묶음내순서"] = np.arange(len(work)) % 10

test_mask = work["묶음내순서"] < 3

test_df = work.loc[test_mask].copy()
train_df = work.loc[~test_mask].copy()

st.caption(
    f"훈련 데이터: {len(train_df):,}명 / "
    f"테스트 데이터: {len(test_df):,}명"
)

# --------------------------------------------------
# BMI를 선택한 경우에만 훈련 데이터 중앙값으로 결측값 처리
# --------------------------------------------------

if "bmi" in selected_features:

    bmi_median = train_df["bmi"].median()

    train_df["bmi"] = train_df["bmi"].fillna(
        bmi_median
    )

    test_df["bmi"] = test_df["bmi"].fillna(
        bmi_median
    )

    st.write(
        f"체질량지수의 빈칸은 훈련 데이터의 중앙값 "
        f"**{bmi_median:.1f}**으로 채웠습니다."
    )

# --------------------------------------------------
# 선택한 변수에 결측값이 있는 경우
# 훈련 데이터 중앙값으로 처리
# --------------------------------------------------

for column in selected_features:

    if train_df[column].isna().any():

        median_value = train_df[column].median()

        train_df[column] = train_df[column].fillna(
            median_value
        )

        test_df[column] = test_df[column].fillna(
            median_value
        )

X_train_raw = train_df[selected_features]
X_test_raw = test_df[selected_features]

y_train = train_df["stroke"]
y_test = test_df["stroke"]

# --------------------------------------------------
# 크기 맞추기
# 훈련 데이터로만 평균과 표준편차를 계산
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# --------------------------------------------------
# 로지스틱 회귀
# --------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

logistic_model.fit(
    X_train,
    y_train,
)

logistic_train_pred = logistic_model.predict(
    X_train
)

logistic_test_pred = logistic_model.predict(
    X_test
)

logistic_train_acc = accuracy_score(
    y_train,
    logistic_train_pred,
)

logistic_test_acc = accuracy_score(
    y_test,
    logistic_test_pred,
)

# --------------------------------------------------
# 의사결정트리
# 질문 최대 3번
# 마지막 마디에 5명 미만이면 더 나누지 않음
# --------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
)

tree_model.fit(
    X_train,
    y_train,
)

tree_train_pred = tree_model.predict(
    X_train
)

tree_test_pred = tree_model.predict(
    X_test
)

tree_train_acc = accuracy_score(
    y_train,
    tree_train_pred,
)

tree_test_acc = accuracy_score(
    y_test,
    tree_test_pred,
)

# --------------------------------------------------
# 기준 모델
# 입력을 하나도 보지 않고
# 훈련 데이터에서 많은 쪽으로만 답
# --------------------------------------------------

dummy_model = DummyClassifier(
    strategy="most_frequent",
)

dummy_model.fit(
    X_train,
    y_train,
)

dummy_train_pred = dummy_model.predict(
    X_train
)

dummy_test_pred = dummy_model.predict(
    X_test
)

dummy_train_acc = accuracy_score(
    y_train,
    dummy_train_pred,
)

dummy_test_acc = accuracy_score(
    y_test,
    dummy_test_pred,
)

# --------------------------------------------------
# 정확도 카드
# --------------------------------------------------

st.subheader("모델 정확도")


def accuracy_card(
    title,
    test_accuracy,
    train_accuracy,
):
    st.markdown(
        f"""
        <div style="
            border: 1px solid #dddddd;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            height: 100%;
        ">
            <div style="
                font-size: 18px;
                font-weight: 600;
            ">
                {title}
            </div>

            <div style="
                font-size: 42px;
                font-weight: 700;
                margin: 15px 0;
            ">
                {test_accuracy:.1%}
            </div>

            <div style="
                display: flex;
                justify-content: center;
                gap: 35px;
                color: #666666;
            ">
                <div>
                    훈련 데이터<br>
                    <strong>{train_accuracy:.1%}</strong>
                </div>

                <div>
                    테스트 데이터<br>
                    <strong>{test_accuracy:.1%}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


col1, col2, col3 = st.columns(3)

with col1:
    accuracy_card(
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_test_acc,
        logistic_train_acc,
    )

with col2:
    accuracy_card(
        "의사결정트리(질문으로 답하는 모델)",
        tree_test_acc,
        tree_train_acc,
    )

with col3:
    accuracy_card(
        "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
        dummy_test_acc,
        dummy_train_acc,
    )

# --------------------------------------------------
# 두 속성을 가로축과 세로축으로 사용
# --------------------------------------------------

st.subheader("두 속성으로 본 테스트 데이터")

x_feature = selected_features[0]
y_feature = selected_features[1]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정
fixed_values = {}

for feature in selected_features:

    if feature != x_feature and feature != y_feature:

        fixed_values[feature] = test_df[
            feature
        ].median()

if fixed_values:

    fixed_text = []

    for feature, value in fixed_values.items():

        fixed_text.append(
            f"{KOREAN_NAMES[feature]} = {value:.2f}"
        )

    st.write(
        "두 축이 아닌 속성은 테스트 데이터의 중앙값에 "
        "고정해서 계산했습니다: "
        + ", ".join(fixed_text)
    )

# --------------------------------------------------
# 테스트 데이터 산점도
# --------------------------------------------------

negative_mask = test_df["stroke"] == 0
positive_mask = test_df["stroke"] == 1

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=test_df.loc[
            negative_mask,
            x_feature,
        ],
        y=test_df.loc[
            negative_mask,
            y_feature,
        ],
        mode="markers",
        name="아님",
        marker=dict(
            size=7,
        ),
        hovertemplate=(
            f"{KOREAN_NAMES[x_feature]}: %{{x}}<br>"
            f"{KOREAN_NAMES[y_feature]}: %{{y}}<br>"
            "실제 결과: 아님"
            "<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=test_df.loc[
            positive_mask,
            x_feature,
        ],
        y=test_df.loc[
            positive_mask,
            y_feature,
        ],
        mode="markers",
        name="뇌졸중",
        marker=dict(
            size=7,
        ),
        hovertemplate=(
            f"{KOREAN_NAMES[x_feature]}: %{{x}}<br>"
            f"{KOREAN_NAMES[y_feature]}: %{{y}}<br>"
            "실제 결과: 뇌졸중"
            "<extra></extra>"
        ),
    )
)

# --------------------------------------------------
# 로지스틱 회귀 0.5 경계선
# --------------------------------------------------

x_index = selected_features.index(
    x_feature
)

y_index = selected_features.index(
    y_feature
)

coef = logistic_model.coef_[0]
intercept = logistic_model.intercept_[0]

x_min = float(
    test_df[x_feature].min()
)

x_max = float(
    test_df[x_feature].max()
)

y_min = float(
    test_df[y_feature].min()
)

y_max = float(
    test_df[y_feature].max()
)

x_padding = (x_max - x_min) * 0.1

if x_padding == 0:
    x_padding = 1

y_padding = (y_max - y_min) * 0.1

if y_padding == 0:
    y_padding = 1

x_line = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    200,
)

fixed_effect = 0.0

for i, feature in enumerate(
    selected_features
):

    if feature != x_feature and feature != y_feature:

        value = fixed_values[feature]

        scaled_value = (
            value - scaler.mean_[i]
        ) / scaler.scale_[i]

        fixed_effect += (
            coef[i] * scaled_value
        )

boundary_exists = False
y_line = None

if abs(coef[y_index]) > 1e-12:

    x_scaled = (
        x_line - scaler.mean_[x_index]
    ) / scaler.scale_[x_index]

    y_scaled = -(
        intercept
        + fixed_effect
        + coef[x_index] * x_scaled
    ) / coef[y_index]

    y_line = (
        y_scaled * scaler.scale_[y_index]
        + scaler.mean_[y_index]
    )

    visible = (
        (y_line >= y_min - y_padding)
        & (y_line <= y_max + y_padding)
    )

    boundary_exists = bool(
        visible.any()
    )

if boundary_exists:

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="로지스틱 회귀 0.5 경계",
            line=dict(
                width=3,
                dash="dash",
            ),
        )
    )

fig.update_layout(
    height=600,
    xaxis_title=KOREAN_NAMES[
        x_feature
    ],
    yaxis_title=KOREAN_NAMES[
        y_feature
    ],
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

if not boundary_exists:

    st.write(
        "로지스틱 회귀의 0.5 경계선은 "
        "현재 그림의 범위 밖에 있습니다."
    )

# --------------------------------------------------
# 의사결정트리 영역
# --------------------------------------------------

st.subheader("의사결정트리가 나눈 영역")

grid_x = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    80,
)

grid_y = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    80,
)

mesh_x, mesh_y = np.meshgrid(
    grid_x,
    grid_y,
)

grid_data = []

for y_value, x_value in zip(
    mesh_y.ravel(),
    mesh_x.ravel(),
):

    row = {}

    for feature in selected_features:

        if feature == x_feature:

            row[feature] = x_value

        elif feature == y_feature:

            row[feature] = y_value

        else:

            row[feature] = fixed_values[
                feature
            ]

    grid_data.append(row)

grid_df = pd.DataFrame(
    grid_data,
    columns=selected_features,
)

grid_scaled = scaler.transform(
    grid_df
)

grid_prediction = tree_model.predict(
    grid_scaled
)

grid_prediction = grid_prediction.reshape(
    mesh_x.shape
)

fig_tree = go.Figure()

fig_tree.add_trace(
    go.Heatmap(
        x=grid_x,
        y=grid_y,
        z=grid_prediction,
        colorscale=[
            [
                0.0,
                "rgba(100, 160, 230, 0.15)",
            ],
            [
                0.499,
                "rgba(100, 160, 230, 0.15)",
            ],
            [
                0.5,
                "rgba(230, 100, 100, 0.15)",
            ],
            [
                1.0,
                "rgba(230, 100, 100, 0.15)",
            ],
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역",
    )
)

fig_tree.add_trace(
    go.Scatter(
        x=test_df.loc[
            negative_mask,
            x_feature,
        ],
        y=test_df.loc[
            negative_mask,
            y_feature,
        ],
        mode="markers",
        name="아님",
        marker=dict(size=7),
    )
)

fig_tree.add_trace(
    go.Scatter(
        x=test_df.loc[
            positive_mask,
            x_feature,
        ],
        y=test_df.loc[
            positive_mask,
            y_feature,
        ],
        mode="markers",
        name="뇌졸중",
        marker=dict(size=7),
    )
)

fig_tree.update_layout(
    height=600,
    xaxis_title=KOREAN_NAMES[
        x_feature
    ],
    yaxis_title=KOREAN_NAMES[
        y_feature
    ],
)

st.plotly_chart(
    fig_tree,
    use_container_width=True,
)

# --------------------------------------------------
# 의사결정트리 질문
# --------------------------------------------------

st.subheader("의사결정트리가 던진 질문")

tree = tree_model.tree_

tree_questions = []

for node in range(tree.node_count):

    feature_index = tree.feature[node]

    if feature_index >= 0:

        feature_name = selected_features[
            feature_index
        ]

        threshold = tree.threshold[node]

        tree_questions.append(
            (
                feature_name,
                threshold,
            )
        )

if tree_questions:

    for feature_name, threshold in tree_questions:

        st.write(
            f"**{KOREAN_NAMES[feature_name]}**가 "
            f"**{threshold:.2f} 이하인가?**"
        )

else:

    st.write(
        "이 나무에서는 질문을 하지 않았습니다."
    )

actual_question_features = []

for feature_name, threshold in tree_questions:

    if feature_name not in actual_question_features:

        actual_question_features.append(
            feature_name
        )

if actual_question_features:

    st.write(
        "이 나무가 실제로 물은 속성: "
        + ", ".join(
            KOREAN_NAMES[x]
            for x in actual_question_features
        )
    )

else:

    st.write(
        "이 나무가 실제로 물은 속성: 없음"
    )

# --------------------------------------------------
# Graphviz DOT 문자열
# --------------------------------------------------

st.subheader("의사결정트리의 가지 그림")


def make_dot(model, feature_names):

    tree = model.tree_

    lines = [
        "digraph Tree {",
        'graph [rankdir=TB];',
        'node [shape=box, style="rounded,filled"];',
    ]

    for node in range(tree.node_count):

        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        samples = int(
            tree.n_node_samples[node]
        )

        counts = tree.value[node][0]

        negative_count = int(
            round(counts[0])
        )

        positive_count = int(
            round(counts[1])
        )

        if samples > 0:

            positive_ratio = (
                positive_count / samples
            )

        else:

            positive_ratio = 0

        is_leaf = (
            left_child == -1
            and right_child == -1
        )

        if is_leaf:

            predicted_class = int(
                model.classes_[
                    np.argmax(counts)
                ]
            )

            if predicted_class == 1:

                answer = "뇌졸중"
                fill_color = "#f4cccc"

            else:

                answer = "아님"
                fill_color = "#cfe2f3"

            label = (
                f"사람 수: {samples}\\n"
                f"뇌졸중: {positive_count}명\\n"
                f"뇌졸중 비율: "
                f"{positive_ratio:.1%}\\n"
                f"답: {answer}"
            )

        else:

            feature_index = tree.feature[node]

            feature_name = feature_names[
                feature_index
            ]

            threshold = tree.threshold[node]

            label = (
                f"{KOREAN_NAMES[feature_name]}"
                f" ≤ {threshold:.2f}\\n"
                f"사람 수: {samples}\\n"
                f"뇌졸중: {positive_count}명\\n"
                f"뇌졸중 비율: "
                f"{positive_ratio:.1%}"
            )

            fill_color = "#f7f7f7"

        label = label.replace(
            '"',
            '\\"',
        )

        lines.append(
            f'{node} [label="{label}", '
            f'fillcolor="{fill_color}"];'
        )

    for node in range(tree.node_count):

        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        if left_child != -1:

            lines.append(
                f'{node} -> {left_child} '
                f'[label="예"];'
            )

        if right_child != -1:

            lines.append(
                f'{node} -> {right_child} '
                f'[label="아니요"];'
            )

    lines.append("}")

    return "\n".join(lines)


dot_string = make_dot(
    tree_model,
    selected_features,
)

st.graphviz_chart(
    dot_string
)

# --------------------------------------------------
# 답을 내는 마디 수
# --------------------------------------------------

leaf_nodes = []

for node in range(tree.node_count):

    if (
        tree.children_left[node] == -1
        and tree.children_right[node] == -1
    ):

        leaf_nodes.append(node)

negative_leaf_count = 0

for node in leaf_nodes:

    counts = tree.value[node][0]

    predicted_class = int(
        tree_model.classes_[
            np.argmax(counts)
        ]
    )

    if predicted_class == 0:

        negative_leaf_count += 1

st.write(
    f"답을 내는 마디는 모두 "
    f"**{len(leaf_nodes)}칸**이고, "
    f"그중 **{negative_leaf_count}칸**은 "
    f"아님이라고 답합니다."
)

if actual_question_features:

    st.write(
        "이 나무가 실제로 물은 속성: "
        + ", ".join(
            KOREAN_NAMES[x]
            for x in actual_question_features
        )
    )

else:

    st.write(
        "이 나무가 실제로 물은 속성: 없음"
    )
