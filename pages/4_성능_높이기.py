
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 성능 높이기")
st.caption(
    "뇌졸중 환자가 드물다는 문제를 클래스 가중치로 해결해 봅시다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "ghdwlswn814/data/main/stroke.csv"
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

@st.cache_data(ttl=3600)
def load_data():

    data = pd.read_csv(DATA_URL)

    numeric_columns = [
        "id",
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
        "stroke"
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=["id", "stroke"]
    )

    data = data.sort_values(
        "id"
    ).reset_index(drop=True)

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 입력 특성 선택
# --------------------------------------------------

st.subheader("1. 분석에 사용할 속성")

feature_names = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

default_features = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

selected_features = st.multiselect(
    "모델에 입력할 속성을 선택하세요.",
    options=list(feature_names.keys()),
    default=default_features,
    format_func=lambda x: feature_names[x]
)

if len(selected_features) < 2:
    st.warning("속성을 2개 이상 선택하세요.")
    st.stop()


# --------------------------------------------------
# 4. 훈련용 / 테스트용 데이터 분리
# --------------------------------------------------

# ID 순서대로 정렬된 데이터에서
# 10명씩 묶어 앞 3명은 테스트용,
# 뒤 7명은 훈련용으로 사용

df["group_position"] = (
    np.arange(len(df)) % 10
)

test_mask = (
    df["group_position"] < 3
)

train_df = df.loc[
    ~test_mask
].copy()

test_df = df.loc[
    test_mask
].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)


# --------------------------------------------------
# 5. 결측치 처리
# --------------------------------------------------

# BMI가 선택되었을 때 훈련용 중앙값으로 대체

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(
        bmi_median
    )

    X_test["bmi"] = X_test["bmi"].fillna(
        bmi_median
    )


# 나머지 선택 속성의 결측치가 있을 경우에도
# 훈련용 중앙값을 사용

for col in selected_features:

    median_value = X_train[col].median()

    X_train[col] = X_train[col].fillna(
        median_value
    )

    X_test[col] = X_test[col].fillna(
        median_value
    )


# --------------------------------------------------
# 6. 표준화
# --------------------------------------------------

# 로지스틱 회귀에 사용할 표준화
# 훈련용 데이터에만 fit

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# --------------------------------------------------
# 7. 가중치 적용 전후 네 모델 학습
# --------------------------------------------------

# 로지스틱 회귀: 가중치 적용 전
logistic_before = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_before.fit(
    X_train_scaled,
    y_train
)


# 로지스틱 회귀: 가중치 적용 후
logistic_after = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

logistic_after.fit(
    X_train_scaled,
    y_train
)


# 의사결정트리: 가중치 적용 전
tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_before.fit(
    X_train,
    y_train
)


# 의사결정트리: 가중치 적용 후
tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

tree_after.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 8. 테스트 데이터 예측
# --------------------------------------------------

logistic_pred_before = logistic_before.predict(
    X_test_scaled
)

logistic_pred_after = logistic_after.predict(
    X_test_scaled
)

tree_pred_before = tree_before.predict(
    X_test
)

tree_pred_after = tree_after.predict(
    X_test
)


# --------------------------------------------------
# 9. 평가 지표 계산 함수
# --------------------------------------------------

def calculate_scores(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[1, 0]
    )

    TP = int(cm[0, 0])
    FN = int(cm[0, 1])
    FP = int(cm[1, 0])
    TN = int(cm[1, 1])

    total = TP + FN + FP + TN

    accuracy = (
        (TP + TN) / total
        if total > 0 else None
    )

    actual_positive = TP + FN

    recall = (
        TP / actual_positive
        if actual_positive > 0 else None
    )

    predicted_positive = TP + FP

    if predicted_positive == 0:

        precision = None
        f1 = None

    else:

        precision = TP / predicted_positive

        denominator = precision + recall

        if denominator == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall
                / denominator
            )

    return {
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "TN": TN,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }


# 네 가지 결과 계산
logistic_scores_before = calculate_scores(
    y_test,
    logistic_pred_before
)

logistic_scores_after = calculate_scores(
    y_test,
    logistic_pred_after
)

tree_scores_before = calculate_scores(
    y_test,
    tree_pred_before
)

tree_scores_after = calculate_scores(
    y_test,
    tree_pred_after
)


# --------------------------------------------------
# 10. 테스트 데이터 규모
# --------------------------------------------------

st.divider()

st.subheader("2. 실험에 사용하는 테스트 데이터")

test_count = len(test_df)

stroke_count = int(
    (y_test == 1).sum()
)

normal_count = int(
    (y_test == 0).sum()
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "테스트용 사람 수",
    f"{test_count:,}명"
)

c2.metric(
    "실제 뇌졸중",
    f"{stroke_count:,}명"
)

c3.metric(
    "실제 뇌졸중 아님",
    f"{normal_count:,}명"
)


# --------------------------------------------------
# 11. 혼동행렬 표 생성 함수
# --------------------------------------------------

def make_confusion_table(before, after):

    def values(scores):

        TP = scores["TP"]
        FN = scores["FN"]
        FP = scores["FP"]
        TN = scores["TN"]

        return {
            "실제 뇌졸중": [
                f"TP: {TP:,}명",
                f"FN: {FN:,}명",
                f"{TP + FN:,}명"
            ],
            "실제 뇌졸중 아님": [
                f"FP: {FP:,}명",
                f"TN: {TN:,}명",
                f"{FP + TN:,}명"
            ],
            "합계": [
                f"{TP + FP:,}명",
                f"{FN + TN:,}명",
                f"{TP + FN + FP + TN:,}명"
            ]
        }

    b = values(before)
    a = values(after)

    result = pd.DataFrame(
        {
            "실제 / 예측": [
                "실제 뇌졸중",
                "실제 뇌졸중 아님",
                "열 합계"
            ],
            "가중치 전: 뇌졸중 예측": b["실제 뇌졸중"],
            "가중치 전: 정상 예측": b["실제 뇌졸중 아님"],
            "가중치 전: 행 합계": b["합계"],
            "가중치 후: 뇌졸중 예측": a["실제 뇌졸중"],
            "가중치 후: 정상 예측": a["실제 뇌졸중 아님"],
            "가중치 후: 행 합계": a["합계"]
        }
    )

    return result


# --------------------------------------------------
# 12. 지표 비교 표 생성 함수
# --------------------------------------------------

def format_score(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


def make_metric_table(before, after):

    metric_names = {
        "정확도": "accuracy",
        "재현율": "recall",
        "정밀도": "precision",
        "F1": "f1"
    }

    rows = []

    for korean_name, key in metric_names.items():

        rows.append(
            {
                "평가 지표": korean_name,
                "가중치 적용 전": format_score(
                    before[key]
                ),
                "가중치 적용 후": format_score(
                    after[key]
                )
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------
# 13. 찾아낸 환자 / 놓친 환자 / 헛짚은 사람
# --------------------------------------------------

def make_change_summary(before, after):

    return (
        f"찾아낸 환자: {before['TP']:,}명 → {after['TP']:,}명"
        f"　|　놓친 환자: {before['FN']:,}명 → {after['FN']:,}명"
        f"　|　헛짚은 사람: {before['FP']:,}명 → {after['FP']:,}명"
    )


# --------------------------------------------------
# 14. 로지스틱 회귀 결과
# --------------------------------------------------

st.divider()

st.subheader(
    "3. 로지스틱 회귀(확률로 답하는 모델)"
)

st.caption(
    "가중치 적용 전과 후의 혼동행렬 및 평가 지표를 비교합니다."
)

st.markdown("#### 혼동행렬 비교")

st.dataframe(
    make_confusion_table(
        logistic_scores_before,
        logistic_scores_after
    ),
    hide_index=True,
    use_container_width=True
)

st.caption(
    "TP: 실제 뇌졸중을 찾아냄 · FN: 실제 뇌졸중을 놓침 · "
    "FP: 정상이지만 뇌졸중이라고 예측 · TN: 정상을 맞힘"
)

st.markdown("#### 평가 지표 비교")

st.dataframe(
    make_metric_table(
        logistic_scores_before,
        logistic_scores_after
    ),
    hide_index=True,
    use_container_width=True
)

st.info(
    make_change_summary(
        logistic_scores_before,
        logistic_scores_after
    )
)


# --------------------------------------------------
# 15. 의사결정트리 결과
# --------------------------------------------------

st.divider()

st.subheader(
    "4. 의사결정트리(질문으로 답하는 모델)"
)

st.caption(
    "가중치 적용 전과 후의 혼동행렬 및 평가 지표를 비교합니다."
)

st.markdown("#### 혼동행렬 비교")

st.dataframe(
    make_confusion_table(
        tree_scores_before,
        tree_scores_after
    ),
    hide_index=True,
    use_container_width=True
)

st.caption(
    "TP: 실제 뇌졸중을 찾아냄 · FN: 실제 뇌졸중을 놓침 · "
    "FP: 정상이지만 뇌졸중이라고 예측 · TN: 정상을 맞힘"
)

st.markdown("#### 평가 지표 비교")

st.dataframe(
    make_metric_table(
        tree_scores_before,
        tree_scores_after
    ),
    hide_index=True,
    use_container_width=True
)

st.info(
    make_change_summary(
        tree_scores_before,
        tree_scores_after
    )
)


# --------------------------------------------------
# 16. 평가 지표 설명
# --------------------------------------------------

with st.expander("평가 지표 계산 방법 보기"):

    st.markdown(
        """
        | 평가 지표 | 계산 방법 |
        |---|---|
        | 정확도 | (TP + TN) / 전체 |
        | 재현율 | TP / (TP + FN) |
        | 정밀도 | TP / (TP + FP) |
        | F1 | 2 × 정밀도 × 재현율 / (정밀도 + 재현율) |

        - **TP**: 실제 뇌졸중이고 뇌졸중이라고 예측
        - **FN**: 실제 뇌졸중인데 뇌졸중이 아니라고 예측
        - **FP**: 실제 뇌졸중이 아닌데 뇌졸중이라고 예측
        - **TN**: 실제 뇌졸중이 아니고 뇌졸중이 아니라고 예측

        가중치 균형 조정은 적은 수의 뇌졸중 환자를
        더 중요하게 학습하도록 만드는 방법입니다.

        이 방법을 사용하면 재현율이 높아질 수 있지만,
        정상인을 뇌졸중이라고 판단하는 FP도 증가할 수 있습니다.
        """
    )


# --------------------------------------------------
# 17. 시각화용 함수
# --------------------------------------------------

def get_model_probability(
    model,
    raw_data,
    model_type
):

    if model_type == "logistic":

        scaled_data = scaler.transform(
            raw_data[selected_features]
        )

        return model.predict_proba(
            scaled_data
        )[:, 1]

    return model.predict_proba(
        raw_data[selected_features]
    )[:, 1]


def make_base_data(size=1):

    base = {}

    for col in selected_features:

        base[col] = float(
            X_train[col].median()
        )

    return base


def get_axis_range(col):

    low = float(
        X_train[col].quantile(0.01)
    )

    high = float(
        X_train[col].quantile(0.99)
    )

    if low == high:
        low -= 1
        high += 1

    padding = (high - low) * 0.08

    return (
        low - padding,
        high + padding
    )


# --------------------------------------------------
# 18. 2차원 산점도와 경계선
# --------------------------------------------------

def draw_2d_graph(
    x_col,
    y_col,
    model_before,
    model_after,
    model_type
):

    x_min, x_max = get_axis_range(x_col)
    y_min, y_max = get_axis_range(y_col)

    x_values = np.linspace(
        x_min,
        x_max,
        100
    )

    y_values = np.linspace(
        y_min,
        y_max,
        100
    )

    xx, yy = np.meshgrid(
        x_values,
        y_values
    )

    grid = pd.DataFrame(
        [
            make_base_data()
            for _ in range(xx.size)
        ]
    )

    grid[x_col] = xx.ravel()
    grid[y_col] = yy.ravel()

    probability_before = get_model_probability(
        model_before,
        grid,
        model_type
    ).reshape(xx.shape)

    probability_after = get_model_probability(
        model_after,
        grid,
        model_type
    ).reshape(xx.shape)

    fig = go.Figure()

    # 실제 뇌졸중 여부에 따른 점
    normal = test_df.loc[
        y_test.values == 0
    ]

    stroke = test_df.loc[
        y_test.values == 1
    ]

    fig.add_trace(
        go.Scatter(
            x=normal[x_col],
            y=normal[y_col],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#3478C8",
                size=7,
                opacity=0.55
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stroke[x_col],
            y=stroke[y_col],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#E04B4B",
                size=9,
                opacity=0.8,
                symbol="circle"
            )
        )
    )

    boundary_messages = []

    # 가중치 적용 전 경계선
    for label, probability, color, dash in [
        (
            "가중치 전 0.5 경계선",
            probability_before,
            "#00A878",
            "dash"
        ),
        (
            "가중치 후 0.5 경계선",
            probability_after,
            "#E28A20",
            "solid"
        )
    ]:

        min_p = float(np.nanmin(probability))
        max_p = float(np.nanmax(probability))

        if min_p <= 0.5 <= max_p:

            fig.add_trace(
                go.Contour(
                    x=x_values,
                    y=y_values,
                    z=probability,
                    name=label,
                    showscale=False,
                    contours=dict(
                        start=0.5,
                        end=0.5,
                        size=1,
                        coloring="none"
                    ),
                    line=dict(
                        color=color,
                        width=3,
                        dash=dash
                    ),
                    hovertemplate=(
                        label +
                        "<br>확률: 0.5<extra></extra>"
                    )
                )
            )

        else:

            boundary_messages.append(
                f"{label}은 현재 그래프 범위 밖에 있습니다."
            )

    fig.update_layout(
        title="실제 뇌졸중 여부와 0.5 예측 경계선",
        xaxis_title=feature_names[x_col],
        yaxis_title=feature_names[y_col],
        height=650,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    for message in boundary_messages:
        st.warning(message)

    if not boundary_messages:
        st.caption(
            "두 경계선 모두 그래프 범위 안에서 확인됩니다."
        )


# --------------------------------------------------
# 19. 3차원 산점도와 0.5 경계면
# --------------------------------------------------

def draw_3d_graph(
    x_col,
    y_col,
    z_col,
    model_before,
    model_after,
    model_type
):

    x_min, x_max = get_axis_range(x_col)
    y_min, y_max = get_axis_range(y_col)
    z_min, z_max = get_axis_range(z_col)

    x_values = np.linspace(
        x_min,
        x_max,
        24
    )

    y_values = np.linspace(
        y_min,
        y_max,
        24
    )

    z_values = np.linspace(
        z_min,
        z_max,
        36
    )

    xx, yy = np.meshgrid(
        x_values,
        y_values
    )

    # 가중치 전후 각각의 경계면 계산
    def calculate_boundary(model):

        grid = pd.DataFrame(
            [
                make_base_data()
                for _ in range(xx.size * len(z_values))
            ]
        )

        # 모든 격자점의 좌표 생성
        grid_x = np.repeat(
            xx.ravel(),
            len(z_values)
        )

        grid_y = np.repeat(
            yy.ravel(),
            len(z_values)
        )

        grid_z = np.tile(
            z_values,
            xx.size
        )

        grid[x_col] = grid_x
        grid[y_col] = grid_y
        grid[z_col] = grid_z

        probability = get_model_probability(
            model,
            grid,
            model_type
        ).reshape(
            xx.shape[0],
            xx.shape[1],
            len(z_values)
        )

        boundary_z = np.full(
            xx.shape,
            np.nan
        )

        # 확률이 0.5를 통과하는 위치를 찾음
        for i in range(xx.shape[0]):

            for j in range(xx.shape[1]):

                values = probability[i, j, :] - 0.5

                crossings = np.where(
                    values[:-1] * values[1:] <= 0
                )[0]

                if len(crossings) == 0:
                    continue

                k = crossings[0]

                p1 = values[k]
                p2 = values[k + 1]

                if np.isclose(p1, p2):
                    ratio = 0.5
                else:
                    ratio = -p1 / (p2 - p1)

                boundary_z[i, j] = (
                    z_values[k]
                    + ratio * (
                        z_values[k + 1] - z_values[k]
                    )
                )

        return boundary_z

    boundary_before = calculate_boundary(
        model_before
    )

    boundary_after = calculate_boundary(
        model_after
    )

    fig = go.Figure()

    # 실제 정상인
    normal = test_df.loc[
        y_test.values == 0
    ]

    # 실제 뇌졸중 환자
    stroke = test_df.loc[
        y_test.values == 1
    ]

    fig.add_trace(
        go.Scatter3d(
            x=normal[x_col],
            y=normal[y_col],
            z=normal[z_col],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#3478C8",
                size=3,
                opacity=0.55
            )
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=stroke[x_col],
            y=stroke[y_col],
            z=stroke[z_col],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#E04B4B",
                size=4,
                opacity=0.85
            )
        )
    )

    # 가중치 전 경계면
    if np.isfinite(boundary_before).any():

        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=boundary_before,
                name="가중치 전 0.5 경계면",
                colorscale=[
                    [0, "#00A878"],
                    [1, "#00A878"]
                ],
                opacity=0.48,
                showscale=False,
                connectgaps=False,
                hovertemplate=(
                    "가중치 전 경계면"
                    "<br>나이/속성 좌표 표시"
                    "<extra></extra>"
                )
            )
        )

    # 가중치 후 경계면
    if np.isfinite(boundary_after).any():

        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=boundary_after,
                name="가중치 후 0.5 경계면",
                colorscale=[
                    [0, "#E28A20"],
                    [1, "#E28A20"]
                ],
                opacity=0.48,
                showscale=False,
                connectgaps=False,
                hovertemplate=(
                    "가중치 후 경계면"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title="3차원 실제 데이터와 0.5 예측 경계면",
        scene=dict(
            xaxis_title=feature_names[x_col],
            yaxis_title=feature_names[y_col],
            zaxis_title=feature_names[z_col]
        ),
        height=750,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    before_exists = np.isfinite(
        boundary_before
    ).any()

    after_exists = np.isfinite(
        boundary_after
    ).any()

    if not before_exists:
        st.warning(
            "가중치 적용 전 0.5 경계면은 현재 그래프 범위 밖에 있습니다."
        )

    if not after_exists:
        st.warning(
            "가중치 적용 후 0.5 경계면은 현재 그래프 범위 밖에 있습니다."
        )

    if before_exists and after_exists:
        st.caption(
            "초록색은 가중치 적용 전, 주황색은 적용 후의 경계면입니다. "
            "마우스로 회전하고 확대하여 살펴보세요."
        )


# --------------------------------------------------
# 20. 경계선 / 경계면 시각화
# --------------------------------------------------

st.divider()

st.subheader("5. 예측 경계선과 경계면 비교")

st.markdown(
    """
    실제 뇌졸중 여부를 점의 색으로 표시하고,
    뇌졸중이라고 예측할 확률이 0.5가 되는 경계를 겹쳐 봅니다.

    - 초록색: 가중치 적용 전
    - 주황색: 가중치 적용 후
    - 빨간 점: 실제 뇌졸중
    - 파란 점: 실제 뇌졸중 아님
    """
)

st.caption(
    "3개 이상의 속성을 선택하면 앞의 세 속성을 3차원 축으로 사용합니다. "
    "나머지 선택 속성은 훈련 데이터의 중앙값으로 고정합니다."
)


# --------------------------------------------------
# 21. 축 선택
# --------------------------------------------------

if len(selected_features) == 2:

    st.markdown("#### 가로축과 세로축 선택")

    axis_col1, axis_col2 = st.columns(2)

    with axis_col1:

        x_col = st.selectbox(
            "가로축",
            options=selected_features,
            index=0,
            format_func=lambda x: feature_names[x]
        )

    with axis_col2:

        y_col = st.selectbox(
            "세로축",
            options=selected_features,
            index=1,
            format_func=lambda x: feature_names[x]
        )

    # 두 축이 겹치는 경우 방지
    if x_col == y_col:
        st.warning("가로축과 세로축은 서로 다른 속성을 선택하세요.")

    else:

        # 로지스틱 회귀
        st.markdown(
            "#### 로지스틱 회귀(확률로 답하는 모델)"
        )

        draw_2d_graph(
            x_col,
            y_col,
            logistic_before,
            logistic_after,
            "logistic"
        )

        # 의사결정트리
        st.markdown(
            "#### 의사결정트리(질문으로 답하는 모델)"
        )

        draw_2d_graph(
            x_col,
            y_col,
            tree_before,
            tree_after,
            "tree"
        )


else:

    st.markdown("#### 3차원 축 선택")

    axis_col1, axis_col2, axis_col3 = st.columns(3)

    with axis_col1:

        x_col = st.selectbox(
            "가로축",
            options=selected_features,
            index=0,
            format_func=lambda x: feature_names[x],
            key="axis_x_3d"
        )

    with axis_col2:

        y_options = [
            x for x in selected_features
            if x != x_col
        ]

        y_col = st.selectbox(
            "세로축",
            options=y_options,
            index=0,
            format_func=lambda x: feature_names[x],
            key="axis_y_3d"
        )

    with axis_col3:

        z_options = [
            x for x in selected_features
            if x not in [x_col, y_col]
        ]

        z_col = st.selectbox(
            "깊이축",
            options=z_options,
            index=0,
            format_func=lambda x: feature_names[x],
            key="axis_z_3d"
        )

    # 로지스틱 회귀
    st.markdown(
        "#### 로지스틱 회귀(확률로 답하는 모델)"
    )

    draw_3d_graph(
        x_col,
        y_col,
        z_col,
        logistic_before,
        logistic_after,
        "logistic"
    )

    # 의사결정트리
    st.markdown(
        "#### 의사결정트리(질문으로 답하는 모델)"
    )

    draw_3d_graph(
        x_col,
        y_col,
        z_col,
        tree_before,
        tree_after,
        "tree"
    )


# --------------------------------------------------
# 22. 마무리
# --------------------------------------------------

st.divider()

st.subheader("6. 실험에서 생각해 볼 점")

st.markdown(
    """
    1. 가중치를 조정한 뒤 찾아낸 뇌졸중 환자는 몇 명 늘었을까요?
    2. 놓친 환자가 줄어드는 대신 헛짚은 사람은 얼마나 늘었을까요?
    3. 정확도가 낮아졌더라도 재현율이 높아졌다면 더 좋은 모델일까요?
    4. 두 모델 중 어느 모델이 뇌졸중 환자를 찾는 데 더 적합할까요?

    **중요:** 클래스 가중치를 조정하면 적은 수의 양성 데이터를
    더 중요하게 학습하지만, 반드시 모든 평가 지표가 좋아지는 것은 아닙니다.
    """
)

st.caption(
    "이 페이지는 분류 모델의 성능 비교를 위한 교육용 실습입니다. "
    "실제 의료 진단을 대신할 수 없습니다."
)
