```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="분류 모델",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/ghdwlswn814/data/refs/heads/main/stroke.csv"

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


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 번호가 없는 경우를 대비해 원래 행 순서로 번호를 만듦
    if "id" not in df.columns:
        df.insert(0, "id", np.arange(1, len(df) + 1))

    return df


df = load_data()

st.title("분류 모델")
st.write("선택한 건강 정보를 이용해 뇌졸중 여부를 분류하는 모델을 비교합니다.")


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
# 데이터 준비
# --------------------------------------------------
work = df.copy()

# 번호 순으로 정렬
work = work.sort_values("id").reset_index(drop=True)

# 입력 변수 숫자형 변환
for col in FEATURES:
    work[col] = pd.to_numeric(work[col], errors="coerce")

# 정답
work["stroke"] = pd.to_numeric(work["stroke"], errors="coerce")
work = work.dropna(subset=["stroke"]).reset_index(drop=True)
work["stroke"] = work["stroke"].astype(int)

# --------------------------------------------------
# 10명씩 묶어서 각 묶음의 앞 3명을 테스트로 고정
# --------------------------------------------------
work["_group_no"] = np.arange(len(work)) // 10
work["_position"] = np.arange(len(work)) % 10

test_mask = work["_position"] < 3

test_df = work.loc[test_mask].copy()
train_df = work.loc[~test_mask].copy()

# 사용자가 요청한 데이터에서는 테스트 데이터가 1,533명이어야 함
st.caption(
    f"훈련 데이터: {len(train_df):,}명 · "
    f"테스트 데이터: {len(test_df):,}명"
)


# --------------------------------------------------
# BMI 결측값 처리
# 선택한 경우에만, 훈련 데이터 중앙값으로 채움
# --------------------------------------------------
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()

    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

    st.write(
        f"체질량지수의 빈칸은 **훈련 데이터의 중앙값 "
        f"{bmi_median:.1f}**으로 채웠습니다."
    )


# --------------------------------------------------
# 선택한 입력 변수의 결측값 처리
# BMI 이외의 입력 변수에 결측값이 있다면
# 훈련 데이터 중앙값으로 처리
# --------------------------------------------------
for col in selected_features:
    if train_df[col].isna().any():
        median_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_value)
        test_df[col] = test_df[col].fillna(median_value)


X_train_raw = train_df[selected_features].copy()
X_test_raw = test_df[selected_features].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()


# --------------------------------------------------
# 크기 맞추기: 표준화
# 훈련 데이터로만 기준을 계산
# --------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)


# --------------------------------------------------
# 1. 로지스틱 회귀
# --------------------------------------------------
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
)

logistic_model.fit(X_train, y_train)

logistic_train_pred = logistic_model.predict(X_train)
logistic_test_pred = logistic_model.predict(X_test)

logistic_train_acc = accuracy_score(y_train, logistic_train_pred)
logistic_test_acc = accuracy_score(y_test, logistic_test_pred)


# --------------------------------------------------
# 2. 의사결정트리
# 질문은 최대 3번
# 마지막 마디에 5명 미만이면 더 나누지 않음
# --------------------------------------------------
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
)

tree_model.fit(X_train, y_train)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

tree_train_acc = accuracy_score(y_train, tree_train_pred)
tree_test_acc = accuracy_score(y_test, tree_test_pred)


# --------------------------------------------------
# 3. 아무 입력도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델
# --------------------------------------------------
dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(X_train, y_train)

dummy_train_pred = dummy_model.predict(X_train)
dummy_test_pred = dummy_model.predict(X_test)

dummy_train_acc = accuracy_score(y_train, dummy_train_pred)
dummy_test_acc = accuracy_score(y_test, dummy_test_pred)


# --------------------------------------------------
# 정확도 카드
# --------------------------------------------------
st.subheader("모델 정확도")

card1, card2, card3 = st.columns(3)


def accuracy_card(container, title, test_acc, train_acc):
    with container:
        st.markdown(
            f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 20px;
                height: 100%;
                text-align: center;
                background-color: #fafafa;
            ">
                <div style="font-size: 18px; font-weight: 600;">
                    {title}
                </div>
                <div style="
                    font-size: 42px;
                    font-weight: 700;
                    margin: 12px 0;
                ">
                    {test_acc:.1%}
                </div>
                <div style="
                    display: flex;
                    justify-content: center;
                    gap: 30px;
                    font-size: 14px;
                    color: #666;
                ">
                    <div>
                        <div>훈련 데이터</div>
                        <strong>{train_acc:.1%}</strong>
                    </div>
                    <div>
                        <div>테스트 데이터</div>
                        <strong>{test_acc:.1%}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


accuracy_card(
    card1,
    "로지스틱 회귀(확률로 답하는 모델)",
    logistic_test_acc,
    logistic_train_acc,
)

accuracy_card(
    card2,
    "의사결정트리(질문으로 답하는 모델)",
    tree_test_acc,
    tree_train_acc,
)

accuracy_card(
    card3,
    "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
    dummy_test_acc,
    dummy_train_acc,
)


# --------------------------------------------------
# 산점도
# --------------------------------------------------
st.subheader("두 속성으로 본 분류 결과")

axis_options = selected_features

x_feature, y_feature = axis_options[:2]

# 선택한 두 속성이 같아지는 상황을 방지
if len(axis_options) >= 2:
    x_feature = axis_options[0]
    y_feature = axis_options[1]


# 다른 속성은 테스트 데이터 중앙값으로 고정
fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = test_df[feature].median()


# --------------------------------------------------
# 로지스틱 회귀 0.5 경계선 계산
# --------------------------------------------------
coef = logistic_model.coef_[0]
intercept = logistic_model.intercept_[0]

x_index = selected_features.index(x_feature)
y_index = selected_features.index(y_feature)

x_mean = scaler.mean_[x_index]
x_scale = scaler.scale_[x_index]

y_mean = scaler.mean_[y_index]
y_scale = scaler.scale_[y_index]


# 다른 변수를 중앙값으로 고정했을 때
fixed_scaled_sum = 0.0

for i, feature in enumerate(selected_features):
    if feature not in [x_feature, y_feature]:
        value = fixed_values[feature]
        fixed_scaled = (value - scaler.mean_[i]) / scaler.scale_[i]
        fixed_scaled_sum += coef[i] * fixed_scaled


# 0.5 경계:
# intercept + beta_x*z_x + beta_y*z_y + fixed = 0
#
# z_y = -(intercept + fixed + beta_x*z_x) / beta_y
#
# 다시 원래 단위로 변환
x_min = float(test_df[x_feature].min())
x_max = float(test_df[x_feature].max())
y_min = float(test_df[y_feature].min())
y_max = float(test_df[y_feature].max())

x_padding = (x_max - x_min) * 0.08 if x_max != x_min else 1
y_padding = (y_max - y_min) * 0.08 if y_max != y_min else 1

x_line = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    200,
)

boundary_visible = True
y_line = None

if abs(coef[y_index]) < 1e-12:
    boundary_visible = False
else:
    z_x = (x_line - x_mean) / x_scale

    z_y = -(
        intercept
        + fixed_scaled_sum
        + coef[x_index] * z_x
    ) / coef[y_index]

    y_line = z_y * y_scale + y_mean

    # 그림 영역 안에 선이 실제로 들어오는지 확인
    if np.all(
        (y_line < y_min - y_padding)
        | (y_line > y_max + y_padding)
    ):
        boundary_visible = False


# --------------------------------------------------
# 의사결정트리 영역 계산
# --------------------------------------------------
grid_x = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    80,
)

grid_y
