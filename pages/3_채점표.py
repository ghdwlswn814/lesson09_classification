
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="채점표",
    page_icon="📋",
    layout="wide"
)

st.title("📋 채점표")
st.caption(
    "테스트 데이터로 두 분류 모델의 예측 결과를 비교하고 "
    "놓친 환자와 헛짚은 사람을 확인합니다."
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

    df = pd.read_csv(DATA_URL)

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
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # ID와 정답이 없는 행 제거
    df = df.dropna(
        subset=["id", "stroke"]
    )

    # ID 오름차순 정렬
    df = df.sort_values(
        "id"
    ).reset_index(drop=True)

    return df


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 입력 특성 선택
# --------------------------------------------------

st.subheader("1. 분석에 사용할 특성")

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
    "모델에 입력할 특성을 선택하세요.",
    options=list(feature_names.keys()),
    default=default_features,
    format_func=lambda x: feature_names[x]
)

if len(selected_features) < 2:
    st.warning("모델 학습을 위해 특성을 2개 이상 선택하세요.")
    st.stop()


# --------------------------------------------------
# 4. 훈련용 / 테스트용 데이터 분리
# --------------------------------------------------

# 기존 분류 모델 페이지와 동일한 방식
# ID 순서로 10명씩 묶고 앞의 3명을 테스트용으로 지정

df["group_number"] = (
    np.arange(len(df)) // 10
)

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

# 입력 데이터와 정답 데이터
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)


# --------------------------------------------------
# 5. 결측치 처리
# --------------------------------------------------

# BMI를 선택한 경우에만 훈련용 중앙값으로 대체
if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(
        bmi_median
    )

    X_test["bmi"] = X_test["bmi"].fillna(
        bmi_median
    )


# --------------------------------------------------
# 6. 두 모델 학습
# --------------------------------------------------

# 로지스틱 회귀: 표준화
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# 모델 1: 로지스틱 회귀
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)


# 모델 2: 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)


# 기준 모델: 훈련용에서 많은 쪽으로만 답하는 모델
baseline_model = DummyClassifier(
    strategy="most_frequent"
)

baseline_model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 7. 테스트 데이터 예측
# --------------------------------------------------

logistic_pred = logistic_model.predict(
    X_test_scaled
)

tree_pred = tree_model.predict(
    X_test
)

baseline_pred = baseline_model.predict(
    X_test
)


# --------------------------------------------------
# 8. 혼동행렬과 평가 지표 계산 함수
# --------------------------------------------------

def calculate_scores(y_true, y_pred):

    # 행: 실제 / 열: 예측
    # 순서: 뇌졸중(1), 뇌졸중 아님(0)

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

    # 정확도
    accuracy = (
        (TP + TN) / total
        if total > 0 else None
    )

    # 재현율: 실제 뇌졸중 환자 중 찾아낸 비율
    actual_positive = TP + FN

    recall = (
        TP / actual_positive
        if actual_positive > 0 else None
    )

    # 정밀도: 뇌졸중이라고 예측한 사람 중 실제 환자 비율
    predicted_positive = TP + FP

    if predicted_positive == 0:
        precision = None
        f1 = None

    else:
        precision = TP / predicted_positive

        # F1 = 2 * 정밀도 * 재현율 / (정밀도 + 재현율)
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


logistic_scores = calculate_scores(
    y_test,
    logistic_pred
)

tree_scores = calculate_scores(
    y_test,
    tree_pred
)

baseline_scores = calculate_scores(
    y_test,
    baseline_pred
)


# --------------------------------------------------
# 9. 테스트 데이터 규모
# --------------------------------------------------

st.divider()

st.subheader("2. 테스트 데이터")

test_count = len(test_df)

test_stroke_count = int(
    (y_test == 1).sum()
)

test_normal_count = int(
    (y_test == 0).sum()
)

st.markdown(
    f"""
    **테스트용 사람 수: {test_count:,}명**
    · 실제 뇌졸중: {test_stroke_count:,}명
    · 실제 뇌졸중 아님: {test_normal_count:,}명
    """
)


# --------------------------------------------------
# 10. 혼동행렬 표시 함수
# --------------------------------------------------

def show_confusion_matrix(title, scores):

    TP = scores["TP"]
    FN = scores["FN"]
    FP = scores["FP"]
    TN = scores["TN"]

    actual_stroke = TP + FN
    actual_normal = FP + TN

    predicted_stroke = TP + FP
    predicted_normal = FN + TN

    matrix_df = pd.DataFrame(
        [
            [
                f"TP: {TP:,}명",
                f"FN: {FN:,}명",
                f"{actual_stroke:,}명"
            ],
            [
                f"FP: {FP:,}명",
                f"TN: {TN:,}명",
                f"{actual_normal:,}명"
            ],
            [
                f"{predicted_stroke:,}명",
                f"{predicted_normal:,}명",
                f"{TP + FN + FP + TN:,}명"
            ]
        ],
        index=[
            "실제: 뇌졸중",
            "실제: 뇌졸중 아님",
            "열 합계"
        ],
        columns=[
            "예측: 뇌졸중",
            "예측: 뇌졸중 아님",
            "행 합계"
        ]
    )

    st.markdown(f"#### {title}")

    st.caption(
        "행은 실제 정답, 열은 모델의 예측입니다."
    )

    st.table(matrix_df)


# --------------------------------------------------
# 11. 두 모델 혼동행렬
# --------------------------------------------------

st.divider()

st.subheader("3. 두 모델의 혼동행렬")

col1, col2 = st.columns(2)

with col1:

    show_confusion_matrix(
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_scores
    )

with col2:

    show_confusion_matrix(
        "의사결정트리(질문으로 답하는 모델)",
        tree_scores
    )


# --------------------------------------------------
# 12. 평가 지표 비교
# --------------------------------------------------

st.divider()

st.subheader("4. 평가 지표 비교")

def format_score(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


comparison_df = pd.DataFrame(
    [
        {
            "모델": "로지스틱 회귀(확률로 답하는 모델)",
            "정확도": format_score(
                logistic_scores["accuracy"]
            ),
            "재현율": format_score(
                logistic_scores["recall"]
            ),
            "정밀도": format_score(
                logistic_scores["precision"]
            ),
            "F1": format_score(
                logistic_scores["f1"]
            )
        },
        {
            "모델": "의사결정트리(질문으로 답하는 모델)",
            "정확도": format_score(
                tree_scores["accuracy"]
            ),
            "재현율": format_score(
                tree_scores["recall"]
            ),
            "정밀도": format_score(
                tree_scores["precision"]
            ),
            "F1": format_score(
                tree_scores["f1"]
            )
        },
        {
            "모델": "훈련용에서 많은 쪽으로만 답하는 모델",
            "정확도": format_score(
                baseline_scores["accuracy"]
            ),
            "재현율": format_score(
                baseline_scores["recall"]
            ),
            "정밀도": format_score(
                baseline_scores["precision"]
            ),
            "F1": format_score(
                baseline_scores["f1"]
            )
        }
    ]
)

st.dataframe(
    comparison_df,
    hide_index=True,
    use_container_width=True
)


# --------------------------------------------------
# 13. 평가 지표 설명
# --------------------------------------------------

with st.expander("평가 지표는 어떻게 계산할까요?"):

    st.markdown(
        """
        - **정확도**: 전체 테스트 사람 중 정답을 맞힌 비율
        - **재현율**: 실제 뇌졸중 환자 중 뇌졸중이라고 찾아낸 비율
        - **정밀도**: 뇌졸중이라고 예측한 사람 중 실제 환자인 비율
        - **F1**: 정밀도와 재현율을 함께 고려한 지표

        | 지표 | 계산식 |
        |---|---|
        | 정확도 | (TP + TN) / 전체 |
        | 재현율 | TP / (TP + FN) |
        | 정밀도 | TP / (TP + FP) |
        | F1 | 2 × 정밀도 × 재현율 / (정밀도 + 재현율) |

        TP: 실제 뇌졸중이고 뇌졸중이라고 예측

        FN: 실제 뇌졸중인데 뇌졸중이 아니라고 예측

        FP: 실제 뇌졸중이 아닌데 뇌졸중이라고 예측

        TN: 실제 뇌졸중이 아니고 뇌졸중이 아니라고 예측

        뇌졸중이라고 예측한 사람이 0명이면 정밀도와 F1은
        계산할 수 없습니다.
        """
    )


# --------------------------------------------------
# 14. 놓친 환자와 헛짚은 사람
# --------------------------------------------------

st.divider()

st.subheader("5. 놓친 환자와 헛짚은 사람")

st.markdown(
    """
    두 종류의 오류를 직접 확인해 봅시다.

    - **놓친 환자(FN)**: 실제 뇌졸중인데 뇌졸중이 아니라고 예측한 사람
    - **헛짚은 사람(FP)**: 실제 뇌졸중이 아닌데 뇌졸중이라고 예측한 사람
    """
)


# 모델 선택
model_choice = st.selectbox(
    "오류 목록을 확인할 모델을 선택하세요.",
    [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)"
    ]
)

if model_choice.startswith("로지스틱"):

    selected_pred = logistic_pred
    selected_scores = logistic_scores

else:

    selected_pred = tree_pred
    selected_scores = tree_scores


# --------------------------------------------------
# 15. 오류 목록 만들기
# --------------------------------------------------

error_df = test_df.copy()

error_df["actual"] = y_test.values
error_df["predicted"] = selected_pred


# FN: 실제 1, 예측 0
missed_patients = error_df[
    (error_df["actual"] == 1) &
    (error_df["predicted"] == 0)
].copy()


# FP: 실제 0, 예측 1
false_alarm_patients = error_df[
    (error_df["actual"] == 0) &
    (error_df["predicted"] == 1)
].copy()


# 나이 많은 순서
missed_patients = missed_patients.sort_values(
    "age",
    ascending=False
)

false_alarm_patients = false_alarm_patients.sort_values(
    "age",
    ascending=False
)


# --------------------------------------------------
# 16. 목록 출력용 함수
# --------------------------------------------------

def prepare_display(data):

    display_df = data[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease"
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압",
            "heart_disease": "심장병"
        }
    )

    # 고혈압과 심장병을 알아보기 쉽게 표시
    def yes_no(value):

        if pd.isna(value):
            return "알 수 없음"

        if value == 1:
            return "있음"

        if value == 0:
            return "없음"

        return "알 수 없음"

    display_df["고혈압"] = (
        display_df["고혈압"].apply(yes_no)
    )

    display_df["심장병"] = (
        display_df["심장병"].apply(yes_no)
    )

    # 번호는 정수로 표시
    display_df["번호"] = (
        display_df["번호"].astype(int)
    )

    # 앞의 20명만 표시
    display_df = display_df.head(20)

    return display_df


# --------------------------------------------------
# 17. 두 목록 나란히 표시
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        f"#### 놓친 환자 (전체 {len(missed_patients):,}명)"
    )

    st.caption(
        "실제 뇌졸중인데 아니라고 예측한 사람"
    )

    st.dataframe(
        prepare_display(missed_patients),
        hide_index=True,
        use_container_width=True,
        height=500
    )

    if len(missed_patients) > 20:

        st.caption(
            f"전체 {len(missed_patients):,}명 중 "
            "나이가 많은 순서로 앞의 20명만 표시합니다."
        )


with col2:

    st.markdown(
        f"#### 헛짚은 사람 (전체 {len(false_alarm_patients):,}명)"
    )

    st.caption(
        "실제 뇌졸중이 아닌데 뇌졸중이라고 예측한 사람"
    )

    st.dataframe(
        prepare_display(false_alarm_patients),
        hide_index=True,
        use_container_width=True,
        height=500
    )

    if len(false_alarm_patients) > 20:

        st.caption(
            f"전체 {len(false_alarm_patients):,}명 중 "
            "나이가 많은 순서로 앞의 20명만 표시합니다."
        )


# --------------------------------------------------
# 18. 혼동행렬과 오류 목록의 일치 확인
# --------------------------------------------------

st.divider()

st.markdown("#### 오류 목록과 혼동행렬 비교")

FN_count = selected_scores["FN"]
FP_count = selected_scores["FP"]

st.markdown(
    f"""
    - 놓친 환자 목록: **{len(missed_patients):,}명**
      = 혼동행렬의 FN: **{FN_count:,}명**

    - 헛짚은 사람 목록: **{len(false_alarm_patients):,}명**
      = 혼동행렬의 FP: **{FP_count:,}명**
    """
)

if (
    len(missed_patients) == FN_count
    and len(false_alarm_patients) == FP_count
):

    st.success(
        "확인 완료! 두 목록의 인원수가 혼동행렬의 FN, FP와 일치합니다."
    )

else:

    st.error(
        "오류 목록과 혼동행렬의 인원수가 일치하지 않습니다."
    )


# --------------------------------------------------
# 19. 마무리 안내
# --------------------------------------------------

st.divider()

st.caption(
    "주의: 이 페이지는 학습용 데이터 분석 실습입니다. "
    "모델의 예측 결과는 실제 의료 진단을 대신할 수 없습니다."
)
