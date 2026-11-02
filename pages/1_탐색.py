import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/ghdwlswn814/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()


st.title("🔎 탐색")
st.caption("뇌졸중 데이터의 분포와 그룹별 차이를 살펴봅니다.")


# ============================================================
# 1. 나이와 평균 혈당의 분포
# ============================================================
st.header("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수",
        },
    )
    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
    )
    st.plotly_chart(fig_glucose, use_container_width=True)


# ============================================================
# 2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교
# ============================================================
st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

box_df = df.copy()
box_df["stroke_group"] = box_df["stroke"].map(
    {0: "뇌졸중 없음", 1: "뇌졸중 있음"}
)

col1, col2 = st.columns(2)

with col1:
    age_box = px.box(
        box_df,
        x="stroke_group",
        y="age",
        color="stroke_group",
        points=False,
        title="뇌졸중 여부에 따른 나이",
        labels={
            "stroke_group": "뇌졸중 여부",
            "age": "나이",
        },
    )
    age_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이",
    )
    st.plotly_chart(age_box, use_container_width=True)

with col2:
    glucose_box = px.box(
        box_df.dropna(subset=["avg_glucose_level"]),
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        points=False,
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "stroke_group": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당",
        },
    )
    glucose_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당",
    )
    st.plotly_chart(glucose_box, use_container_width=True)


mean_table = (
    df.groupby("stroke")[["age", "avg_glucose_level"]]
    .mean()
    .rename(index={0: "뇌졸중 없음", 1: "뇌졸중 있음"})
    .rename(columns={
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당",
    })
    .reset_index()
    .rename(columns={"stroke": "그룹"})
)

mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.subheader("두 그룹의 평균값")
st.dataframe(mean_table, use_container_width=True, hide_index=True)


# ============================================================
# 3. 고혈압 여부와 심장병 여부에 따른 뇌졸중 비율
# ============================================================
st.header("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")

col1, col2 = st.columns(2)


def make_rate_table(column, label_map):
    result = (
        df.groupby(column)["stroke"]
        .agg(["mean", "count"])
        .reset_index()
    )

    result["group"] = result[column].map(label_map)
    result["stroke_rate"] = result["mean"] * 100

    return result


hypertension_table = make_rate_table(
    "hypertension",
    {0: "고혈압 없음", 1: "고혈압 있음"},
)

heart_table = make_rate_table(
    "heart_disease",
    {0: "심장병 없음", 1: "심장병 있음"},
)

with col1:
    fig_hypertension = px.bar(
        hypertension_table,
        x="group",
        y="stroke_rate",
        text="stroke_rate",
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={
            "group": "고혈압 여부",
            "stroke_rate": "뇌졸중 비율(%)",
        },
    )
    fig_hypertension.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    fig_hypertension.update_layout(
        yaxis_title="뇌졸중 비율(%)",
        xaxis_title="",
    )
    st.plotly_chart(fig_hypertension, use_container_width=True)

with col2:
    fig_heart = px.bar(
        heart_table,
        x="group",
        y="stroke_rate",
        text="stroke_rate",
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={
            "group": "심장병 여부",
            "stroke_rate": "뇌졸중 비율(%)",
        },
    )
    fig_heart.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    fig_heart.update_layout(
        yaxis_title="뇌졸중 비율(%)",
        xaxis_title="",
    )
    st.plotly_chart(fig_heart, use_container_width=True)


# ============================================================
# 4. BMI 빈 값 그룹과 전체의 뇌졸중 비율
# ============================================================
st.header("4. BMI가 비어 있는 사람의 뇌졸중 비율")

missing_bmi = df["bmi"].isna()

missing_bmi_count = int(missing_bmi.sum())

if missing_bmi_count > 0:
    missing_bmi_stroke_rate = df.loc[missing_bmi, "stroke"].mean() * 100
else:
    missing_bmi_stroke_rate = 0

overall_stroke_rate = df["stroke"].mean() * 100

bmi_rate_table = pd.DataFrame(
    {
        "구분": [
            "BMI가 비어 있는 사람",
            "전체 사람",
        ],
        "사람 수": [
            missing_bmi_count,
            len(df),
        ],
        "뇌졸중 비율(%)": [
            missing_bmi_stroke_rate,
            overall_stroke_rate,
        ],
    }
)

bmi_rate_table["뇌졸중 비율(%)"] = bmi_rate_table["뇌졸중 비율(%)"].round(2)

st.dataframe(
    bmi_rate_table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 5. 흡연 상태별 사람 수
# ============================================================
st.header("5. 흡연 상태별 사람 수")

smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking_table["흡연 상태"] = smoking_table["흡연 상태"].fillna("빈 값")

st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True,
)
