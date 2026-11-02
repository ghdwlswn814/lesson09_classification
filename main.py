import json
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/ghdwlswn814/data/refs/heads/main/stroke.csv"

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "뇌졸중_예측_실습실_기록.json"

COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]


# =========================
# 기록 불러오기 / 저장하기
# =========================
def load_records():
    default = {
        "column_meanings": {column: "" for column in COLUMNS},
        "source": "",
    }

    if not RECORD_FILE.exists():
        return default

    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as file:
            saved = json.load(file)

        default["column_meanings"].update(
            saved.get("column_meanings", {})
        )
        default["source"] = saved.get("source", "")
        return default

    except (OSError, json.JSONDecodeError):
        return default


def save_records(column_meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "column_meanings": column_meanings,
        "source": source,
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# =========================
# 데이터 불러오기
# =========================
@st.cache_data
def load_data():
    data = pd.read_csv(
        DATA_URL,
        encoding="utf-8",
    )
    return data


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()


# =========================
# 저장된 교재 기록 불러오기
# =========================
records = load_records()


# =========================
# 제목
# =========================
st.title("🧠 뇌졸중 예측 실습실")

st.write(
    "뇌졸중(stroke) 여부가 기록된 데이터를 살펴보고 "
    "이후 분류 모델을 실습하기 위한 데이터입니다."
)


# =========================
# 숫자 카드
# =========================
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())

if total_people > 0:
    stroke_ratio = stroke_count / total_people
else:
    stroke_ratio = 0


st.subheader("데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 사람 수", f"{total_people:,}명")

with col2:
    st.metric("열 개수", f"{column_count}개")

with col3:
    st.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")

with col4:
    st.metric("stroke가 1인 비율", f"{stroke_ratio:.1%}")


st.divider()


# =========================
# 열 정보 표
# =========================
st.subheader("열 정보")

st.write(
    "‘우리말 뜻’은 교재를 보고 직접 입력하세요. "
    "입력한 내용은 저장 버튼을 누르면 JSON 파일에 기록됩니다."
)


def get_value_type(series):
    values = series.dropna()

    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(values.tolist())

        if unique_values.issubset({0, 1}):
            return "0 또는 1"

        return "숫자"

    unique_values = values.astype(str).unique().tolist()

    if len(unique_values) <= 10:
        return ", ".join(unique_values)

    return f"범주형 ({len(unique_values)}종)"


table_data = []

for column in df.columns:
    table_data.append(
        {
            "열 이름": column,
            "우리말 뜻": records["column_meanings"].get(column, ""),
            "값의 종류": get_value_type(df[column]),
            "빈 값 개수": int(df[column].isna().sum()),
        }
    )

info_df = pd.DataFrame(table_data)

edited_df = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고하여 직접 입력하세요.",
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
        ),
    },
    key="column_info",
)


# =========================
# 처음 5줄
# =========================
st.divider()

st.subheader("데이터 처음 5줄")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)


# =========================
# 데이터 출처
# =========================
st.divider()

st.subheader("데이터 출처")

source = st.text_area(
    "교재에 있는 데이터 출처를 입력하세요.",
    value=records["source"],
    height=100,
    placeholder="교재의 데이터 출처를 여기에 입력하세요.",
)


# =========================
# 저장
# =========================
if st.button("💾 내 기록 저장", type="primary"):
    meanings = {}

    for _, row in edited_df.iterrows():
        meanings[str(row["열 이름"])] = str(row["우리말 뜻"])

    save_records(meanings, source)

    st.success(
        "저장했습니다. "
        "내_기록/뇌졸중_예측_실습실_기록.json 파일에 기록되었습니다."
    )


st.caption(
    "데이터 출처: ghdwlswn814/data의 stroke.csv"
)
