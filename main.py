def save_records(column_meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "column_meanings": {
            column: str(column_meanings.get(column, ""))
            for column in COLUMN_MEANINGS
        },
        "source": str(source),
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def value_type(series):
    """열의 값 형태를 학습용으로 간단하게 설명한다."""
    non_null = series.dropna()

    if pd.api.types.is_numeric_dtype(series):
        unique = sorted(non_null.unique().tolist())
        if set(unique).issubset({0, 1}):
            return "0 또는 1"
        return "숫자"

    values = non_null.astype(str).unique().tolist()

    if len(values) <= 10:
        return ", ".join(values)
    return f"범주형 ({len(values)}종)"


df = load_data()
records = load_records()

total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = stroke_count / total_people if total_people else 0

st.title("🧠 뇌졸중 예측 실습실")
st.caption("뇌졸중(stroke) 여부가 기록된 사람들의 데이터를 살펴보고, 이후 예측 모델을 실습하기 위한 데이터 탐색 화면입니다.")

st.subheader("데이터 한눈에 보기")

card1, card2, card3, card4 = st.columns(4)

card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("열 개수", f"{column_count}개")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
card4.metric("stroke가 1인 비율", f"{stroke_ratio:.1%}")

st.divider()

st.subheader("열 정보")
st.write("우리말 뜻은 교재를 참고하여 직접 입력해 보세요. 입력한 내용은 `내_기록` 폴더의 JSON 파일에 저장됩니다.")

info_rows = []

for column in df.columns:
    info_rows.append(
        {
            "열 이름": column,
            "우리말 뜻": records["column_meanings"].get(column, ""),
            "값의 종류": value_type(df[column]),
            "빈 값 개수": int(df[column].isna().sum()),
        }
    )

info_df = pd.DataFrame(info_rows)

edited_info_df = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "열 이름": st.column_config.TextColumn(disabled=True),
        "우리말 뜻": st.column_config.TextColumn(
            help="교재를 참고하여 우리말 뜻을 직접 입력하세요."
        ),
        "값의 종류": st.column_config.TextColumn(disabled=True),
        "빈 값 개수": st.column_config.NumberColumn(disabled=True),
    },
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    key="column_info_editor",
)

st.divider()

st.subheader("데이터 처음 5줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

st.divider()

st.subheader("데이터 출처")
source_text = st.text_area(
    "교재에 있는 데이터 출처를 입력하세요.",
    value=records["source"],
    height=100,
    placeholder="교재에 적힌 데이터 출처를 여기에 입력하세요.",
)

if st.button("💾 내 기록 저장", type="primary"):
    meanings_to_save = dict(
        zip(edited_info_df["열 이름"], edited_info_df["우리말 뜻"])
    )
    save_records(meanings_to_save, source_text)
    st.success("우리말 뜻과 데이터 출처를 `내_기록/뇌졸중_예측_실습실_기록.json`에 저장했습니다.")

st.caption(
    "데이터 파일: ghdwlswn814/data의 stroke.csv · UTF-8 인코딩 · bmi 열에는 빈 값이 포함되어 있습니다."
)
