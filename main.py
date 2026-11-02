
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
