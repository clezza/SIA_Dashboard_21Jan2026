from pathlib import Path

import re

import altair as alt
import pandas as pd
import streamlit as st

try:
    from wordcloud import STOPWORDS, WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    WORDCLOUD_AVAILABLE = False
    STOPWORDS = set()

BASIC_STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "during",
    "each",
    "even",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "here",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "made",
    "more",
    "most",
    "my",
    "no",
    "not",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "really",
    "said",
    "same",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


st.set_page_config(page_title="SIA Review Pulse", layout="wide")

DATA_PATH = Path(__file__).parent / "Data" / "singapore_airlines_reviews.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["published_date"] = pd.to_datetime(
        df["published_date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce")
    return df


df = load_data().dropna(subset=["published_date"])

DOMAIN_STOPWORDS = {
    "airline",
    "airlines",
    "flight",
    "flights",
    "singapore",
    "sia",
    "plane",
    "air",
    "crew",
    "seat",
    "seats",
}
STOPWORD_SET = set(BASIC_STOPWORDS)
if WORDCLOUD_AVAILABLE:
    STOPWORD_SET = STOPWORD_SET.union(STOPWORDS)
STOPWORD_SET = STOPWORD_SET.union(DOMAIN_STOPWORDS)

if WORDCLOUD_AVAILABLE:

    def render_wordcloud(text: str, title: str) -> None:
        if not text.strip():
            st.info(f"No text available for {title.lower()}.")
            return
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=STOPWORD_SET,
            collocations=False,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.caption(title)
        st.pyplot(fig, use_container_width=True)


def keyword_counts(text_series: pd.Series) -> pd.Series:
    text = " ".join(text_series.dropna().astype(str).tolist()).lower()
    words = re.findall(r"[a-z']+", text)
    tokens = [word for word in words if word not in STOPWORD_SET and len(word) >= 3]
    if not tokens:
        return pd.Series(dtype=int)
    return pd.Series(tokens).value_counts()

st.title("SIA Review Pulse")
st.caption("A quick look at Singapore Airlines review sentiment and trends.")

st.sidebar.header("Filters")

min_date = df["published_date"].min().date()
max_date = df["published_date"].max().date()
default_start = (pd.Timestamp(max_date) - pd.DateOffset(months=12)).date()
if default_start < min_date:
    default_start = min_date

start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=max_date)
if start_date > end_date:
    start_date, end_date = end_date, start_date

rating_min = int(df["rating"].min()) if df["rating"].notna().any() else 1
rating_max = int(df["rating"].max()) if df["rating"].notna().any() else 5
rating_range = st.sidebar.slider(
    "Rating range",
    min_value=rating_min,
    max_value=rating_max,
    value=(rating_min, rating_max),
)

platform_options = sorted(df["published_platform"].dropna().unique().tolist())
platform_filter = st.sidebar.multiselect(
    "Platform",
    options=platform_options,
    default=platform_options,
)

type_options = sorted(df["type"].dropna().unique().tolist())
type_filter = st.sidebar.multiselect(
    "Review type",
    options=type_options,
    default=type_options,
)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

filtered = df[
    (df["published_date"] >= start_ts)
    & (df["published_date"] <= end_ts)
    & (df["rating"] >= rating_range[0])
    & (df["rating"] <= rating_range[1])
]

if platform_filter:
    filtered = filtered[filtered["published_platform"].isin(platform_filter)]
if type_filter:
    filtered = filtered[filtered["type"].isin(type_filter)]

if filtered.empty:
    st.warning("No reviews match the current filters.")
    st.stop()

total_reviews = len(filtered)
avg_rating = filtered["rating"].mean()
positive_pct = (filtered["rating"] >= 4).mean() * 100
negative_pct = (filtered["rating"] <= 2).mean() * 100

st.markdown(
    (
        f"**Current snapshot:** {total_reviews:,} reviews, average rating "
        f"{avg_rating:.2f}. Positive (4–5): {positive_pct:.1f}% · "
        f"Negative (1–2): {negative_pct:.1f}%."
    )
)

st.subheader("Review Statistics")

median_rating = filtered["rating"].median()
helpful_total = filtered["helpful_votes"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{total_reviews:,}")
col2.metric("Average Rating", f"{avg_rating:.2f}")
col3.metric("Median Rating", f"{median_rating:.1f}")
col4.metric("Helpful Votes", f"{helpful_total:,.0f}")

st.subheader("Rating Distribution")
rating_counts = (
    filtered["rating"]
    .dropna()
    .astype(int)
    .value_counts()
    .sort_index()
    .rename("count")
)
rating_percent = (rating_counts / rating_counts.sum() * 100).rename("percent")
rating_percent_df = rating_percent.reset_index().rename(columns={"index": "rating"})
rating_chart = (
    alt.Chart(rating_percent_df)
    .mark_bar()
    .encode(
        x=alt.X("rating:O", title="Rating"),
        y=alt.Y(
            "percent:Q",
            title="Share of reviews (%)",
            scale=alt.Scale(domain=[0, float(rating_percent.max())]),
        ),
        tooltip=["rating:O", "percent:Q"],
    )
    .properties(height=280)
)
st.altair_chart(rating_chart, use_container_width=True)

st.subheader("Reviews Over Time")
reviews_over_time = (
    filtered.set_index("published_date")
    .resample("M")
    .size()
    .rename("reviews")
)
reviews_over_time_df = reviews_over_time.reset_index()
max_reviews = int(reviews_over_time_df["reviews"].max())
time_chart = (
    alt.Chart(reviews_over_time_df)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "published_date:T",
            scale=alt.Scale(
                domain=[
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date) + pd.offsets.MonthEnd(0),
                ]
            ),
            title="Month",
        ),
        y=alt.Y(
            "reviews:Q",
            scale=alt.Scale(domain=[0, max_reviews]),
            title="Reviews",
        ),
        tooltip=["published_date:T", "reviews:Q"],
    )
    .properties(height=280)
)
st.altair_chart(time_chart, use_container_width=True)

st.subheader("Average Rating Over Time (Monthly)")
avg_rating_over_time = (
    filtered.set_index("published_date")["rating"]
    .resample("M")
    .mean()
    .rename("average_rating")
)
st.line_chart(avg_rating_over_time)

st.subheader("Platform Mix")
platform_counts = filtered["published_platform"].value_counts().rename("reviews")
st.bar_chart(platform_counts)

st.subheader("Helpful Review Insights")
helpful_base = filtered.copy()
helpful_base["review_length"] = (
    helpful_base["text"].fillna("").astype(str).str.split().str.len()
)
helpful_only = helpful_base[helpful_base["helpful_votes"] > 0]

if helpful_only.empty:
    st.info("No reviews with helpful votes in the current filters.")
else:
    length_bins = pd.cut(
        helpful_only["review_length"],
        bins=[0, 50, 100, 200, 400, 100000],
        include_lowest=True,
    )
    avg_helpful_by_length = (
        helpful_only.groupby(length_bins)["helpful_votes"]
        .mean()
        .rename("avg_helpful_votes")
    )
    avg_helpful_by_length = avg_helpful_by_length.copy()
    avg_helpful_by_length.index = avg_helpful_by_length.index.astype(str)
    st.bar_chart(avg_helpful_by_length)

    if WORDCLOUD_AVAILABLE:
        helpful_text = " ".join(
            helpful_only["text"].dropna().astype(str).tolist()
        )
        render_wordcloud(helpful_text, "Keywords in Helpful Reviews")
    else:
        st.info(
            "Install the optional dependency to see word clouds: "
            "`python -m pip install wordcloud`"
        )

st.subheader("Helpful Votes vs Sentiment")
sentiment_base = helpful_base.copy()
sentiment_base["sentiment"] = pd.cut(
    sentiment_base["rating"],
    bins=[0, 2, 3, 5],
    labels=["Negative (1-2)", "Neutral (3)", "Positive (4-5)"],
    include_lowest=True,
)
sentiment_base = sentiment_base.dropna(subset=["sentiment"])

if sentiment_base.empty:
    st.info("No reviews with sentiment labels in the current filters.")
else:
    avg_helpful_by_sentiment = (
        sentiment_base.groupby("sentiment")["helpful_votes"]
        .mean()
        .rename("avg_helpful_votes")
    )
    total_helpful_by_sentiment = (
        sentiment_base.groupby("sentiment")["helpful_votes"]
        .sum()
        .rename("total_helpful_votes")
    )
    total_votes = total_helpful_by_sentiment.sum()
    helpful_share = (
        (total_helpful_by_sentiment / total_votes * 100)
        if total_votes > 0
        else total_helpful_by_sentiment * 0
    ).rename("helpful_share_percent")

    col_sent1, col_sent2 = st.columns(2)
    with col_sent1:
        st.caption("Average helpful votes per review")
        st.bar_chart(avg_helpful_by_sentiment)
    with col_sent2:
        st.caption("Share of helpful votes by sentiment")
        st.bar_chart(helpful_share)

st.subheader("Keyword Drilldown")

if WORDCLOUD_AVAILABLE:
    positive_text = " ".join(
        filtered.loc[filtered["rating"] >= 4, "text"].dropna().astype(str).tolist()
    )
    negative_text = " ".join(
        filtered.loc[filtered["rating"] <= 2, "text"].dropna().astype(str).tolist()
    )

    col_pos_wc, col_neg_wc = st.columns(2)
    with col_pos_wc:
        render_wordcloud(positive_text, "Positive Reviews (Ratings 4-5)")
    with col_neg_wc:
        render_wordcloud(negative_text, "Negative Reviews (Ratings 1-2)")
else:
    st.info(
        "Install the optional dependency to see word clouds: "
        "`python -m pip install wordcloud`"
    )

tabs = st.tabs(["Positive (4-5)", "Negative (1-2)"])
sentiment_sets = [
    ("pos", filtered.loc[filtered["rating"] >= 4, "text"]),
    ("neg", filtered.loc[filtered["rating"] <= 2, "text"]),
]

for (key_prefix, series), tab in zip(sentiment_sets, tabs):
    with tab:
        counts = keyword_counts(series)
        if counts.empty:
            st.info("No keywords available for these filters.")
            continue

        filter_text = st.text_input(
            "Filter keywords (contains)",
            value="",
            key=f"{key_prefix}_keyword_filter",
        )
        max_freq = int(counts.max())
        min_freq_default = 2 if max_freq >= 2 else 1
        min_freq = st.slider(
            "Minimum frequency",
            min_value=1,
            max_value=max_freq,
            value=min_freq_default,
            key=f"{key_prefix}_min_freq",
        )
        max_keywords = max(1, min(50, len(counts)))
        top_default = min(20, max_keywords)
        top_n = st.slider(
            "Top keywords",
            min_value=1,
            max_value=max_keywords,
            value=top_default,
            key=f"{key_prefix}_top_n",
        )

        filtered_counts = counts[counts >= min_freq]
        if filter_text.strip():
            filtered_counts = filtered_counts[
                filtered_counts.index.str.contains(
                    filter_text.strip(), case=False, regex=False
                )
            ]
        filtered_counts = filtered_counts.head(top_n)

        if filtered_counts.empty:
            st.info("No keywords match the current keyword filters.")
        else:
            st.bar_chart(filtered_counts.rename("count"))
            st.dataframe(
                filtered_counts.rename("count")
                .reset_index()
                .rename(columns={"index": "keyword"}),
                hide_index=True,
                use_container_width=True,
            )

            selected_keyword = st.selectbox(
                "Inspect reviews containing a keyword",
                options=filtered_counts.index.tolist(),
                key=f"{key_prefix}_keyword_select",
            )
            if selected_keyword:
                subset = filtered[filtered["rating"] >= 4] if key_prefix == "pos" else (
                    filtered[filtered["rating"] <= 2]
                )
                matching = subset[
                    subset["text"]
                    .fillna("")
                    .str.contains(selected_keyword, case=False, regex=False)
                ][
                    [
                        "published_date",
                        "rating",
                        "title",
                        "text",
                        "helpful_votes",
                    ]
                ].sort_values("published_date", ascending=False)
                with st.expander(
                    f"Show reviews with '{selected_keyword}' "
                    f"({len(matching):,})",
                    expanded=False,
                ):
                    st.dataframe(
                        matching,
                        hide_index=True,
                        use_container_width=True,
                    )
