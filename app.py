import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page settings
st.set_page_config(
    page_title="Brain Data Explorer",
    page_icon="🧠",
    layout="wide"
)

# Dashboard heading
st.title("🧠 Interactive Visual Analytics for Brain Data Exploration")
st.write(
    "A simple dashboard to explore brain-related datasets "
    "using interactive charts and data analysis."
)

st.info("Educational project only. This dashboard does not provide medical diagnoses.")

# Sidebar
st.sidebar.header("Dataset Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset (CSV)",
    type=["csv"]
)

# Generate sample demonstration data
def create_sample_data():
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        "Participant_ID": [f"P{i+1:03}" for i in range(n)],
        "Group": np.random.choice(
            ["Control", "Group A", "Group B"], n
        ),
        "Age": np.random.randint(18, 65, n),
        "Memory_Score": np.random.randint(40, 100, n),
        "Reaction_Time": np.random.randint(200, 500, n),
        "Connectivity_Score": np.random.uniform(0.1, 1.0, n),
        "Brain_Activity": np.random.randint(20, 100, n)
    })

# Load dataset
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")
        st.stop()
else:
    df = create_sample_data()
    st.sidebar.caption("Using synthetic sample data.")

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

# Column information
numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()

# Dashboard metrics
st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", len(df))
c2.metric("Total Columns", len(df.columns))
c3.metric("Missing Values", int(df.isna().sum().sum()))
c4.metric("Duplicate Rows", int(df.duplicated().sum()))

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset",
    "Data Filtering",
    "Visualizations",
    "Statistical Analysis",
    "Machine Learning"
])

# TAB 1: Dataset
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique Values": [int(df[c].nunique()) for c in df.columns]
    })

    st.dataframe(column_info, use_container_width=True)

# TAB 2: Filtering
with tab2:
    st.subheader("Filter Dataset")

    filtered_df = df.copy()

    if categorical_columns:
        category = st.selectbox(
            "Select a category column",
            ["None"] + categorical_columns
        )

        if category != "None":
            values = df[category].dropna().unique().tolist()

            selected_values = st.multiselect(
                "Choose values",
                values,
                default=values
            )

            filtered_df = filtered_df[
                filtered_df[category].isin(selected_values)
            ]

    if numeric_columns:
        number_column = st.selectbox(
            "Select a numeric column",
            ["None"] + numeric_columns
        )

        if number_column != "None":
            series = pd.to_numeric(
                filtered_df[number_column], errors="coerce"
            ).dropna()

            if not series.empty and series.min() < series.max():
                low = float(series.min())
                high = float(series.max())

                value_range = st.slider(
                    "Choose numeric range",
                    min_value=low,
                    max_value=high,
                    value=(low, high)
                )

                filtered_df = filtered_df[
                    pd.to_numeric(
                        filtered_df[number_column], errors="coerce"
                    ).between(value_range[0], value_range[1])
                ]

    st.write(f"Records displayed: {len(filtered_df)}")
    st.dataframe(filtered_df, use_container_width=True)

    st.download_button(
        "Download Filtered CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_brain_data.csv",
        mime="text/csv"
    )

# TAB 3: Visualizations
with tab3:
    st.subheader("Interactive Data Visualization")

    if numeric_columns:
        chart_type = st.selectbox(
            "Choose Chart",
            ["Histogram", "Scatter Plot", "Bar Chart", "Correlation Heatmap"]
        )

        if chart_type == "Histogram":
            column = st.selectbox(
                "Choose numeric column",
                numeric_columns
            )

            fig = px.histogram(
                df,
                x=column,
                title=f"Distribution of {column}"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot":
            if len(numeric_columns) >= 2:
                x_axis = st.selectbox(
                    "X-axis",
                    numeric_columns,
                    index=0
                )
                y_axis = st.selectbox(
                    "Y-axis",
                    numeric_columns,
                    index=1
                )

                color_column = st.selectbox(
                    "Color by category",
                    ["None"] + categorical_columns
                )

                fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=(
                        None if color_column == "None"
                        else color_column
                    ),
                    title=f"{y_axis} vs {x_axis}"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("A scatter plot needs at least two numeric columns.")

        elif chart_type == "Bar Chart":
            if categorical_columns:
                category = st.selectbox(
                    "Choose category",
                    categorical_columns
                )
                value = st.selectbox(
                    "Choose numeric measurement",
                    numeric_columns
                )

                grouped = (
                    df.groupby(category, dropna=False)[value]
                    .mean()
                    .reset_index()
                )

                fig = px.bar(
                    grouped,
                    x=category,
                    y=value,
                    title=f"Average {value} by {category}"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical columns are available.")

        elif chart_type == "Correlation Heatmap":
            if len(numeric_columns) >= 2:
                correlation = df[numeric_columns].corr()

                fig = px.imshow(
                    correlation,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Between Measurements"
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Correlation shows association, not necessarily causation."
                )
            else:
                st.warning("A heatmap needs at least two numeric columns.")
    else:
        st.warning("No numeric columns found in the dataset.")

# TAB 4: Statistical Analysis
with tab4:
    st.subheader("Statistical Summary")

    if numeric_columns:
        st.dataframe(
            df[numeric_columns].describe().T.round(2),
            use_container_width=True
        )

        st.subheader("Missing Values by Column")

        missing = (
            df.isna().sum()
            .rename("Missing Values")
            .reset_index()
        )
        missing.columns = ["Column", "Missing Values"]

        fig = px.bar(
            missing,
            x="Column",
            y="Missing Values",
            title="Missing Values in Dataset"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("Duplicate rows:", int(df.duplicated().sum()))
    else:
        st.warning("No numeric data is available for statistical analysis.")

# TAB 5: Machine Learning
with tab5:
    st.subheader("K-Means Clustering")

    st.write(
        "K-Means groups records according to similarities in selected "
        "numeric measurements. The groups are exploratory, not medical categories."
    )

    if len(numeric_columns) >= 2:
        features = st.multiselect(
            "Select at least two numeric features",
            numeric_columns,
            default=numeric_columns[:2]
        )

        number_of_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=6,
            value=3
        )

        if st.button("Run Clustering"):
            if len(features) < 2:
                st.error("Please select at least two features.")
            else:
                work = df[features].replace(
                    [np.inf, -np.inf], np.nan
                )

                valid_rows = work.notna().all(axis=1)

                if valid_rows.sum() <= number_of_clusters:
                    st.error(
                        "There are not enough complete records for clustering."
                    )
                else:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(
                        work.loc[valid_rows]
                    )

                    model = KMeans(
                        n_clusters=number_of_clusters,
                        random_state=42,
                        n_init=10
                    )

                    labels = model.fit_predict(scaled_data)

                    result = df.loc[valid_rows].copy()
                    result["Cluster"] = labels.astype(str)

                    st.success("Clustering completed.")

                    fig = px.scatter(
                        result,
                        x=features[0],
                        y=features[1],
                        color="Cluster",
                        title="K-Means Clustering Results"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(result, use_container_width=True)

                    st.download_button(
                        "Download Clustered Dataset",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="clustered_brain_data.csv",
                        mime="text/csv"
                    )
    else:
        st.warning("Clustering needs at least two numeric columns.")

# Footer
st.markdown("---")
st.caption(
    "Student Project | Python | Streamlit | Pandas | Plotly | Scikit-learn"
)
