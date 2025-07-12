import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import json
import os

st.set_page_config(page_title="Finance App", page_icon="💰", layout="wide")

category_file = "categories.json"

if "categories" not in st.session_state:
    st.session_state.categories = {
        "Uncategorized": [],
    }
    
if os.path.exists(category_file):
    with open(category_file, "r") as f:
        st.session_state.categories = json.load(f)
        
def save_categories():
    with open(category_file, "w") as f:
        json.dump(st.session_state.categories, f)

def categorize_transactions(df):
    df["Category"] = "Uncategorized"
    
    for category, keywords in st.session_state.categories.items():
        if category == "Uncategorized" or not keywords:
            continue
        
        lowered_keywords = [keyword.lower().strip() for keyword in keywords]
        
        for idx, row in df.iterrows():
            details = row["Details"].lower().strip()
            if details in lowered_keywords:
                df.at[idx, "Category"] = category
                
    return df  

def load_transactions(file):
    try:
        df = pd.read_csv(file)
        df.columns = [col.strip() for col in df.columns]
        df["Amount"] = df["Amount"].str.replace(",", "").astype(float)
        df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y") 
        
        return categorize_transactions(df)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def add_keyword_to_category(category, keyword):
    keyword = keyword.strip()
    if keyword and keyword not in st.session_state.categories[category]:
        st.session_state.categories[category].append(keyword)
        save_categories()
        return True
    
    return False

def show_budget_recommendations(df):
    st.subheader("💡 Budget Recommendations Based on Behavior")

    df["Month"] = df["Date"].dt.to_period("M")
    monthly_summary = df.groupby(["Month", "Category"])["Amount"].sum().reset_index()

    if monthly_summary.empty:
        st.info("Not enough data to generate recommendations.")
        return

    recent_month = monthly_summary["Month"].max()
    last_month = monthly_summary[monthly_summary["Month"] == recent_month]
    avg_per_category = monthly_summary.groupby("Category")["Amount"].mean().reset_index()

    comparison = pd.merge(
        last_month,
        avg_per_category,
        on="Category",
        suffixes=("_last", "_avg")
    )

    comparison["Deviation (%)"] = ((comparison["Amount_last"] - comparison["Amount_avg"]) / comparison["Amount_avg"]) * 100
    comparison["Suggested Budget"] = (comparison["Amount_avg"] * 0.9).round(2)

    def status_tag(row):
        if row["Deviation (%)"] > 20:
            return "⚠️ Overspending"
        elif row["Deviation (%)"] < -20:
            return "🟢 Spending Less"
        else:
            return "✅ Within Range"

    def generate_recommendation(row):
        if row["Deviation (%)"] > 20:
            return f"Reduce expenses in {row['Category']}."
        elif row["Deviation (%)"] < -20:
            return f"Good control in {row['Category']}."
        else:
            return "Maintain this level."

    comparison["Status"] = comparison.apply(status_tag, axis=1)
    comparison["Recommendation"] = comparison.apply(generate_recommendation, axis=1)

    st.dataframe(
        comparison[[
            "Category", "Amount_last", "Amount_avg",
            "Deviation (%)", "Suggested Budget", "Status", "Recommendation"
        ]],
        use_container_width=True
    )

    fig = px.bar(
        comparison,
        x="Category",
        y=["Amount_last", "Suggested Budget"],
        barmode="group",
        title="Last Month vs Suggested Budget",
        labels={"value": "Amount", "variable": "Type"},
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Deviation Range Legend"):
        st.markdown("""
        - ✅ **Within Range**: Deviation between -20% and +20%  
        - ⚠️ **Overspending**: Deviation > +20%  
        - 🟢 **Spending Less**: Deviation < -20%
        """)

def show_spending_forecast(df):
    st.subheader("📊 Predicted Spending for Next Month")

    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby(["Month", "Category"])["Amount"].sum().reset_index()

    if monthly.empty:
        st.info("Not enough data to forecast.")
        return

    predictions = []

    for category in monthly["Category"].unique():
        cat_data = monthly[monthly["Category"] == category].copy()
        cat_data = cat_data.sort_values("Month")

        # Assign numeric month numbers for ML
        cat_data["Month_Num"] = range(1, len(cat_data) + 1)
        X = cat_data[["Month_Num"]]
        y = cat_data["Amount"]

        if len(X) >= 2:  # At least 2 months required for a trend
            model = LinearRegression()
            model.fit(X, y)

            next_month = pd.DataFrame({"Month_Num": [X["Month_Num"].max() + 1]})
            predicted = model.predict(next_month)[0]

            predictions.append({
                "Category": category,
                "Predicted Amount": round(predicted, 2)
            })

    if not predictions:
        st.info("Need at least 2 months of data per category to forecast.")
    else:
        forecast_df = pd.DataFrame(predictions)
        st.dataframe(forecast_df, use_container_width=True)

        fig = px.bar(
            forecast_df,
            x="Category",
            y="Predicted Amount",
            title="📈 Predicted Spending by Category (Next Month)",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📊 Finance Dashboard")

    uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_transactions(uploaded_file)

        if df is not None:
            debits_df = df[df["Debit/Credit"] == "Debit"].copy()
            credits_df = df[df["Debit/Credit"] == "Credit"].copy()
            st.session_state.debits_df = debits_df.copy()

            tab1, tab2, tab3, tab4 = st.tabs([
                    "💸 Expenses (Debits)",
                    "💰 Payments (Credits)",
                    "📈 Budget Recommendations",
                    "📊 Spending Forecast"
                    ])


            with tab1:
                new_category = st.text_input("New Category Name")
                add_button = st.button("Add Category")

                if add_button and new_category:
                    if new_category not in st.session_state.categories:
                        st.session_state.categories[new_category] = []
                        save_categories()
                        st.rerun()

                st.subheader("Your Expenses")
                edited_df = st.data_editor(
                    st.session_state.debits_df[["Date", "Details", "Amount", "Category"]],
                    column_config={
                        "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                        "Amount": st.column_config.NumberColumn("Amount", format="%.2f PKR"),
                        "Category": st.column_config.SelectboxColumn(
                            "Category",
                            options=list(st.session_state.categories.keys())
                        )
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="category_editor"
                )

                save_button = st.button("Apply Changes", type="primary")
                if save_button:
                    for idx, row in edited_df.iterrows():
                        new_category = row["Category"]
                        if new_category == st.session_state.debits_df.at[idx, "Category"]:
                            continue
                        details = row["Details"]
                        st.session_state.debits_df.at[idx, "Category"] = new_category
                        add_keyword_to_category(new_category, details)

                st.subheader('Expense Summary')
                category_totals = st.session_state.debits_df.groupby("Category")["Amount"].sum().reset_index()
                category_totals = category_totals.sort_values("Amount", ascending=False)

                st.dataframe(
                    category_totals,
                    column_config={
                        "Amount": st.column_config.NumberColumn("Amount", format="%.2f PKR")
                    },
                    use_container_width=True,
                    hide_index=True
                )

                fig = px.pie(
                    category_totals,
                    values="Amount",
                    names="Category",
                    title="Expenses by Category"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Payments Summary")
                total_payments = credits_df["Amount"].sum()
                st.metric("Total Payments", f"{total_payments:,.2f} PKR")
                st.write(credits_df)

            with tab3:
                show_budget_recommendations(debits_df)

            with tab4:
                show_spending_forecast(debits_df)


main()
