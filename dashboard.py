import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression, LogisticRegression # Added Logistic
from itertools import combinations

SETTINGS_FILE = "exclusion_settings.json"
DEFAULT_EXCLUDES = [
    'livability','head_placed','farm_no','house_no', 
    'goal_weight','mortality_rate','age_days',
    'hen_age_weeks','production_days','feed_efficiency'
]

def load_permanent_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except:
            return DEFAULT_EXCLUDES
    return DEFAULT_EXCLUDES

def save_permanent_settings(exclusions):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(exclusions, f)

if 'excluded_vars' not in st.session_state:
    st.session_state.excluded_vars = load_permanent_settings()

if 'top_2d_global' not in st.session_state:
    st.session_state.top_2d_global = None
if 'top_3d_global' not in st.session_state:
    st.session_state.top_3d_global = None
if 'target_results' not in st.session_state:
    st.session_state.target_results = None

st.set_page_config(page_title="Litter Survey Dashboard", layout="wide")
st.title("Litter Survey Dashboard")

try:
    df_raw = pd.read_csv('../data/processed/Final_Cleaned_Litter_Survey.csv')
    all_cols = df_raw.columns.tolist()

    st.sidebar.header("Variable Exclusions")
    
    chosen_excludes = st.sidebar.multiselect(
        "Select variables to ignore:",
        options=all_cols,
        default=st.session_state.excluded_vars
    )

    if chosen_excludes != st.session_state.excluded_vars:
        st.session_state.excluded_vars = chosen_excludes
        save_permanent_settings(chosen_excludes)
        st.rerun()

    df_filtered = df_raw.drop(columns=st.session_state.excluded_vars, errors='ignore')

    df_numeric_only = df_filtered.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna()
    df_numeric_only = df_numeric_only.astype(np.float64)
    numeric_cols = df_numeric_only.columns.tolist()

    auto_cat_cols = [col for col in df_filtered.select_dtypes(include=['object', 'category']).columns 
                    if df_filtered[col].nunique() < 25]

    mode = st.sidebar.selectbox("Analysis Mode", [
        "1 Variable (Distribution)", 
        "2 Variables (Correlation)", 
        "3 Variables (3D Plane)",
        "Correlation Matrix",
        "Categorical Analysis (Bar Charts)",
        "Logistic Regression (Binary Class)"
    ])

    st.sidebar.markdown("---")
    st.sidebar.header("Target-Specific Discovery")
    
    if numeric_cols:
        target_var = st.sidebar.selectbox("Pick a Target Variable", numeric_cols)
        positive = st.sidebar.checkbox("Positive Correlation", True)
        negative = st.sidebar.checkbox("Negative Correlation", True)
        size_head = st.sidebar.number_input("# of Columns to Display: ", 1, 50, 5)

        if st.sidebar.button(f"Analyze for {target_var}"):
            r_list = []
            for col in numeric_cols:
                if col != target_var:
                    r = df_numeric_only[target_var].corr(df_numeric_only[col])
                    if (positive and r >= 0) or (negative and r <= 0):
                        r_list.append({'Variable': col, 'r': round(r, 3), 'abs_r': abs(r)})
            
            trio_list = []
            others = [c for c in numeric_cols if c != target_var]
            for p1, p2 in combinations(others, 2):
                X = df_numeric_only[[p1, p2]].values
                y = df_numeric_only[target_var].values
                score = LinearRegression().fit(X, y).score(X, y)
                trio_list.append({'Predictors': f"{p1} + {p2}", 'R²': round(score, 3)})
            
            st.session_state.target_results = {
                'name': target_var,
                '2d': pd.DataFrame(r_list).sort_values(by='abs_r', ascending=False).head(size_head),
                '3d': pd.DataFrame(trio_list).sort_values(by='R²', ascending=False).head(size_head)
            }

        if st.session_state.target_results:
            res = st.session_state.target_results
            st.sidebar.info(f"Results for: {res['name']}")
            st.sidebar.write("**Top 2D Partners (R)**")
            st.sidebar.table(res['2d'][['Variable', 'r']])
            st.sidebar.write("**Top 3D Pairings (R²)**")
            st.sidebar.table(res['3d'])
    
    st.sidebar.markdown("---")

    st.sidebar.header("Global Discovery")
    globalpositive = st.sidebar.checkbox("Positive Correlation", True, key="GlobalPos")
    globalnegative = st.sidebar.checkbox("Negative Correlation", True, key="GlobalNeg")
    globalhead = st.sidebar.number_input("# of Columns to Display: ", 1, 50, 10, key="GlobalHead")
    
    if mode == "2 Variables (Correlation)":
        if st.sidebar.button("Find Top 10 2D Links"):
            pairs = list(combinations(numeric_cols, 2))
            results = []
            for p1, p2 in pairs:
                r = df_numeric_only[p1].corr(df_numeric_only[p2])
                if not np.isnan(r):
                    if (globalpositive and r >= 0) or (globalnegative and r <= 0):
                        results.append({'A': p1, 'B': p2, 'r': round(r, 3), 'abs_r': abs(r)})
            st.session_state.top_2d_global = pd.DataFrame(results).sort_values(by='abs_r', ascending=False).head(globalhead)

    elif mode == "3 Variables (3D Plane)":
        if st.sidebar.button("Find Top 10 3D Models"):
            trios = []
            for target in numeric_cols:
                others = [c for c in numeric_cols if c != target]
                for p1, p2 in combinations(others, 2):
                    X, y = df_numeric_only[[p1, p2]].values, df_numeric_only[target].values
                    score = LinearRegression().fit(X, y).score(X, y)
                    trios.append({'Target': target, 'Predictors': f"{p1} + {p2}", 'R²': round(score, 3)})
            st.session_state.top_3d_global = pd.DataFrame(trios).sort_values(by='R²', ascending=False).head(10)

    if mode == "2 Variables (Correlation)" and st.session_state.top_2d_global is not None:
        st.sidebar.table(st.session_state.top_2d_global[['A', 'B', 'r']])
    
    if mode == "3 Variables (3D Plane)" and st.session_state.top_3d_global is not None:
        st.sidebar.table(st.session_state.top_3d_global)

    if mode == "1 Variable (Distribution)":
        var = st.selectbox("Select Variable", numeric_cols)
        st.plotly_chart(px.histogram(df_numeric_only, x=var, marginal="box", color_discrete_sequence=['#2E8B57']), use_container_width=True)

    elif mode == "2 Variables (Correlation)":
        c1, c2 = st.columns(2)
        xv, yv = c1.selectbox("X", numeric_cols, 0), c2.selectbox("Y", numeric_cols, 1)
        st.metric("Correlation (r)", f"{df_numeric_only[xv].corr(df_numeric_only[yv]):.3f}")
        st.plotly_chart(px.scatter(df_numeric_only, x=xv, y=yv, trendline="ols"), use_container_width=True)

    elif mode == "3 Variables (3D Plane)":
        if len(numeric_cols) >= 3:
            c1, c2, c3 = st.columns(3)
            xv, yv, zv = c1.selectbox("X", numeric_cols, 0), c2.selectbox("Y", numeric_cols, 1), c3.selectbox("Z (Target)", numeric_cols, 2)
            X_vals, Z_vals = df_numeric_only[[xv, yv]].values, df_numeric_only[zv].values
            model = LinearRegression().fit(X_vals, Z_vals)
            st.metric("Model Confidence (R²)", f"{model.score(X_vals, Z_vals):.3f}")
            
            xr, yr = np.linspace(df_numeric_only[xv].min(), df_numeric_only[xv].max(), 20), np.linspace(df_numeric_only[yv].min(), df_numeric_only[yv].max(), 20)
            xx, yy = np.meshgrid(xr, yr)
            zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            fig = go.Figure(data=[
                go.Scatter3d(x=df_numeric_only[xv], y=df_numeric_only[yv], z=df_numeric_only[zv], mode='markers', 
                             marker=dict(size=4, color=df_numeric_only[zv], colorscale='Viridis')),
                go.Surface(x=xr, y=yr, z=zz, opacity=0.4, showscale=False)
            ])
            fig.update_layout(scene=dict(xaxis_title=xv, yaxis_title=yv, zaxis_title=zv), margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig, use_container_width=True)

    elif mode == "Correlation Matrix":
        st.subheader("Logistic Regression Accuracy Matrix")
        st.markdown("This matrix shows the **Accuracy** of predicting if the 'Row' variable is **High** (above median) using the 'Column' variable as a predictor.")
        
        subset_cols = st.multiselect("Select variables for matrix:", numeric_cols, default=numeric_cols[:15])
        
        if len(subset_cols) > 1:
            acc_matrix = pd.DataFrame(index=subset_cols, columns=subset_cols)

            progress_bar = st.progress(0)
            total_steps = len(subset_cols)
            
            for i, target in enumerate(subset_cols):
                y_median = df_numeric_only[target].median()
                y_bin = (df_numeric_only[target] > y_median).astype(int)
                
                for predictor in subset_cols:
                    if target == predictor:
                        acc_matrix.loc[target, predictor] = 1.0
                    else:
                        try:
                            X = df_numeric_only[[predictor]].values
                            model = LogisticRegression().fit(X, y_bin)
                            acc_matrix.loc[target, predictor] = model.score(X, y_bin)
                        except ValueError:
                            acc_matrix.loc[target, predictor] = np.nan
                
                progress_bar.progress((i + 1) / total_steps)
            
            progress_bar.empty()
            
            acc_matrix = acc_matrix.astype(float)

            fig = px.imshow(
                acc_matrix,
                text_auto='.2f',
                color_continuous_scale='YlGnBu',
                aspect="auto",
                labels=dict(color="Accuracy"),
                zmin=0.5, zmax=1.0
            )
            
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least two variables.")
    elif mode == "Categorical Analysis (Bar Charts)":
        if auto_cat_cols:
            c1, c2 = st.columns(2)
            cat_v = c1.selectbox("Survey Category", auto_cat_cols)
            num_v = c2.selectbox("Numeric Metric", numeric_cols)
            
            df_exploded = df_filtered.copy().dropna(subset=[cat_v])
            
            df_exploded[cat_v] = df_exploded[cat_v].astype(str).str.split(r',\s*')
            df_exploded = df_exploded.explode(cat_v)
            
            df_exploded[cat_v] = df_exploded[cat_v].str.strip()
            
            df_exploded = df_exploded[~df_exploded[cat_v].isin(['nan', 'NaN', ''])]

            df_grp = df_exploded.groupby(cat_v)[num_v].mean().reset_index().sort_values(num_v, ascending=False)
            
            if not df_grp.empty:
                fig_bar = px.bar(df_grp, x=cat_v, y=num_v, color=num_v, text_auto='.2f')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No valid survey responses found for this category.")
    elif mode == "Logistic Regression (Binary Class)":
        st.header("2D Logistic Regression: Probability Curve")
        
        col1, col2 = st.columns(2)
        target_bin = col1.selectbox("Target (to be binarized)", numeric_cols, index=0)
        predictor = col2.selectbox("Predictor (X)", numeric_cols, index=1)

        median_val = df_numeric_only[target_bin].median()
        y_binary = (df_numeric_only[target_bin] > median_val).astype(int)
        
        X = df_numeric_only[[predictor]].values
        
        log_model = LogisticRegression().fit(X, y_binary)
        
        x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        y_probs = log_model.predict_proba(x_range)[:, 1]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_range.flatten(), 
            y=y_probs, 
            name="Probability Curve",
            line=dict(color='#2E8B57', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=df_numeric_only[predictor], 
            y=y_binary, 
            mode='markers', 
            name="Observations",
            marker=dict(color=y_binary, colorscale='Viridis', opacity=0.5)
        ))

        fig.update_layout(
            title=f"Probability of High {target_bin} vs {predictor}",
            xaxis_title=predictor,
            yaxis_title=f"Probability of {target_bin} > {median_val:.2f}",
            yaxis=dict(range=[-0.05, 1.05]),
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        acc = log_model.score(X, y_binary)
        st.metric("Model Accuracy", f"{acc:.2%}")

except Exception as e:
    st.error(f"Dashboard Error: {e}")