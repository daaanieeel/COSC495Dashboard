import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations

st.set_page_config(page_title="Litter Survey Dashboard", layout="wide")

if 'top_2d_global' not in st.session_state:
    st.session_state.top_2d_global = None
if 'top_3d_global' not in st.session_state:
    st.session_state.top_3d_global = None
if 'target_results' not in st.session_state:
    st.session_state.target_results = None

st.title("Litter Survey Dashboard")

EXCLUDE_LIST = ['livability','head_placed','farm_no','house_no', 'goal_weight']

try:
    df_raw = pd.read_csv('data/processed/Final_Cleaned_Litter_Survey.csv')
    df_filtered = df_raw.drop(columns=EXCLUDE_LIST, errors='ignore')

    df_numeric_only = df_filtered.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna()
    df_numeric_only = df_numeric_only.astype(np.float64)
    numeric_cols = df_numeric_only.columns.tolist()

    auto_cat_cols = [col for col in df_filtered.select_dtypes(include=['object', 'category']).columns 
                    if df_filtered[col].nunique() < 15]

    mode = st.sidebar.selectbox("Analysis Mode", [
        "1 Variable (Distribution)", 
        "2 Variables (Correlation)", 
        "3 Variables (3D Plane)",
        "Categorical Analysis (Bar Charts)"
    ])

    st.sidebar.markdown("---")
    st.sidebar.header("Target-Specific Discovery")
    target_var = st.sidebar.selectbox("Pick a Target Variable", numeric_cols)

    if st.sidebar.button(f"Analyze for {target_var}"):
        r_list = []
        for col in numeric_cols:
            if col != target_var:
                r = df_numeric_only[target_var].corr(df_numeric_only[col])
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
            '2d': pd.DataFrame(r_list).sort_values(by='abs_r', ascending=False).head(5),
            '3d': pd.DataFrame(trio_list).sort_values(by='R²', ascending=False).head(5)
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
    
    if mode == "2 Variables (Correlation)":
        if st.sidebar.button("Find Top 10 2D Links"):
            pairs = list(combinations(numeric_cols, 2))
            results = []
            for p1, p2 in pairs:
                r = df_numeric_only[p1].corr(df_numeric_only[p2])
                if not np.isnan(r):
                    results.append({'A': p1, 'B': p2, 'r': round(r, 3), 'abs_r': abs(r)})
            st.session_state.top_2d_global = pd.DataFrame(results).sort_values(by='abs_r', ascending=False).head(10)

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

    elif mode == "Categorical Analysis (Bar Charts)":
        if auto_cat_cols:
            c1, c2 = st.columns(2)
            cat_v = c1.selectbox("Survey Category", auto_cat_cols)
            num_v = c2.selectbox("Numeric Metric", numeric_cols)
            df_grp = df_filtered.groupby(cat_v)[num_v].mean().reset_index().sort_values(num_v, ascending=False)
            st.plotly_chart(px.bar(df_grp, x=cat_v, y=num_v, color=num_v, text_auto='.2f'), use_container_width=True)

except Exception as e:
    st.error(f"Dashboard Error: {e}")