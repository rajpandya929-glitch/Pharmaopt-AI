import io
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import differential_evolution
from pyDOE3 import ff2n, bbdesign

st.set_page_config(page_title="PharmaOpt AI", page_icon="🧪", layout="wide")
st.title("🧪 PharmaOpt AI")
st.caption("Pharmaceutical DoE & Statistical Optimization — Research Prototype 0.1")
st.warning(
    "Research prototype only. Validate designs and statistical results independently before using them in a thesis, publication, or laboratory decision. "
    "Optimization predictions are not experimentally confirmed results."
)

tab_design, tab_analyze, tab_optimize, tab_help = st.tabs(
    ["1 · Generate Design", "2 · Analyze Data", "3 · Optimize", "4 · Guide"]
)

def factor_names(n):
    return [f"Factor_{i+1}" for i in range(n)]

def scaled_to_actual(coded, lows, highs):
    return lows + (np.asarray(coded) + 1) * (highs-lows)/2

with tab_design:
    st.subheader("Generate an experimental design")
    dtype = st.selectbox("Design type", ["2-level full factorial", "Box–Behnken (quadratic RSM)"])
    nfac = st.number_input("Number of factors", min_value=2 if dtype.startswith("2-level") else 3,
                           max_value=8 if dtype.startswith("2-level") else 7,
                           value=3, step=1)
    names = []
    lows = []
    highs = []
    cols = st.columns(3)
    st.markdown("#### Define factor names and actual low/high settings")
    for i in range(int(nfac)):
        with st.container(border=True):
            c1,c2,c3 = st.columns(3)
            with c1:
                nm = st.text_input(f"Factor {i+1} name", value=["Polymer concentration","Binder concentration","Compression force","Factor 4","Factor 5","Factor 6","Factor 7","Factor 8"][i], key=f"dn{i}")
            with c2:
                lo = st.number_input(f"{nm or 'Factor'} low", value=float([10,2,50,1,1,1,1,1][i]), key=f"lo{i}")
            with c3:
                hi = st.number_input(f"{nm or 'Factor'} high", value=float([20,6,100,2,2,2,2,2][i]), key=f"hi{i}")
            names.append(nm.strip() or f"Factor_{i+1}")
            lows.append(lo); highs.append(hi)
    center_points = st.number_input("Additional center points (Box–Behnken only)", min_value=0, max_value=10, value=3,
                                    disabled=not dtype.startswith("Box"))
    if st.button("Generate design", type="primary", key="generate_design"):
        if len(set(names)) != len(names):
            st.error("Each factor must have a unique name.")
        elif any(h <= l for l,h in zip(lows,highs)):
            st.error("Every high setting must be greater than its low setting.")
        else:
            if dtype.startswith("2-level"):
                coded = ff2n(int(nfac))
                design_label = "2-level full factorial"
            else:
                coded = bbdesign(int(nfac), center=int(center_points))
                design_label = "Box–Behnken"
            actual = scaled_to_actual(coded, np.array(lows), np.array(highs))
            df = pd.DataFrame(actual, columns=names)
            df.insert(0, "Run", np.arange(1, len(df)+1))
            df.insert(1, "Coded_" + names[0], coded[:,0])
            st.session_state["generated_design"] = df
            st.session_state["design_label"] = design_label
            st.success(f"Generated {len(df)} runs for {design_label}.")
    if "generated_design" in st.session_state:
        st.dataframe(st.session_state["generated_design"], use_container_width=True)
        csv = st.session_state["generated_design"].to_csv(index=False).encode("utf-8")
        st.download_button("Download design as CSV", csv, "pharmaopt_design.csv", "text/csv")
        st.info("Run order is not randomized in this prototype. Randomize the order before conducting experiments, and plan replication/center points according to your study objective.")

with tab_analyze:
    st.subheader("Analyze your experimental data")
    st.write("Upload a CSV with one row per experiment, columns for numeric factors, and at least one numeric response.")
    uploaded = st.file_uploader("Upload experimental data (.csv)", type=["csv"], key="analysis_upload")
    use_example = st.button("Load example data")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            st.session_state["analysis_data"] = data
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    if use_example:
        rng = np.random.default_rng(12)
        x1 = np.array([-1,1,-1,1,-1,1,-1,1], dtype=float)
        x2 = np.array([-1,-1,1,1,-1,-1,1,1], dtype=float)
        x3 = np.array([-1,-1,-1,-1,1,1,1,1], dtype=float)
        y = 70 + 8*x1 - 4*x2 + 5*x3 + 3*x1*x2 - 2*x2*x3 + rng.normal(0,1.4,8)
        st.session_state["analysis_data"] = pd.DataFrame({
            "Polymer_pct": 15 + 5*x1, "Binder_pct": 4 + 2*x2,
            "Compression_force_N": 75 + 25*x3, "Release_8h_pct": y
        })
    if "analysis_data" in st.session_state:
        data = st.session_state["analysis_data"].copy()
        st.dataframe(data.head(20), use_container_width=True)
        numeric = list(data.select_dtypes(include=np.number).columns)
        if len(numeric) < 2:
            st.error("The CSV needs at least one numeric factor and one numeric response.")
        else:
            response = st.selectbox("Response variable", numeric, index=len(numeric)-1)
            factors = st.multiselect("Factor columns", [c for c in numeric if c != response],
                                     default=[c for c in numeric if c != response][:min(3,len(numeric)-1)])
            model_kind = st.selectbox("Model", ["Linear main effects", "Two-factor interactions", "Quadratic response surface"])
            if st.button("Run statistical analysis", type="primary"):
                if not factors:
                    st.error("Select at least one factor.")
                else:
                    clean = data[factors + [response]].dropna()
                    if len(clean) <= len(factors)+1:
                        st.error("Not enough complete observations for this model. Add data or choose a simpler model.")
                    else:
                        q = " + ".join([f'Q("{x}")' for x in factors])
                        if model_kind == "Two-factor interactions":
                            terms = [f'Q("{x}")' for x in factors]
                            terms += [f'Q("{a}")*Q("{b}")' for a,b in itertools.combinations(factors,2)]
                            formula = f'Q("{response}") ~ ' + " + ".join(terms)
                        elif model_kind == "Quadratic response surface":
                            terms = [f'Q("{x}")' for x in factors]
                            terms += [f'I(Q("{x}")**2)' for x in factors]
                            terms += [f'Q("{a}")*Q("{b}")' for a,b in itertools.combinations(factors,2)]
                            formula = f'Q("{response}") ~ ' + " + ".join(terms)
                        else:
                            formula = f'Q("{response}") ~ {q}'
                        try:
                            fit = smf.ols(formula, data=clean).fit()
                            st.session_state["fit_summary"] = fit.summary().as_text()
                            st.session_state["fit_params"] = {"factors":factors, "response":response, "model":model_kind}
                            st.session_state["fit_formula"] = formula
                            st.session_state["fit_data"] = clean
                            st.session_state["fit_model"] = fit
                            st.success("Model fitted. Review diagnostics carefully; a successful fit does not prove the model is adequate.")
                        except Exception as e:
                            st.error(f"Model could not be fitted: {e}")
            if "fit_model" in st.session_state and st.session_state.get("fit_params",{}).get("response")==response:
                fit=st.session_state["fit_model"]
                st.markdown("#### Model summary")
                st.code(st.session_state["fit_summary"])
                metrics=st.columns(4)
                metrics[0].metric("R²", f"{fit.rsquared:.4f}")
                metrics[1].metric("Adjusted R²", f"{fit.rsquared_adj:.4f}")
                metrics[2].metric("RMSE", f"{np.sqrt(np.mean(fit.resid**2)):.4f}")
                metrics[3].metric("Observations", int(fit.nobs))
                try:
                    anova = sm.stats.anova_lm(fit, typ=2)
                    st.markdown("#### ANOVA (Type II)")
                    st.dataframe(anova, use_container_width=True)
                except Exception as e:
                    st.info(f"ANOVA table unavailable for this fitted model: {e}")
                pred = fit.fittedvalues
                resid = fit.resid
                diag = pd.DataFrame({"Predicted":pred, "Residual":resid})
                fig = px.scatter(diag, x="Predicted", y="Residual", title="Residuals vs fitted values")
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("Download cleaned analysis data", st.session_state["fit_data"].to_csv(index=False).encode(), "clean_data.csv", "text/csv")

with tab_optimize:
    st.subheader("Optimize a fitted response model")
    st.write("First run an analysis in the Analyze Data tab. Optimization uses that fitted model and searches within the ranges you specify.")
    if "fit_model" not in st.session_state:
        st.info("No fitted model yet. Load example data or upload your own data in Analyze Data.")
    else:
        fit = st.session_state["fit_model"]
        params = st.session_state["fit_params"]
        factors = params["factors"]
        response = params["response"]
        st.caption(f"Current response: {response} · Model: {params['model']}")
        goal = st.selectbox("Optimization goal", ["Maximize response", "Minimize response", "Target response"])
        target = st.number_input("Target response", value=80.0, disabled=goal!="Target response")
        ranges = {}
        st.markdown("#### Search bounds")
        for f in factors:
            col1,col2=st.columns(2)
            vals = st.session_state["fit_data"][f]
            with col1: lo=st.number_input(f"{f} minimum", value=float(vals.min()), key=f"optlo_{f}")
            with col2: hi=st.number_input(f"{f} maximum", value=float(vals.max()), key=f"opthi_{f}")
            ranges[f]=(lo,hi)
        if st.button("Find predicted optimum", type="primary"):
            if any(hi<=lo for lo,hi in ranges.values()):
                st.error("Each maximum must exceed its minimum.")
            else:
                bounds=[ranges[f] for f in factors]
                def predict_one(x):
                    row=pd.DataFrame([dict(zip(factors,x))])
                    return float(fit.predict(row).iloc[0])
                def objective(x):
                    y=predict_one(x)
                    return -y if goal=="Maximize response" else y if goal=="Minimize response" else abs(y-target)
                result=differential_evolution(objective,bounds,seed=7,polish=True)
                optimum=dict(zip(factors,result.x))
                predicted=predict_one(result.x)
                st.success("Predicted optimum calculated. This is a model-based estimate, not a confirmed experimental result.")
                c=st.columns(len(factors)+1)
                for i,(k,v) in enumerate(optimum.items()): c[i].metric(k, f"{v:.4g}")
                c[-1].metric("Predicted response", f"{predicted:.4g}")
                st.download_button("Download optimum settings", pd.DataFrame([optimum | {f"Predicted_{response}":predicted}]).to_csv(index=False).encode(), "predicted_optimum.csv", "text/csv")
                st.warning("Perform a confirmation experiment at the predicted settings and compare observed vs predicted values. Do not extrapolate beyond the design space.")

with tab_help:
    st.subheader("Quick guide")
    st.markdown("""
**Generate Design**
1. Choose full factorial or Box–Behnken.
2. Set factor names and actual low/high ranges.
3. Generate and download the design.
4. Randomize run order before lab work and decide on replicates/center points with your supervisor.

**Analyze Data**
1. Upload a CSV or click “Load example data”.
2. Select a response and factor columns.
3. Select a model appropriate to the scientific objective and available runs.
4. Review the regression output, ANOVA, R², adjusted R², RMSE, and residual plot.

**Optimize**
1. Fit a model first.
2. Choose maximize, minimize, or target.
3. Set bounds within the experimental domain.
4. Find predicted settings, then run a confirmation experiment.

**CSV format**
- One row per experimental run.
- One column per numeric factor.
- One numeric column per response.
- Use consistent units; avoid merged cells and text in numeric columns.
- Keep a copy of the original raw data.

**Current limitations**
- No user accounts or permanent cloud storage.
- No automated randomization in design generation.
- The analysis module offers basic OLS models; not every model is appropriate for every design.
- No formal regulatory validation or audit trail.
- No AI-generated scientific conclusions in this prototype.
""")
    st.markdown("Design algorithm references: [PyDOE3 factorial designs](https://pydoe3.readthedocs.io/en/latest/reference/factorial/) · [PyDOE3 response-surface designs](https://pydoe3.readthedocs.io/en/latest/reference/response_surface/)")
st.divider()
st.caption("PharmaOpt AI Research Prototype 0.1 · Built for learning and research exploration, not for validated regulatory use.")
