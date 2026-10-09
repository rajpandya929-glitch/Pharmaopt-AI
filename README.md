# PharmaOpt AI — Research Prototype 0.1

A browser-deployable prototype for pharmaceutical design of experiments (DoE), statistical analysis, and model-based optimization.

## What it does
- Generate 2-level full factorial designs.
- Generate Box–Behnken response-surface designs.
- Map coded design settings to actual low/high factor values.
- Upload CSV data or load sample data.
- Fit linear, two-factor-interaction, and quadratic OLS models.
- Display regression summary, ANOVA (where available), R², adjusted R², RMSE, and residual plots.
- Optimize a fitted response within user-specified bounds.
- Download designs and results as CSV.

## Deploy in a browser (no local Python installation required)
1. Create or sign into a GitHub account.
2. Create a new repository, e.g. `pharmaopt-ai`.
3. Upload `app.py`, `requirements.txt`, and this README.
4. Open https://share.streamlit.io/ and connect GitHub.
5. Choose **Create app**, select your repository, branch `main`, and entrypoint `app.py`, then deploy.
6. Wait for the build to finish and open the resulting app URL.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## First test
- Open **Analyze Data**.
- Click **Load example data**.
- Select `Release_8h_pct` as the response.
- Select `Polymer_pct`, `Binder_pct`, and `Compression_force_N` as factors.
- Choose **Two-factor interactions** and run the analysis.
- Go to **Optimize** and test maximize/minimize/target.

## Important scientific limitations
- Prototype only; not validated for regulatory or clinical decisions.
- Design generators are not randomized in this version.
- Box–Behnken is supported for at least 3 factors.
- A quadratic model requires enough independent observations; avoid overfitting.
- The optimization is only as trustworthy as the fitted model and its experimental domain.
- Confirm predicted optima experimentally.
- Keep a separate copy of original data and document any preprocessing.
- Do not upload confidential, unpublished, identifiable, or commercially sensitive data to a public app.
