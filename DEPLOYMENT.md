# Free Deployment Guide

Use Streamlit Community Cloud for the free hosted site.

## Files Required Online

The dashboard only needs:

- `streamlit_app.py`
- `requirements.txt`
- `runtime.txt`
- `.streamlit/config.toml`
- `src/`
- `models/`
- `plots/`
- `reports/`
- `README.md`

The raw and processed CSV files are not needed by the hosted app because the app loads trained model bundles from `models/`.

## Deploy On Streamlit Community Cloud

1. Create a GitHub repository, for example `FRP_PROJECT`.
2. Upload the deployment files to that repository.
3. Go to `https://share.streamlit.io`.
4. Sign in with GitHub.
5. Create a new app.
6. Select:
   - Repository: your `FRP_PROJECT` repo
   - Branch: `main`
   - Main file path: `streamlit_app.py`
7. Deploy.

After deployment, Streamlit gives a public URL ending in `.streamlit.app`.

## Local Run

```bash
cd /d C:\Users\RHINO\Desktop\FRP_PROJECT
python -m streamlit run streamlit_app.py
```
