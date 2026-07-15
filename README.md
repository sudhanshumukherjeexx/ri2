# RI2 (Rapid Insights Data Engine)

<p align="center">
  <img src="logo/logo_2.png" alt="RI2 logo" width="480" />
</p>

_Analyze, visualize, and transform your data — no code, right in your browser._

RI2 is a no-code toolkit for cleaning, engineering, visualizing, and
statistically analyzing tabular data (CSV, XLSX, XLS). Upload a file or pick a
bundled sample and, without writing a line of code, you can:

- **Feature engineer** — fix column types, remove duplicates, impute missing
  values (8 strategies), encode categoricals, and scale/transform numeric
  features (8 methods)
- **Visualize** — 15+ chart types across basic, advanced, specialized, and
  geospatial categories
- **Analyze** — skewness/kurtosis, normality tests, IQR outlier detection, and
  6 hypothesis tests (t-test, ANOVA, chi-square, Mann-Whitney, Wilcoxon,
  Kruskal-Wallis)

Every one of those is real pandas/numpy/scipy/scikit-learn code — it just runs
as WebAssembly inside a Web Worker in your browser tab via
[Pyodide](https://pyodide.org/), instead of on a server. There is no backend:
your data is never uploaded anywhere, and the whole app deploys as static
files to GitHub Pages.

## Develop

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Deploy

Pushing to `main` builds and publishes to GitHub Pages automatically via
`.github/workflows/deploy.yml` (requires **Settings → Pages → Source →
GitHub Actions** to be enabled once).

## AI Insight

The "AI Insight" buttons call OpenAI through a stateless CORS-forwarding
proxy (see `proxy/README.md`) using the visitor's own API key, since
OpenAI's API can't be called directly from browser JS. The app works fully
without it.

## Architecture

- `src/workers/pyodide.worker.ts` — loads Pyodide + pandas/numpy/scipy/
  scikit-learn/plotly, execs the Python modules in `src/workers/python/`
  into a shared namespace.
- `src/workers/pyodideClient.ts` — `callPy(fn, args)` RPC bridge from React
  to the worker's Python `dispatch()` registry.
- `src/state/DataContext.tsx` — tracks loaded/derived dataset previews and
  per-page completion state.
- `src/pagesConfig.ts` — the pipeline order (also drives the sidebar,
  completion dots, and prev/next pager cards).

## History: why we moved off Streamlit

RI2 started as a [Streamlit](https://streamlit.io/) app — the original
version, preserved for reference in [`Archive/`](Archive/), is a full Python
Kubernetes deployment (`Archive/Dockerfile`, `Archive/deployment.yaml`,
`Archive/service.yaml`, `Archive/hpa.yaml`) with an AutoML page and two
LLM-chat pages that ran model-generated Python via `exec()`.

Streamlit needs a persistent server process for every visitor's session, so
that version could only run on paid, self-managed infrastructure — a real
operational cost for what's fundamentally meant to be a free, no-code data
tool anyone can open and use. It also meant carrying a large dependency
surface (xgboost, lightgbm, langchain) and a genuine security liability:
executing LLM-generated code server-side.

This rewrite drops the server entirely. Pyodide lets the exact same
computational core (pandas, numpy, scipy, scikit-learn) run inside the
visitor's own browser tab, so the whole app is just static files — free to
host, with no infrastructure to operate and no data ever leaving the tab.
AutoML and the chat pages were cut in the process: their heaviest
dependencies (xgboost, lightgbm, multiprocessing) have no WebAssembly
equivalent, and a keyless-by-default static app has no business running
arbitrary LLM-generated code server-side anyway. The "AI Insight" feature
that replaced them calls OpenAI directly from the browser with the visitor's
own key instead.
