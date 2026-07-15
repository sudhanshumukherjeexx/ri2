# Archive: original Streamlit R.I.D.E (no longer maintained)

This is the original Streamlit version of R.I.D.E, kept for reference. It has
been replaced by the static, Pyodide-powered app now at the repo root — see
the root [`README.md`](../README.md#history-why-we-moved-off-streamlit) for
why.

This folder is a snapshot, not a maintained app: nothing here is tested or
updated going forward.

- `Hello.py` — Streamlit entry point (`streamlit run Hello.py`)
- `pages/`, `utilities/` — the app's panels and backend logic
- `docs/` — the old mkdocs user manual (one page per panel)
- `Dockerfile`, `deployment.yaml`, `service.yaml`, `hpa.yaml`, `pv.yaml`,
  `pvc.yaml` — the Kubernetes deployment this version required
- `requirements.txt` — Python dependencies
- `datasets/`, `images/`, `audio/`, `plot_images/`, `cache/` — assets used by
  the old app
