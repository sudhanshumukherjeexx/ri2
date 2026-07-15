import pandas as pd

_openpyxl_ready = False


async def _ensure_openpyxl():
    global _openpyxl_ready
    if _openpyxl_ready:
        return
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        import micropip

        await micropip.install("openpyxl")
    _openpyxl_ready = True


def _looks_like_default_header(columns):
    """Detect pandas' 'Unnamed: 0' style fallback header, same signal the
    original loader used to catch header-less CSVs (single-letter A/B/C
    columns don't occur with pandas' own inference, but a fully-generic
    RangeIndex-like header does)."""
    return all(str(c).startswith("Unnamed:") for c in columns)


@register("load_csv")
async def _load_csv(args):
    key = args["key"]
    path = args["path"]
    filename = args.get("filename", path)

    if filename.lower().endswith((".xlsx", ".xls")):
        await _ensure_openpyxl()
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
        if _looks_like_default_header(df.columns):
            df = pd.read_csv(path, header=None)
            df.columns = [f"column_{i}" for i in range(df.shape[1])]

    set_df(key, df)
    return preview(key)


@register("load_sample")
def _load_sample(args):
    # Sample datasets are fetched by the browser and written to the same
    # virtual-FS path as an upload, so this just delegates to load_csv.
    return _load_csv(args)
