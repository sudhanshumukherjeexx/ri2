import inspect
import json

REGISTRY = {}


def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn

    return deco


@register("ping")
def _ping(args):
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


@register("list_keys")
def _list_keys(args):
    return list_keys()


@register("preview")
def _preview(args):
    return preview(args["key"], args.get("n", 10))


async def dispatch(fn_name, args_json):
    try:
        args = json.loads(args_json) if args_json else {}
        if fn_name not in REGISTRY:
            raise ValueError(f"Unknown function: {fn_name}")
        result = REGISTRY[fn_name](args)
        # Handlers are plain functions by default; a few (e.g. xlsx loading,
        # which lazily micropip-installs openpyxl) are async and return a
        # coroutine instead -- await it if so.
        if inspect.isawaitable(result):
            result = await result
        return to_json({"ok": True, "result": result})
    except Exception as e:  # noqa: BLE001 - boundary must never raise into JS
        return to_json({"ok": False, "error": str(e)})
