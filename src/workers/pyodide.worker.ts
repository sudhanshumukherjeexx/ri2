// @ts-nocheck -- worker runtime script; Pyodide's dynamic CDN import and the
// WebWorker global scope don't mix cleanly with the app's DOM-flavored tsconfig.

import storeSrc from './python/store.py?raw'
import dispatchSrc from './python/dispatch.py?raw'
import loadDataSrc from './python/load_data.py?raw'
import overviewSrc from './python/overview.py?raw'
import cleanupSrc from './python/cleanup.py?raw'
import imputeSrc from './python/impute.py?raw'
import encodingSrc from './python/encoding.py?raw'
import scalingSrc from './python/scaling.py?raw'
import edaSrc from './python/eda.py?raw'
import diagnosticsSrc from './python/diagnostics.py?raw'
import statsTestsSrc from './python/stats_tests.py?raw'

const PYODIDE_VERSION = '314.0.2'
const CDN_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

const PYTHON_MODULES = [
  storeSrc,
  dispatchSrc,
  loadDataSrc,
  overviewSrc,
  cleanupSrc,
  imputeSrc,
  encodingSrc,
  scalingSrc,
  edaSrc,
  diagnosticsSrc,
  statsTestsSrc,
]

let pyodideReadyPromise = null

async function initPyodide() {
  const { loadPyodide } = await import(/* @vite-ignore */ `${CDN_BASE}pyodide.mjs`)
  const pyodide = await loadPyodide({ indexURL: CDN_BASE })
  // openpyxl (needed only for .xlsx uploads) isn't in Pyodide's built-in
  // package repo -- it's installed lazily from PyPI via micropip the first
  // time an Excel file is actually loaded, so it never blocks CSV startup.
  await pyodide.loadPackage(['pandas', 'numpy', 'scipy', 'scikit-learn', 'micropip'])
  // plotly isn't in Pyodide's built-in package repo either, but every EDA
  // chart needs it, so install it eagerly here instead of lazily per click.
  const micropip = pyodide.pyimport('micropip')
  await micropip.install(['plotly'])
  pyodide.FS.mkdir('/uploads')
  for (const src of PYTHON_MODULES) {
    pyodide.runPython(src)
  }
  return pyodide
}

function getPyodide() {
  if (!pyodideReadyPromise) {
    pyodideReadyPromise = initPyodide()
  }
  return pyodideReadyPromise
}

self.onmessage = async (event) => {
  const { id, fn, args, buffer, path } = event.data
  try {
    const pyodide = await getPyodide()

    if (buffer) {
      pyodide.FS.writeFile(path, new Uint8Array(buffer))
    }

    const dispatch = pyodide.globals.get('dispatch')
    // dispatch is an async def in Python; Pyodide wraps the returned
    // coroutine as a real JS Promise, so a plain await just works.
    const resultJson = await dispatch(fn, JSON.stringify(args ?? {}))
    const parsed = JSON.parse(resultJson)
    self.postMessage({ id, ...parsed })
  } catch (err) {
    self.postMessage({ id, ok: false, error: err?.message ?? String(err) })
  }
}
