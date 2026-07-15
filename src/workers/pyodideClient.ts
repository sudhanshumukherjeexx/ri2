type PendingEntry = {
  resolve: (value: any) => void
  reject: (reason: unknown) => void
}

export type KernelStatus = 'idle' | 'loading' | 'ready' | 'error'

let worker: Worker | null = null
let nextId = 1
const pending = new Map<number, PendingEntry>()

let kernelStatus: KernelStatus = 'idle'
const statusListeners = new Set<(s: KernelStatus) => void>()

function setKernelStatus(s: KernelStatus) {
  kernelStatus = s
  statusListeners.forEach((listener) => listener(s))
}

/** Subscribe to real Pyodide-worker boot state (used by the topbar's live
 * kernel-status dot -- this reflects actual worker/CDN load state, not a
 * decorative animation). */
export function subscribeKernelStatus(listener: (s: KernelStatus) => void) {
  statusListeners.add(listener)
  listener(kernelStatus)
  return () => {
    statusListeners.delete(listener)
  }
}

export function getKernelStatus() {
  return kernelStatus
}

function getWorker(): Worker {
  if (!worker) {
    setKernelStatus('loading')
    worker = new Worker(new URL('./pyodide.worker.ts', import.meta.url), {
      type: 'module',
    })
    worker.onmessage = (event: MessageEvent) => {
      const { id, ok, result, error } = event.data
      if (kernelStatus !== 'ready') setKernelStatus('ready')
      const entry = pending.get(id)
      if (!entry) return
      pending.delete(id)
      if (ok) entry.resolve(result)
      else entry.reject(new Error(error))
    }
    worker.onerror = (event: ErrorEvent) => {
      // A worker-level error (e.g. failed to load Pyodide from the CDN)
      // isn't tied to one request id -- reject everything still waiting.
      setKernelStatus('error')
      for (const [id, entry] of pending) {
        entry.reject(new Error(event.message))
        pending.delete(id)
      }
    }
  }
  return worker
}

/** Kick off the Pyodide worker before the user needs it, so the topbar's
 * status dot has something real to show right away. Safe to call multiple
 * times. */
export function warmupKernel() {
  callPy('ping', {}).catch(() => {})
}

/** Call a Python function registered in dispatch.py's REGISTRY. */
export function callPy<T = any>(fn: string, args: Record<string, any> = {}): Promise<T> {
  const w = getWorker()
  const id = nextId++
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    w.postMessage({ id, fn, args })
  })
}

/**
 * Write a File's bytes into Pyodide's virtual filesystem, then call a
 * Python function (typically load_csv) that reads from that path.
 */
export async function callPyWithFile<T = any>(
  fn: string,
  file: File,
  args: Record<string, any> = {},
): Promise<T> {
  const w = getWorker()
  const id = nextId++
  const buffer = await file.arrayBuffer()
  const path = `/uploads/${Date.now()}_${file.name}`
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    w.postMessage(
      { id, fn, args: { ...args, path, filename: file.name }, buffer, path },
      [buffer],
    )
  })
}
