import { useSyncExternalStore } from 'react'
import { getKernelStatus, subscribeKernelStatus } from './pyodideClient'

export function useKernelStatus() {
  return useSyncExternalStore(subscribeKernelStatus, getKernelStatus)
}
