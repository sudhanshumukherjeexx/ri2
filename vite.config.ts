import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served from https://sudhanshumukherjeexx.github.io/ri2/ as a GitHub Pages
// project site, so all asset URLs must be prefixed with the repo name.
export default defineConfig({
  base: '/ri2/',
  plugins: [react()],
  worker: {
    format: 'es',
  },
})
