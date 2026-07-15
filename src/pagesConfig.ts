export type PageId =
  | 'overview'
  | 'cleanup'
  | 'impute'
  | 'encoding'
  | 'scaling'
  | 'eda'
  | 'diagnostics'
  | 'stat-tests'

export type PageConfig = {
  id: PageId
  slug: string
  label: string
  stage: string
  /** If set, this page's completion dot reflects whether this key exists in
   * DataContext's `previews` -- real derived-dataset state, not a manual
   * flag. Pages without one (EDA, Diagnostics, Stat Tests) are marked
   * complete manually via markComplete() when their action succeeds. */
  previewKey?: string
}

// The pipeline order -- also drives the homepage centerpiece and the
// prev/next pager cards at the foot of every page.
export const PAGES: PageConfig[] = [
  { id: 'overview', slug: 'overview', label: 'Overview', stage: 'Load', previewKey: 'df' },
  { id: 'cleanup', slug: 'cleanup', label: 'Cleanup & Conversion', stage: 'Clean', previewKey: 'df_cleaned' },
  { id: 'impute', slug: 'impute', label: 'Impute Missing Values', stage: 'Impute', previewKey: 'df_processed' },
  { id: 'encoding', slug: 'encoding', label: 'Feature Encoding', stage: 'Encode', previewKey: 'df_encoded' },
  { id: 'scaling', slug: 'scaling', label: 'Feature Scaling', stage: 'Scale', previewKey: 'df_scaled' },
  { id: 'eda', slug: 'eda', label: 'Exploratory Analysis', stage: 'Explore' },
  { id: 'diagnostics', slug: 'diagnostics', label: 'Distribution Diagnostics', stage: 'Diagnose' },
  { id: 'stat-tests', slug: 'stat-tests', label: 'Statistical Tests', stage: 'Test' },
]

export function pagePath(id: PageId) {
  const page = PAGES.find((p) => p.id === id)!
  return `/ch/${page.slug}`
}
