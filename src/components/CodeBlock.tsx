import { useEffect, useRef, useState } from 'react'
import Prism from 'prismjs'
import 'prismjs/components/prism-python'
import './CodeBlock.css'

type Props = {
  filename: string
  code: string
}

/** A dark terminal card showing the real pandas/scipy code Pyodide just ran
 * -- not decoration, proof this is genuine Python executing in the
 * browser, not a black box. */
export function CodeBlock({ filename, code }: Props) {
  const ref = useRef<HTMLElement>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (ref.current) Prism.highlightElement(ref.current)
  }, [code])

  async function copy() {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span className="code-dots" aria-hidden="true">
          <span className="code-dot code-dot-red" />
          <span className="code-dot code-dot-yellow" />
          <span className="code-dot code-dot-green" />
        </span>
        <span className="code-filename">{filename}</span>
        <button type="button" className="code-copy-btn" onClick={copy}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="code-block-pre">
        <code ref={ref} className="language-python">
          {code}
        </code>
      </pre>
    </div>
  )
}
