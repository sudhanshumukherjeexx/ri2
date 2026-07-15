import { useEffect, useState } from 'react'
import './TheStalk.css'

const AUTO_ADVANCE_MS = 2600

function prefersReducedMotion() {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

type FieldValue = string | number
type Record3 = { age: FieldValue; sex: FieldValue; fare: FieldValue }

type Stage = {
  id: string
  label: string
  record: Record3
  changed: (keyof Record3)[]
  note: string
}

const STAGES: Stage[] = [
  {
    id: 'raw',
    label: 'Raw',
    record: { age: 'NaN', sex: 'male', fare: 71.2833 },
    changed: [],
    note: 'A freshly loaded row: untyped, with a missing Age.',
  },
  {
    id: 'clean',
    label: 'Clean',
    record: { age: 'NaN', sex: 'male', fare: 71.28 },
    changed: ['fare'],
    note: "Cleanup fixes column types and truncates Fare's extra decimals.",
  },
  {
    id: 'impute',
    label: 'Impute',
    record: { age: 29.7, sex: 'male', fare: 71.28 },
    changed: ['age'],
    note: 'Impute fills the missing Age with the column mean.',
  },
  {
    id: 'encode',
    label: 'Encode',
    record: { age: 29.7, sex: 1, fare: 71.28 },
    changed: ['sex'],
    note: 'Encode turns Sex into a number a model can read.',
  },
  {
    id: 'scale',
    label: 'Scale',
    record: { age: 0.12, sex: 1, fare: 1.94 },
    changed: ['age', 'fare'],
    note: 'Scale standardizes Age and Fare onto comparable ranges.',
  },
  {
    id: 'insight',
    label: 'Insight',
    record: { age: 0.12, sex: 1, fare: 1.94 },
    changed: [],
    note: 'Ready for charts, hypothesis tests, and models -- still entirely in this tab.',
  },
]

export function TheStalk() {
  const [index, setIndex] = useState(0)
  const [autoPlay, setAutoPlay] = useState(true)
  const stage = STAGES[index]

  // Cycles the demo on its own so the mechanism is visible before anyone
  // clicks anything; stops for good the moment a visitor takes the wheel,
  // and never runs at all if the OS asked for reduced motion.
  useEffect(() => {
    if (!autoPlay || prefersReducedMotion()) return
    const id = setInterval(() => {
      setIndex((i) => (i + 1) % STAGES.length)
    }, AUTO_ADVANCE_MS)
    return () => clearInterval(id)
  }, [autoPlay])

  function selectStage(i: number) {
    setAutoPlay(false)
    setIndex(i)
  }

  return (
    <div className="stalk">
      <div className="stalk-track" style={{ gridTemplateColumns: `repeat(${STAGES.length}, 1fr)` }}>
        {STAGES.map((s, i) => (
          <button
            key={s.id}
            type="button"
            className={i === index ? 'stalk-node stalk-node-active' : 'stalk-node'}
            onClick={() => selectStage(i)}
            aria-current={i === index}
          >
            <span className="stalk-node-dot" aria-hidden="true" />
            <span className="stalk-node-label">{s.label}</span>
          </button>
        ))}
        <div
          className="stalk-token"
          style={{ left: `${((index + 0.5) / STAGES.length) * 100}%` }}
          aria-hidden="true"
        />
      </div>

      <div className="stalk-record">
        {(Object.keys(stage.record) as (keyof Record3)[]).map((field) => (
          <div
            key={field}
            className={
              stage.changed.includes(field) ? 'stalk-field stalk-field-changed' : 'stalk-field'
            }
          >
            <span className="stalk-field-name">{field}</span>
            <span className="stalk-field-value">{String(stage.record[field])}</span>
          </div>
        ))}
      </div>

      <p className="stalk-note">{stage.note}</p>
    </div>
  )
}
