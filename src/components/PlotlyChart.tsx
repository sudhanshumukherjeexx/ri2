import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'

const Plot = createPlotlyComponent(Plotly)

type Props = {
  data: Partial<Plotly.PlotData>[]
  layout?: Partial<Plotly.Layout>
  height?: number
}

export function PlotlyChart({ data, layout, height = 480 }: Props) {
  return (
    <Plot
      data={data}
      layout={{ autosize: true, margin: { t: 40 }, ...layout }}
      style={{ width: '100%', height: `${height}px` }}
      useResizeHandler
      config={{ responsive: true, displaylogo: false }}
    />
  )
}
