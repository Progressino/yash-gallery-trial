/** Shared X-axis props for long monthly shipment/return bar charts. */
export const monthlyChartXAxisProps = {
  dataKey: 'Month' as const,
  tick: { fontSize: 10 },
  angle: -40,
  textAnchor: 'end' as const,
  height: 56,
  interval: 'preserveStartEnd' as const,
}
