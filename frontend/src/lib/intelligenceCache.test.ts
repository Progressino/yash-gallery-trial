import { describe, expect, it } from 'vitest'
import { bundleHasDisplayData, type IntelligenceBundle } from './intelligenceCache'

describe('bundleHasDisplayData', () => {
  it('unlocks UI for partial bundle with loaded platform units', () => {
    const bundle: IntelligenceBundle = {
      status: 'ready',
      data_completeness: 'partial',
      sales_summary: {
        total_units: 1200,
        total_returns: 50,
        net_units: 1150,
        return_rate: 4.2,
      },
      platform_summary: [
        {
          platform: 'Amazon',
          loaded: true,
          total_units: 1200,
          total_returns: 50,
          return_rate: 4.2,
          top_sku: 'A1',
          trend_direction: 'flat',
          monthly: [],
          by_state: [],
        },
      ],
      top_skus: [],
      anomalies: [],
      dsr_brand_monthly: { rows: [], totals: { YG: 0, Akiko: 0, Other: 0, Untagged: 0 }, note: '' },
    }
    expect(bundleHasDisplayData(bundle)).toBe(true)
  })

  it('blocks warming responses', () => {
    expect(
      bundleHasDisplayData({
        status: 'warming',
        sales_summary: { total_units: 0, total_returns: 0, net_units: 0, return_rate: 0 },
        platform_summary: [],
        top_skus: [],
        anomalies: [],
        dsr_brand_monthly: { rows: [], totals: { YG: 0, Akiko: 0, Other: 0, Untagged: 0 }, note: '' },
      }),
    ).toBe(false)
  })
})
