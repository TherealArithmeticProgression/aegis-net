import React from 'react'

const AnalysisStats = ({ stats }) => {
  return (
    <div className="bg-white p-4 rounded-lg shadow-sm">
      <h4 className="text-sm text-gray-500 uppercase font-bold tracking-wider mb-4">Detailed Stats</h4>
      <div className="space-y-3">
        {Object.entries(stats).map(([key, value]) => (
          <div key={key} className="flex justify-between border-b pb-2 last:border-0 layer-gray-100">
            <span className="text-gray-600 capitalize">{key.replace(/_/g, ' ')}</span>
            <span className="font-medium text-gray-900">{value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default AnalysisStats
