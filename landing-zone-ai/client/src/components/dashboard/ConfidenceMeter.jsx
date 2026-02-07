import React from 'react'

const ConfidenceMeter = ({ score }) => {
  // score is 0-1
  const percentage = Math.round(score * 100);
  const getColor = (p) => {
    if (p > 80) return 'text-green-600';
    if (p > 50) return 'text-yellow-600';
    return 'text-red-600';
  }

  return (
    <div className="bg-white p-4 rounded-lg shadow-sm">
      <h4 className="text-sm text-gray-500 uppercase font-bold tracking-wider mb-2">Confidence</h4>
      <div className={`text-3xl font-bold ${getColor(percentage)}`}>
        {percentage}%
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
        <div 
          className="bg-blue-600 h-2.5 rounded-full" 
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  )
}

export default ConfidenceMeter
