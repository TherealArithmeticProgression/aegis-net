import React, { useState } from 'react'
import UploadPanel from '../components/dashboard/UploadPanel'
import ImageViewer from '../components/dashboard/ImageViewer'
import ConfidenceMeter from '../components/dashboard/ConfidenceMeter'
import AnalysisStats from '../components/dashboard/AnalysisStats'

const Dashboard = () => {
  const [analysis, setAnalysis] = useState(null);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-6">
        <ImageViewer imageUrl={analysis?.imageUrl} />
        <UploadPanel />
      </div>
      <div className="space-y-6">
        {analysis && (
            <>
                <ConfidenceMeter score={analysis.score} />
                <AnalysisStats stats={analysis.stats} />
            </>
        )}
        {!analysis && (
            <div className="bg-white p-6 rounded-xl shadow-sm text-center text-gray-500">
                Upload an image to see analysis
            </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard
