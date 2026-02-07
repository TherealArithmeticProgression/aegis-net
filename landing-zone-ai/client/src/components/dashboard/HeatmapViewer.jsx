import React from 'react'

const HeatmapViewer = ({ heatmapUrl, opacity = 0.5 }) => {
  if (!heatmapUrl) return null;

  return (
    <div className="absolute inset-0 pointer-events-none" style={{ opacity }}>
      <img src={heatmapUrl} alt="Attention Map" className="w-full h-full object-contain" />
    </div>
  )
}

export default HeatmapViewer
