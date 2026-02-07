import React from 'react'
import Button from '../components/common/Button'

const LandingPage = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] text-center">
      <h1 className="text-5xl font-bold text-gray-900 mb-6">
        Advanced Terrain Analysis
      </h1>
      <p className="text-xl text-gray-600 max-w-2xl mb-8">
        AI-powered landing zone detection and safety assessment for autonomous aerial vehicles.
      </p>
      <div className="flex gap-4">
        <Button className="text-lg px-8 py-3">Get Started</Button>
        <Button variant="outline" className="text-lg px-8 py-3">View Documentation</Button>
      </div>
    </div>
  )
}

export default LandingPage
