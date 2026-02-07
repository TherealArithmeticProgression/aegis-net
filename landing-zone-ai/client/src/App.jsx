import React from 'react'
import Navbar from './components/common/Navbar'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-900">Landing Zone AI</h1>
      </main>
    </div>
  )
}

export default App
