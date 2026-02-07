import React from 'react'

const Navbar = () => {
  return (
    <nav className="bg-white shadow-md">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="text-xl font-bold text-blue-600">LandingZone AI</div>
          <div className="flex space-x-4">
            <a href="/" className="text-gray-600 hover:text-blue-600">Home</a>
            <a href="/dashboard" className="text-gray-600 hover:text-blue-600">Dashboard</a>
            <a href="/history" className="text-gray-600 hover:text-blue-600">History</a>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
