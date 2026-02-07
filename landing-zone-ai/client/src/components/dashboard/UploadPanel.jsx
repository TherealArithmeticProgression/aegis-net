import React from 'react'
import Button from '../common/Button'

const UploadPanel = ({ onUpload }) => {
  return (
    <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-100">
      <h3 className="text-lg font-semibold mb-4">Upload Image</h3>
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center bg-gray-50 hover:bg-gray-100 transition-colors">
        <p className="text-gray-500 mb-4">Drag and drop your image here or click to browse</p>
        <Button onClick={() => {}}>Select File</Button>
      </div>
    </div>
  )
}

export default UploadPanel
