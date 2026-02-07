import React from 'react'

const ImageViewer = ({ imageUrl }) => {
  if (!imageUrl) {
    return (
      <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center text-gray-400">
        No image selected
      </div>
    )
  }

  return (
    <div className="relative rounded-lg overflow-hidden shadow-md">
      <img src={imageUrl} alt="Analysis Target" className="w-full h-auto object-contain" />
    </div>
  )
}

export default ImageViewer
