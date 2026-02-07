const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');
const path = require('path');

exports.processImage = async (imagePath) => {
  try {
    const formData = new FormData();
    // Read the file from disk and append it
    formData.append('image', fs.createReadStream(imagePath));

    const pythonUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:5000';
    
    // Call Python Microservice
    const response = await axios.post(`${pythonUrl}/predict`, formData, {
      headers: {
        ...formData.getHeaders()
      }
    });

    // Return the data directly from Python (score, heatmapUrl, stats)
    return response.data;

  } catch (error) {
    if (error.response) {
         // The request was made and the server responded with a status code
         // that falls out of the range of 2xx
         throw new Error(`Python Service Error: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
    } else if (error.request) {
        // The request was made but no response was received
        throw new Error('Python Service did not respond. Is it running?');
    } else {
        throw new Error(`Python Service Connection Error: ${error.message}`);
    }
  }
};
