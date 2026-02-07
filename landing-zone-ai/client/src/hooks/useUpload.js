import { useState } from 'react';
import { uploadImage } from '../services/analysisService';

const useUpload = () => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async (file) => {
    setUploading(true);
    setError(null);
    try {
      const result = await uploadImage(file);
      return result;
    } catch (err) {
      setError(err.message || 'Upload failed');
      throw err;
    } finally {
      setUploading(false);
    }
  };

  return { handleUpload, uploading, error };
};

export default useUpload;
