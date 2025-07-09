import React, { useState } from 'react';
import axios from 'axios';

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    await axios.post('http://localhost:8000/upload/', formData);
    alert('Uploaded successfully');
  };

  return (
    <div className="upload">
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      <button onClick={handleUpload}>Upload PDF</button>
    </div>
  );
}
