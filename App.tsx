import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, AlertCircle, CheckCircle2, Image as ImageIcon, Loader2 } from 'lucide-react';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<{ prediction: string; confidence: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setPrediction(null);
      setError(null);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      setFile(droppedFile);
      setPreview(URL.createObjectURL(droppedFile));
      setPrediction(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData);
      setPrediction(response.data);
    } catch (err) {
      setError('Error analyzing image. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">Deepfake Detection</h1>

        <div className="bg-gray-800 rounded-lg p-8 shadow-xl">
          <div
            className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              className="hidden"
            />

            {preview ? (
              <div className="space-y-4">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-96 mx-auto rounded-lg"
                />
                <p className="text-gray-400">Click or drag to change image</p>
              </div>
            ) : (
              <div className="space-y-4">
                <ImageIcon className="w-16 h-16 mx-auto text-gray-500" />
                <p className="text-xl">Drop an image here or click to upload</p>
                <p className="text-gray-400">Supports: JPG, PNG, GIF</p>
              </div>
            )}
          </div>

          {file && (
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Upload className="w-5 h-5" />
                  <span>Analyze Image</span>
                </>
              )}
            </button>
          )}

          {error && (
            <div className="mt-6 bg-red-900/50 text-red-200 p-4 rounded-lg flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <p>{error}</p>
            </div>
          )}

          {prediction && (
            <div className="mt-6 bg-gray-700/50 p-6 rounded-lg">
              <div className="flex items-center space-x-3 mb-4">
                {prediction.prediction === 'real' ? (
                  <CheckCircle2 className="w-6 h-6 text-green-500" />
                ) : (
                  <AlertCircle className="w-6 h-6 text-red-500" />
                )}
                <h3 className="text-xl font-semibold">
                  {prediction.prediction === 'real' ? 'Real Image' : 'Fake Image'}
                </h3>
              </div>
              <div className="bg-gray-600/50 rounded-lg p-4">
                <div className="flex justify-between items-center">
                  <span>Confidence:</span>
                  <span className="font-semibold">{prediction.confidence}%</span>
                </div>
                <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      prediction.prediction === 'real'
                        ? 'bg-green-500'
                        : 'bg-red-500'
                    }`}
                    style={{ width: `${prediction.confidence}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;