import React, { useState } from "react";
import axios from "axios";

const API_URL = "/api";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(`${API_URL}/detect`, formData);
      setResult(response.data);
    } catch (error) {
      console.error("Upload failed:", error);
      setResult({ error: "Failed to detect plate" });
    }
    setLoading(false);
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Upload Image</h1>

      <form onSubmit={handleUpload} className="space-y-4">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer text-blue-500 hover:text-blue-600"
          >
            Click to upload an image
          </label>
        </div>

        {preview && (
          <div className="mt-4">
            <img
              src={preview}
              alt="Preview"
              className="max-h-96 mx-auto rounded-lg shadow"
            />
          </div>
        )}

        <button
          type="submit"
          disabled={!file || loading}
          className={`w-full py-2 px-4 rounded-md ${
            !file || loading
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-blue-500 hover:bg-blue-600 text-white"
          }`}
        >
          {loading ? "Processing..." : "Detect Plate"}
        </button>
      </form>

      {result && (
        <div className="mt-6 p-4 bg-white rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-2">Detection Result</h2>
          {result.error ? (
            <p className="text-red-500">{result.error}</p>
          ) : (
            <p className="text-green-600">
              Detected plate: {result.plate_number}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
