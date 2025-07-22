import React, { useRef } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const API_URL = "/api";

export default function WebcamPage() {
  const webcamRef = useRef(null);
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState(null);

  const capture = async () => {
    if (loading) return;

    setLoading(true);
    try {
      const imageSrc = webcamRef.current.getScreenshot();
      const blob = await (await fetch(imageSrc)).blob();
      const formData = new FormData();
      formData.append("file", blob, "capture.jpg");

      const response = await axios.post(`${API_URL}/detect`, formData);
      setResult(response.data);
    } catch (error) {
      console.error("Capture failed:", error);
      setResult({ error: "Failed to detect plate" });
    }
    setLoading(false);
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Live Webcam Capture</h1>

      <div className="space-y-4">
        <div className="relative">
          <Webcam
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            className="w-full rounded-lg shadow"
          />
          {loading && (
            <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
              <div className="text-white">Processing...</div>
            </div>
          )}
        </div>

        <button
          onClick={capture}
          disabled={loading}
          className={`w-full py-2 px-4 rounded-md ${
            loading
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-blue-500 hover:bg-blue-600 text-white"
          }`}
        >
          Capture & Detect
        </button>

        {result && (
          <div className="mt-4 p-4 bg-white rounded-lg shadow">
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
    </div>
  );
}
