import React from "react";
import { Link } from "react-router-dom";

export default function Dashboard() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Link
        to="/upload"
        className="p-6 bg-white rounded-lg shadow hover:shadow-lg transition-shadow"
      >
        <h2 className="text-xl font-semibold mb-2">Upload Image</h2>
        <p className="text-gray-600">
          Upload an image to detect license plates
        </p>
      </Link>

      <Link
        to="/webcam"
        className="p-6 bg-white rounded-lg shadow hover:shadow-lg transition-shadow"
      >
        <h2 className="text-xl font-semibold mb-2">Webcam Capture</h2>
        <p className="text-gray-600">
          Use webcam for real-time plate detection
        </p>
      </Link>

      <Link
        to="/history"
        className="p-6 bg-white rounded-lg shadow hover:shadow-lg transition-shadow"
      >
        <h2 className="text-xl font-semibold mb-2">Detection History</h2>
        <p className="text-gray-600">View past detections and results</p>
      </Link>

      <div className="p-6 bg-white rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-2">Statistics</h2>
        <p className="text-gray-600">Total detections today: 0</p>
        <p className="text-gray-600">Success rate: 0%</p>
      </div>
    </div>
  );
}
