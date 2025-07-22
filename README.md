# Plate Recognition System

A full-stack application for license plate detection and management.

## Features

- 📸 Real-time webcam capture
- 🖼️ Image upload support
- 🔍 Automatic plate detection
- 📱 Responsive design with Tailwind CSS
- 🗄️ SQLite database for plate history
- 🔒 Token-based authentication

## Quick Start

```bash
# Start the application
docker compose up --build

# Access the applications:
Frontend: http://localhost:3000
Backend API: http://localhost:8000
```

## Project Structure

- `frontend/` - React application with Vite
- `backend/` - FastAPI server
- Both services are containerized with Docker

## Development

1. Frontend development server: `npm run dev`
2. Backend development server: `uvicorn app.main:app --reload`

## API Endpoints

- `POST /api/detect` - Upload and detect plate
- `GET /api/history` - Get detection history
- `POST /api/login` - User authentication
