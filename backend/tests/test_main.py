import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_detect_endpoint(client):
    """Test the plate detection endpoint"""
    # Create a test image file
    with open("test_image.jpg", "rb") as f:
        response = client.post(
            "/api/detect",
            files={"file": ("test_image.jpg", f, "image/jpeg")}
        )
    
    assert response.status_code == 200
    assert "plate_number" in response.json()
    assert "image_path" in response.json()

def test_history_endpoint(client):
    """Test the history endpoint"""
    response = client.get("/api/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
