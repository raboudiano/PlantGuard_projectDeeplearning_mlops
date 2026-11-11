from backend.app.services.model_service import ModelService


def test_model_service_reports_metadata():
    service = ModelService()
    info = service.get_model_info()
    assert "model_name" in info
    assert "version" in info
    assert "training_status" in info
