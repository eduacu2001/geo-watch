import pytest

from geowatch.config import spacetrack_credentials


@pytest.fixture(autouse=True)
def no_credentials_in_environment(monkeypatch):
    monkeypatch.delenv("SPACETRACK_IDENTITY", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)


def test_missing_env_file(tmp_path):
    with pytest.raises(RuntimeError, match="No .env file"):
        spacetrack_credentials(tmp_path / ".env")


def test_env_saved_as_txt_gets_a_hint(tmp_path):
    (tmp_path / ".env.txt").write_text("SPACETRACK_IDENTITY=a@b.com\n")
    with pytest.raises(RuntimeError, match="rename it"):
        spacetrack_credentials(tmp_path / ".env")


def test_example_values_not_replaced(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "SPACETRACK_IDENTITY=your_email@example.com\nSPACETRACK_PASSWORD=your_password\n"
    )
    with pytest.raises(RuntimeError, match="example values"):
        spacetrack_credentials(env)


def test_valid_credentials(tmp_path):
    env = tmp_path / ".env"
    env.write_text('SPACETRACK_IDENTITY=me@mail.com\nSPACETRACK_PASSWORD="pa ss#word"\n')
    assert spacetrack_credentials(env) == ("me@mail.com", "pa ss#word")
