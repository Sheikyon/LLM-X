import pytest
from pathlib import Path
from llm_x.hub import hf_client


def test_token_lifecycle(tmp_path, monkeypatch):
    """
    End-to-end test of the Hugging Face token storage lifecycle:
    save → list → activate → load → delete
    """
    # Redirect token storage to a temp file so we don't touch real ~/.cache
    fake_token_file = tmp_path / ".test_tokens.json"
    monkeypatch.setattr(hf_client, "TOKEN_PATH", fake_token_file)

    # 1. Save a new dummy token
    result = hf_client.save_new_hf_token("hf_test_123")
    assert result == "saved", "Token was not saved successfully"

    # 2. Check it appears in the list
    tokens = hf_client.get_token_list()
    assert "hf_test_123" in tokens, "Saved token not found in token list"

    # 3. Set it as active (index 1 = first token since we only have one)
    hf_client.set_active_token(1)

    # Verify we can load the active token correctly
    loaded = hf_client.load_hf_token()
    assert loaded == "hf_test_123", "Loaded token does not match saved token"

    # 4. Nuke everything and make sure file is gone
    hf_client.delete_all_tokens()
    assert not fake_token_file.exists(), "Token file was not deleted"