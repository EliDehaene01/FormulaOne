"""
Smoke test for the 429-retry logic in backend/data/openf1_client.py.

Mocks requests.get and time.sleep so this runs instantly and never touches
the real network — we're checking the retry/backoff branching, not OpenF1
itself.

Run with:
    python -m backend.tests.test_openf1_client
"""

from unittest.mock import MagicMock, patch

from backend.data import openf1_client as client


def _response(status_code: int, json_body=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = json_body
    if status_code == 429:
        resp.raise_for_status.side_effect = client.requests.HTTPError(f"{status_code} error", response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_retries_once_after_429_then_succeeds():
    responses = [_response(429), _response(200, json_body=[{"lap_number": 1}])]
    with patch("backend.data.openf1_client.requests.get", side_effect=responses) as mock_get, \
         patch("backend.data.openf1_client.time.sleep") as mock_sleep:
        result = client._get("laps", session_key=1)

    assert result == [{"lap_number": 1}]
    assert mock_get.call_count == 2, "should retry exactly once after the 429"
    assert mock_sleep.call_count == 1, "should back off before the retry"


def test_gives_up_after_max_retries():
    responses = [_response(429) for _ in range(client.MAX_RETRIES + 1)]
    with patch("backend.data.openf1_client.requests.get", side_effect=responses), \
         patch("backend.data.openf1_client.time.sleep"):
        try:
            client._get("laps", session_key=1)
            raised = False
        except client.requests.HTTPError:
            raised = True

    assert raised, "should eventually surface the 429 instead of retrying forever"


def test_retry_after_header_is_respected():
    responses = [_response(429, headers={"Retry-After": "3"}), _response(200, json_body=[])]
    with patch("backend.data.openf1_client.requests.get", side_effect=responses), \
         patch("backend.data.openf1_client.time.sleep") as mock_sleep:
        client._get("laps", session_key=1)

    mock_sleep.assert_called_once_with(3.0)


def test_404_returns_empty_list_without_retry():
    with patch("backend.data.openf1_client.requests.get", return_value=_response(404)) as mock_get, \
         patch("backend.data.openf1_client.time.sleep") as mock_sleep:
        result = client._get("laps", session_key=9900)

    assert result == [], "a 404 (no data for this session) should look like an empty result, not an error"
    assert mock_get.call_count == 1, "404 is not a rate limit — should not retry"
    mock_sleep.assert_not_called()


if __name__ == "__main__":
    test_retries_once_after_429_then_succeeds()
    test_gives_up_after_max_retries()
    test_retry_after_header_is_respected()
    test_404_returns_empty_list_without_retry()
    print("All openf1_client tests passed.")
