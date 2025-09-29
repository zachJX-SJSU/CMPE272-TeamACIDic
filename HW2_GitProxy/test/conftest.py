# conftest.py
import pytest
import aiosqlite
from unittest.mock import AsyncMock

@pytest.fixture
def mock_db_connection():
    """Mocks an aiosqlite connection for isolated database testing."""
    mock_conn = AsyncMock(spec=aiosqlite.Connection)
    mock_conn.commit = AsyncMock()
    mock_conn.execute = AsyncMock()
    
    # Mocking the context manager __aenter__ and __aexit__
    mock_conn.__aenter__.return_value = mock_conn
    mock_conn.__aexit__.return_value = None
    
    return mock_conn

@pytest.fixture
def mock_db_cursor(mock_db_connection):
    """Mocks an aiosqlite cursor for the /events endpoint."""
    mock_cursor = AsyncMock(spec=aiosqlite.Cursor)
    # Mock row_factory setting
    mock_db_connection.row_factory = None 
    
    # Mock the execute for the /events query
    mock_db_connection.execute.return_value.__aenter__.return_value = mock_cursor
    mock_db_connection.execute.return_value.__aenter__.return_value.fetchall = AsyncMock(return_value=[
        {'id': 'test_id_1', 'event': 'issues', 'action': 'opened', 'issue_number': 1, 'timestamp': 1678886400},
        {'id': 'test_id_2', 'event': 'ping', 'action': None, 'issue_number': None, 'timestamp': 1678886460},
    ])
    
    # Mock the Row object behavior (dict-like access)
    def row_factory_side_effect(row_cls):
        def row_factory_func(cursor, row):
            if row is None: return None
            # Simple dict for mock rows
            return {'id': row[0], 'event': row[1], 'action': row[2], 'issue_number': row[3], 'timestamp': row[4]}
        mock_db_connection.row_factory = row_factory_func
        return row_factory_func

    mock_db_connection.row_factory = aiosqlite.Row
    mock_cursor.fetchall.return_value = [
        type('MockRow', (dict,), {'keys': lambda: ['id', 'event', 'action', 'issue_number', 'timestamp']})({
            'id': 'test_id_1', 'event': 'issues', 'action': 'opened', 'issue_number': 1, 'timestamp': 1678886400
        }),
        type('MockRow', (dict,), {'keys': lambda: ['id', 'event', 'action', 'issue_number', 'timestamp']})({
            'id': 'test_id_2', 'event': 'ping', 'action': None, 'issue_number': None, 'timestamp': 1678886460
        })
    ]
    
    return mock_cursor

@pytest.fixture
def mock_aiosqlite_connect(mocker, mock_db_connection):
    """Mocks the aiosqlite.connect function to return our mock connection."""
    mock_connect = mocker.patch("main.aiosqlite.connect", autospec=True)
    # Mock the async context manager for aiosqlite.connect
    mock_connect.return_value.__aenter__.return_value = mock_db_connection
    mock_connect.return_value.__aexit__.return_value = None
    return mock_connect

@pytest.fixture
def mock_env(monkeypatch):
    """Sets necessary environment variables for the application."""
    monkeypatch.setenv("WEBHOOK_SECRET", "test_secret_key")
    # Must import 'app' AFTER setting the env vars to pick up the mocked values
    import main
    return main.app

@pytest.fixture
async def async_client(mock_env, mock_aiosqlite_connect):
    """Creates an async test client for the FastAPI app."""
    from httpx import AsyncClient
    # The 'mock_aiosqlite_connect' fixture ensures the init_db runs against a mock DB
    # during app startup, which is triggered by the AsyncClient's lifespan management.
    async with AsyncClient(app=mock_env, base_url="http://test") as client:
        yield client