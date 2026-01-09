"""
Test Configuration and Fixtures
================================
Shared pytest fixtures for Phase 2 verification tests.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# Import app
from backend.app.main import app as fastapi_app


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# APP FIXTURES
# =============================================================================

@pytest.fixture
def app() -> FastAPI:
    """Get FastAPI application."""
    return fastapi_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Synchronous test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Async test client for async endpoint testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# =============================================================================
# CLIENT CONTEXT FIXTURES
# =============================================================================

@pytest.fixture
def dev_client_id() -> str:
    """Development client ID (non-UUID format)."""
    return "dev-client-001"


@pytest.fixture
def test_client_id() -> UUID:
    """Test client UUID."""
    return uuid4()


@pytest.fixture
def mock_client_context():
    """Mock ClientContext object."""
    from backend.app.core.dependencies import ClientContext
    return ClientContext(
        client_id="dev-client-001",
        client_name="Development Client",
        hubspot_token="test-token-123",
        salesforce_token=None,
    )


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers with client authentication."""
    return {"X-Client-ID": "dev-client-001"}


# =============================================================================
# DATABASE MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_supabase():
    """Mock SupabaseService for testing without real database."""
    mock = AsyncMock()

    # Default empty responses
    mock.fetch_one = AsyncMock(side_effect=Exception("Not found"))
    mock.fetch_many = AsyncMock(return_value=[])
    mock.insert = AsyncMock(return_value={"id": str(uuid4())})
    mock.update = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=True)
    mock.count = AsyncMock(return_value=0)

    return mock


@pytest.fixture
def patch_supabase(mock_supabase):
    """Patch get_supabase to return mock."""
    with patch(
        "backend.services.supabase_client.get_supabase",
        return_value=mock_supabase
    ) as patched:
        yield mock_supabase


# =============================================================================
# HUBSPOT MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_hubspot_contact() -> dict[str, Any]:
    """Sample HubSpot contact data."""
    return {
        "id": "123456",
        "properties": {
            "email": "test@example.com",
            "firstname": "John",
            "lastname": "Doe",
            "company": "Acme Corp",
            "phone": "+1234567890",
            "lifecyclestage": "lead",
        },
        "createdAt": "2024-01-15T10:00:00Z",
        "updatedAt": "2024-01-15T10:00:00Z",
    }


@pytest.fixture
def mock_hubspot_deal() -> dict[str, Any]:
    """Sample HubSpot deal data."""
    return {
        "id": "789012",
        "properties": {
            "dealname": "Enterprise Deal",
            "amount": "50000",
            "dealstage": "qualifiedtobuy",
            "closedate": "2024-06-30",
            "pipeline": "default",
        },
        "createdAt": "2024-01-15T10:00:00Z",
        "updatedAt": "2024-01-15T10:00:00Z",
    }


@pytest.fixture
def mock_hubspot_service():
    """Mock HubSpotService for testing."""
    mock = AsyncMock()

    # Contact operations
    mock.get_contact = AsyncMock(return_value=None)
    mock.list_contacts = AsyncMock(return_value={"results": [], "paging": None})
    mock.create_contact = AsyncMock(return_value={"id": "123456"})
    mock.update_contact = AsyncMock(return_value={"id": "123456"})

    # Deal operations
    mock.get_deal = AsyncMock(return_value=None)
    mock.list_deals = AsyncMock(return_value={"results": [], "paging": None})
    mock.create_deal = AsyncMock(return_value={"id": "789012"})
    mock.update_deal = AsyncMock(return_value={"id": "789012"})

    # Task operations
    mock.create_task = AsyncMock(return_value={"id": "task123"})

    # Note operations
    mock.create_note = AsyncMock(return_value={"id": "note123"})

    return mock


@pytest.fixture
def patch_hubspot(mock_hubspot_service):
    """Patch HubSpot factory to return mock service."""
    with patch(
        "backend.services.hubspot_service.get_hubspot_for_client",
        return_value=mock_hubspot_service
    ):
        yield mock_hubspot_service


# =============================================================================
# APPROVAL FIXTURES
# =============================================================================

@pytest.fixture
def sample_approval_data() -> dict[str, Any]:
    """Sample approval record data."""
    now = datetime.now(timezone.utc)
    return {
        "id": str(uuid4()),
        "client_id": str(uuid4()),
        "workflow_id": str(uuid4()),
        "approval_type": "crm_create_contact",
        "status": "pending",
        "priority": "normal",
        "title": "Create Contact: John Doe",
        "description": "Create new contact in HubSpot",
        "payload": {
            "operation": "create",
            "object_type": "contact",
            "properties": {
                "email": "john@example.com",
                "firstname": "John",
                "lastname": "Doe",
            }
        },
        "risk_level": "medium",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "expires_at": None,
    }


@pytest.fixture
def mock_approval_service(sample_approval_data):
    """Mock ApprovalService for testing."""
    from backend.app.schemas.approvals import (
        ApprovalListResponse,
        ApprovalSummary,
        ApprovalDetail,
        ApprovalActionResponse,
        ApprovalStats,
        ApprovalStatus,
        ApprovalType,
        ApprovalPriority,
        RiskLevel,
    )

    mock = AsyncMock()

    # Create summary from sample data
    summary = ApprovalSummary(
        id=UUID(sample_approval_data["id"]),
        workflow_id=UUID(sample_approval_data["workflow_id"]),
        approval_type=ApprovalType(sample_approval_data["approval_type"]),
        title=sample_approval_data["title"],
        description=sample_approval_data["description"],
        status=ApprovalStatus(sample_approval_data["status"]),
        priority=ApprovalPriority(sample_approval_data["priority"]),
        risk_level=RiskLevel(sample_approval_data["risk_level"]),
        created_at=datetime.fromisoformat(sample_approval_data["created_at"]),
        expires_at=None,
    )

    # List approvals
    mock.list_approvals = AsyncMock(return_value=ApprovalListResponse(
        items=[summary],
        total=1,
        pending_count=1,
        page=1,
        page_size=20,
    ))

    # Get stats
    mock.get_stats = AsyncMock(return_value=ApprovalStats(
        pending=1,
        approved_today=0,
        rejected_today=0,
        expired_today=0,
        by_type={"crm_create_contact": 1},
        by_priority={"normal": 1},
    ))

    return mock


# =============================================================================
# WORKFLOW FIXTURES
# =============================================================================

@pytest.fixture
def sample_workflow_data() -> dict[str, Any]:
    """Sample workflow record data."""
    now = datetime.now(timezone.utc)
    return {
        "id": str(uuid4()),
        "client_id": str(uuid4()),
        "workflow_type": "meeting_analysis",
        "status": "pending",
        "priority": "normal",
        "progress": 0,
        "input_data": {
            "transcript": "Meeting notes content...",
            "meeting_title": "Sales Call",
        },
        "result": None,
        "error": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "started_at": None,
        "completed_at": None,
    }


@pytest.fixture
def mock_workflow_service(sample_workflow_data):
    """Mock WorkflowService for testing."""
    from backend.app.schemas.workflow import (
        WorkflowResponse,
        WorkflowListResponse,
        WorkflowSummary,
        WorkflowStats,
        WorkflowType,
        WorkflowStatus,
        WorkflowPriority,
    )

    mock = AsyncMock()

    workflow_id = UUID(sample_workflow_data["id"])
    now = datetime.now(timezone.utc)

    # Trigger workflow
    mock.trigger_workflow = AsyncMock(return_value=WorkflowResponse(
        id=workflow_id,
        workflow_type=WorkflowType.MEETING_ANALYSIS,
        status=WorkflowStatus.PENDING,
        message="Workflow queued for execution",
        created_at=now,
        estimated_duration_seconds=30,
    ))

    # List workflows
    summary = WorkflowSummary(
        id=workflow_id,
        workflow_type=WorkflowType.MEETING_ANALYSIS,
        status=WorkflowStatus.PENDING,
        priority=WorkflowPriority.NORMAL,
        progress=0,
        created_at=now,
        updated_at=now,
        pending_approval_count=0,
    )
    mock.list_workflows = AsyncMock(return_value=WorkflowListResponse(
        items=[summary],
        total=1,
        page=1,
        page_size=20,
        has_more=False,
    ))

    # Get stats
    mock.get_stats = AsyncMock(return_value=WorkflowStats(
        total=1,
        pending=1,
        running=0,
        awaiting_approval=0,
        completed_today=0,
        failed_today=0,
        by_type={"meeting_analysis": 1},
        by_status={"pending": 1},
    ))

    return mock


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_test_uuid() -> UUID:
    """Generate a test UUID."""
    return uuid4()


def create_iso_timestamp() -> str:
    """Generate current ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()
