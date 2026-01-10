"""
Central State Management for CRM AI Orchestrator
=================================================
Bu dosya projenin "anayasası"dır. Tüm ajanlar bu merkezi state üzerinden haberleşir.
LangGraph, bu TypedDict yapısını kullanarak stateful workflow yönetimi sağlar.

Squad'lar:
- Intelligence Squad: Market Research, SEO Analysis, Web Analysis, Audience Building
- Content Squad: Pipeline Management, Brandvoice Writer, SEO/GEO Optimizer, Social Media, Web Publishing
- Sales Ops Squad: Lead Research, Meeting Notes, CRM Updater, Email Copilot, Task Extractor
"""

from typing import TypedDict, Optional, Literal, Annotated
from datetime import datetime
from enum import Enum
import operator

# Import canonical WorkflowStatus from workflow.py (single source of truth)
from backend.app.schemas.workflow import WorkflowStatus


# ============================================================================
# ENUM DEFINITIONS (LangGraph-specific enums only)
# ============================================================================

class ApprovalType(str, Enum):
    """Onay gerektiren aksiyon türleri"""
    CRM_UPDATE = "crm_update"
    CONTENT_PUBLISH = "content_publish"
    EMAIL_SEND = "email_send"
    TASK_CREATE = "task_create"
    LEAD_ENRICHMENT = "lead_enrichment"


class CRMProvider(str, Enum):
    """Desteklenen CRM sistemleri"""
    HUBSPOT = "hubspot"
    SALESFORCE = "salesforce"


# ============================================================================
# NESTED STATE STRUCTURES
# ============================================================================

class ClientContext(TypedDict):
    """Müşteri bağlamı - Her workflow için temel kimlik bilgileri"""
    client_id: str
    client_name: str
    crm_provider: CRMProvider
    crm_access_token: Optional[str]
    crm_refresh_token: Optional[str]
    brandvoice_doc_id: Optional[str]  # Supabase Storage'daki doküman ID'si
    industry: Optional[str]
    target_audience: Optional[str]


class SEOData(TypedDict):
    """Intelligence Squad - SEO analiz sonuçları"""
    keywords: list[dict]  # [{"keyword": str, "volume": int, "difficulty": float}]
    competitors: list[dict]  # [{"domain": str, "traffic": int, "top_keywords": list}]
    content_gaps: list[str]
    serp_analysis: list[dict]  # [{"query": str, "top_results": list, "featured_snippets": list}]
    youtube_insights: list[dict]  # [{"video_id": str, "title": str, "views": int, "keywords": list}]
    last_updated: Optional[str]


class MarketResearch(TypedDict):
    """Intelligence Squad - Pazar araştırması verileri"""
    industry_trends: list[dict]
    market_size: Optional[dict]
    key_players: list[dict]
    opportunities: list[str]
    threats: list[str]
    news_articles: list[dict]  # [{"title": str, "url": str, "summary": str, "date": str}]


class AudienceData(TypedDict):
    """Intelligence Squad - Hedef kitle analizi"""
    personas: list[dict]  # [{"name": str, "demographics": dict, "pain_points": list, "goals": list}]
    segments: list[dict]
    engagement_patterns: dict
    preferred_channels: list[str]


class ContentDraft(TypedDict):
    """Content Squad - İçerik taslağı"""
    draft_id: str
    content_type: Literal["blog", "social", "email", "landing_page", "video_script"]
    title: str
    body: str
    meta_description: Optional[str]
    target_keywords: list[str]
    status: Literal["draft", "review", "approved", "published"]
    platform: Optional[str]  # "linkedin", "twitter", "blog", etc.
    scheduled_date: Optional[str]
    created_at: str
    updated_at: str


class ContentPipeline(TypedDict):
    """Content Squad - Redaksiyon takvimi ve pipeline yönetimi"""
    calendar: list[dict]  # [{"date": str, "content_type": str, "topic": str, "status": str}]
    active_drafts: list[ContentDraft]
    published_content: list[dict]
    performance_metrics: dict  # {"views": int, "engagement": float, "conversions": int}


class MeetingNote(TypedDict):
    """Sales Ops Squad - Toplantı notu analizi"""
    meeting_id: str
    date: str
    participants: list[str]
    summary: str
    key_points: list[str]
    action_items: list[dict]  # [{"task": str, "assignee": str, "due_date": str}]
    follow_up_required: bool
    sentiment: Literal["positive", "neutral", "negative"]
    deal_stage_update: Optional[str]


class CRMTask(TypedDict):
    """Sales Ops Squad - CRM'e işlenecek görev"""
    task_id: str
    task_type: Literal["create_contact", "update_deal", "create_task", "send_email", "add_note"]
    entity_type: Literal["contact", "company", "deal", "task", "note"]
    entity_id: Optional[str]
    payload: dict  # CRM API'ye gönderilecek veri
    priority: Literal["low", "medium", "high", "urgent"]
    status: Literal["pending", "approved", "executed", "failed"]
    error_message: Optional[str]
    created_at: str
    executed_at: Optional[str]


class LeadData(TypedDict):
    """Sales Ops Squad - Lead araştırma sonuçları"""
    lead_id: str
    company_name: str
    contact_name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    linkedin_url: Optional[str]
    company_size: Optional[str]
    industry: Optional[str]
    enrichment_data: dict  # Ek bilgiler (teknoloji stack, funding, vb.)
    lead_score: Optional[float]
    source: str


class EmailDraft(TypedDict):
    """Sales Ops Squad - Email taslağı"""
    email_id: str
    recipient_id: str
    recipient_email: str
    subject: str
    body: str
    email_type: Literal["cold_outreach", "follow_up", "nurture", "proposal"]
    personalization_data: dict
    status: Literal["draft", "approved", "sent", "bounced", "replied"]
    scheduled_at: Optional[str]
    sent_at: Optional[str]


class ApprovalRequest(TypedDict):
    """Human-in-the-Loop onay isteği"""
    approval_id: str
    approval_type: ApprovalType
    title: str
    description: str
    payload: dict  # Onaylanacak veri
    requested_at: str
    requested_by: str  # Agent ID
    status: Literal["pending", "approved", "rejected"]
    reviewed_at: Optional[str]
    reviewed_by: Optional[str]
    rejection_reason: Optional[str]


class AgentMessage(TypedDict):
    """Ajanlar arası iletişim mesajı"""
    message_id: str
    from_agent: str
    to_agent: Optional[str]  # None = broadcast
    message_type: Literal["info", "request", "response", "error", "handoff"]
    content: str
    metadata: dict
    timestamp: str


# ============================================================================
# MAIN ORCHESTRATOR STATE
# ============================================================================

class OrchestratorState(TypedDict):
    """
    Ana Orkestratör State - LangGraph Workflow'un merkezi hafızası

    Bu state, tüm ajanlar tarafından okunur ve güncellenir.
    LangGraph'ın reducer fonksiyonları ile thread-safe güncelleme sağlanır.
    """

    # --- WORKFLOW META ---
    workflow_id: str
    workflow_type: Literal[
        "full_cycle",           # Tüm squad'ları çalıştır
        "intelligence_only",    # Sadece araştırma
        "content_only",         # Sadece içerik üretimi
        "sales_ops_only",       # Sadece satış operasyonları
        "custom"                # Özel akış
    ]
    status: WorkflowStatus
    started_at: str
    updated_at: str
    completed_at: Optional[str]
    error_message: Optional[str]

    # --- CLIENT CONTEXT ---
    client: ClientContext

    # --- INTELLIGENCE SQUAD OUTPUTS ---
    seo_data: Optional[SEOData]
    market_research: Optional[MarketResearch]
    audience_data: Optional[AudienceData]
    web_analysis: Optional[dict]  # Genel web analiz sonuçları

    # --- CONTENT SQUAD OUTPUTS ---
    content_pipeline: Optional[ContentPipeline]
    content_drafts: Annotated[list[ContentDraft], operator.add]  # Append-only list

    # --- SALES OPS SQUAD OUTPUTS ---
    meeting_notes: Annotated[list[MeetingNote], operator.add]
    crm_tasks: Annotated[list[CRMTask], operator.add]
    leads: Annotated[list[LeadData], operator.add]
    email_drafts: Annotated[list[EmailDraft], operator.add]

    # --- HUMAN-IN-THE-LOOP ---
    pending_approvals: Annotated[list[ApprovalRequest], operator.add]
    approval_history: Annotated[list[ApprovalRequest], operator.add]

    # --- AGENT COMMUNICATION ---
    messages: Annotated[list[AgentMessage], operator.add]

    # --- RAG & KNOWLEDGE BASE ---
    retrieved_documents: list[dict]  # [{"doc_id": str, "content": str, "relevance": float}]
    brandvoice_context: Optional[str]  # RAG'dan çekilen marka sesi özeti

    # --- TRACING & OBSERVABILITY ---
    trace_id: Optional[str]  # LangSmith trace ID
    agent_execution_log: Annotated[list[dict], operator.add]  # [{"agent": str, "action": str, "timestamp": str}]


# ============================================================================
# STATE FACTORY & HELPERS
# ============================================================================

def create_initial_state(
    client_id: str,
    client_name: str,
    crm_provider: CRMProvider,
    workflow_type: str = "full_cycle"
) -> OrchestratorState:
    """
    Yeni bir workflow için başlangıç state'i oluşturur.

    Args:
        client_id: Müşteri benzersiz kimliği
        client_name: Müşteri adı
        crm_provider: CRM sistemi (hubspot/salesforce)
        workflow_type: Çalıştırılacak workflow tipi

    Returns:
        OrchestratorState: Başlatılmış state objesi
    """
    import uuid

    now = datetime.utcnow().isoformat()

    return OrchestratorState(
        # Workflow Meta
        workflow_id=str(uuid.uuid4()),
        workflow_type=workflow_type,
        status=WorkflowStatus.PENDING,
        started_at=now,
        updated_at=now,
        completed_at=None,
        error_message=None,

        # Client Context
        client=ClientContext(
            client_id=client_id,
            client_name=client_name,
            crm_provider=crm_provider,
            crm_access_token=None,
            crm_refresh_token=None,
            brandvoice_doc_id=None,
            industry=None,
            target_audience=None,
        ),

        # Intelligence Squad
        seo_data=None,
        market_research=None,
        audience_data=None,
        web_analysis=None,

        # Content Squad
        content_pipeline=None,
        content_drafts=[],

        # Sales Ops Squad
        meeting_notes=[],
        crm_tasks=[],
        leads=[],
        email_drafts=[],

        # Human-in-the-Loop
        pending_approvals=[],
        approval_history=[],

        # Agent Communication
        messages=[],

        # RAG & Knowledge Base
        retrieved_documents=[],
        brandvoice_context=None,

        # Tracing
        trace_id=None,
        agent_execution_log=[],
    )


def get_pending_approvals_by_type(
    state: OrchestratorState,
    approval_type: ApprovalType
) -> list[ApprovalRequest]:
    """Belirli bir tip için bekleyen onayları döndürür."""
    return [
        approval for approval in state["pending_approvals"]
        if approval["approval_type"] == approval_type
        and approval["status"] == "pending"
    ]


def get_tasks_by_status(
    state: OrchestratorState,
    status: str
) -> list[CRMTask]:
    """Belirli bir durumdaki CRM görevlerini döndürür."""
    return [
        task for task in state["crm_tasks"]
        if task["status"] == status
    ]


def count_pending_items(state: OrchestratorState) -> dict:
    """Bekleyen öğelerin sayısını döndürür - Dashboard için."""
    return {
        "pending_approvals": len([a for a in state["pending_approvals"] if a["status"] == "pending"]),
        "pending_crm_tasks": len([t for t in state["crm_tasks"] if t["status"] == "pending"]),
        "draft_content": len([c for c in state["content_drafts"] if c["status"] == "draft"]),
        "unsent_emails": len([e for e in state["email_drafts"] if e["status"] == "draft"]),
    }
