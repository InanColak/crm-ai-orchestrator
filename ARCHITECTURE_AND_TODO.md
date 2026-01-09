# CRM AI Orchestrator - Architecture & TODO Master Plan

> Bu dosya projenin tum mimari kararlarini ve implementation roadmap'ini icerir.
> Olusturulma: 2026-01-08 | Architect Agent tarafindan
> Son Guncelleme: 2026-01-09 | Phase 4.3 Email Copilot Agent tamamlandi

---

## Executive Summary

HubSpot ve Salesforce danismanlik musterilerimiz icin uctan uca bir "Autonomous Growth & Sales Engine" insa ediyoruz. Bu sistem, pazar arastirmasindan icerik uretimine, SEO analizinden CRM guncellemelerine kadar 16+ karmasik gorevi otonom olarak yurutecek.

### Tech Stack

| Katman | Teknoloji |
|--------|-----------|
| Orchestration | LangGraph (Stateful, cyclic workflows) |
| Backend | FastAPI (Python) - Asenkron, yuksek performans |
| Frontend | Next.js + Tailwind CSS + Lucide Icons |
| Database | Supabase (PostgreSQL + Vector DB + Storage) |
| Intelligence | Claude 3.5 Sonnet / GPT-4o |
| CRM | HubSpot Python SDK + Salesforce SDK |
| Search | Tavily API |
| Observability | LangSmith |

---

## System Architecture

### High-Level Architecture Diagram

```
                    +---------------------+
                    |     Frontend        |
                    |   (Next.js Dashboard)|
                    +----------+----------+
                               |
                               v
                    +----------+----------+
                    |     FastAPI         |
                    |   (REST API Layer)  |
                    +----------+----------+
                               |
                               v
                    +----------+----------+
                    |    LangGraph        |
                    |   (Orchestrator)    |
                    +----------+----------+
                               |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+-------+-------+      +-------+-------+      +-------+-------+
|  Intelligence |      |    Content    |      |   Sales Ops   |
|     Squad     |      |     Squad     |      |     Squad     |
+---------------+      +---------------+      +---------------+
| Market Res.   |      | Pipeline Mgr  |      | Lead Research |
| SEO Analysis  |      | Brandvoice    |      | Meeting Notes |
| Web Analysis  |      | SEO/GEO Opt.  |      | CRM Updater   |
| Audience      |      | Social Dist.  |      | Email Copilot |
+---------------+      | Publisher     |      | Task Extract  |
                       +---------------+      +---------------+
                               |
                               v
                    +----------+----------+
                    |   Human-in-Loop     |
                    |   (Approval System) |
                    +----------+----------+
                               |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+-------+-------+      +-------+-------+      +-------+-------+
|    HubSpot    |      |   Salesforce  |      |   Supabase    |
|     SDK       |      |      SDK      |      |   (Storage)   |
+---------------+      +---------------+      +---------------+
```

### Agent Squad Structure

**Implementation Note:** MVP'de ajanlar ayri dosyalar yerine `backend/graph/nodes.py` icinde node fonksiyonlari olarak implement edilmektedir. Bu yaklasim LangGraph'in native pattern'i ile uyumludur ve state yonetimini basitlestirir.

```
backend/graph/nodes.py           # TUM AGENT NODE'LARI BURADA
|
+-- Intelligence Squad Nodes
|   +-- market_research_node()   # Pazar arastirma
|   +-- seo_analysis_node()      # SEO analizi
|   +-- web_analysis_node()      # Web sitesi analizi
|   +-- audience_builder_node()  # Hedef kitle olusturma
|
+-- Content Squad Nodes
|   +-- pipeline_manager_node()  # Editoryal takvim
|   +-- brandvoice_writer_node() # Marka sesi ile icerik
|   +-- seo_optimizer_node()     # SEO/GEO optimizasyonu
|   +-- social_distributor_node()# Sosyal medya dagilimi
|   +-- web_publisher_node()     # Web yayinlama
|
+-- Sales Ops Squad Nodes        # MVP ONCELIKLI
    +-- meeting_notes_node()     # [x] Toplanti notlari analizi (Phase 3.1)
    +-- task_extractor_node()    # [x] Gorev cikarma (Phase 3.2)
    +-- crm_updater_node()       # [x] CRM guncelleme (Phase 3.3)
    +-- lead_research_node()     # [x] Lead arastirma (Phase 4.2)
    +-- email_copilot_node()     # [x] Email asistani (Phase 4.3)
    +-- email_delivery_node()    # [x] Email gonderim (Phase 4.3)

backend/app/schemas/             # PYDANTIC SCHEMAS
+-- meeting_notes.py             # [x] NormalizedMeetingInput, MeetingAnalysis
+-- tasks.py                     # [x] ExtractedTask, TaskExtractionResult (Phase 3.2)
+-- crm_updates.py               # [x] CRMUpdate, CRMUpdateOperation (Phase 3.3)
+-- lead_research.py             # [x] LeadResearchInput, LeadResearchResult (Phase 4.2)
+-- email.py                     # [x] EmailCopilotInput, EmailGenerationResult (Phase 4.3)

backend/prompts/templates/       # PROMPT TEMPLATES (YAML + Jinja2)
+-- sales_ops.yaml               # [x] Meeting notes, task extraction prompts
+-- intelligence.yaml            # [ ] Market research prompts
+-- content.yaml                 # [ ] Content generation prompts
```

---

## Architecture Decision Records (ADR)

### ADR-001: Sistem Mimarisi Karari

#### Context
16+ karmasik gorevi otonom yurutecek bir sistem tasarlamamiz gerekiyor. Sistem:
- Moduler ve olceklenebilir olmali
- Her musteri icin izole calismali
- Human-in-the-loop (HITL) destegi saglamali
- Gozlemlenebilir (observable) olmali

#### Options

**Option A: Monolitik Workflow**
- (+) Basit implementasyon
- (-) Olceklenebilirlik sorunu, test zorlugu

**Option B: Hierarchical Multi-Agent (Supervisor + Squads)**
- (+) Moduler, her squad bagimsiz test edilebilir
- (+) Paralel isleme mumkun
- (+) Squad seviyesinde izolasyon
- (-) Daha karmasik orchestration

**Option C: Flat Agent Pool**
- (+) Maksimum esneklik
- (-) Koordinasyon karmasikligi, state yonetimi zorlugu

#### Decision: Option B - Hierarchical Multi-Agent

```
                    +---------------------+
                    |     Supervisor      |
                    |      (Router)       |
                    +----------+----------+
                               |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+-------+-------+      +-------+-------+      +-------+-------+
|  Intelligence |      |    Content    |      |   Sales Ops   |
|     Squad     |      |     Squad     |      |     Squad     |
+-------+-------+      +-------+-------+      +-------+-------+
| Market Res.   |      | Pipeline Mgr  |      | Lead Research |
| SEO Analysis  |      | Brandvoice    |      | Meeting Notes |
| Web Analysis  |      | SEO/GEO Opt.  |      | CRM Updater   |
| Audience      |      | Social Dist.  |      | Email Copilot |
+---------------+      | Publisher     |      | Task Extract  |
                       +---------------+      +---------------+
```

---

### ADR-002: State Management Stratejisi

#### Context
Tum ajanlarin tutarli bir sekilde veri paylasmasi gerekiyor.

#### Decision: Centralized TypedDict + Reducers

Mevcut `state.py` zaten bu yapiyi implement ediyor:

```python
class OrchestratorState(TypedDict):
    # Workflow Metadata
    workflow_id: str
    workflow_type: WorkflowType
    client_context: ClientContext

    # Squad Outputs
    intelligence_output: IntelligenceOutput
    content_output: ContentOutput
    sales_ops_output: SalesOpsOutput

    # HITL
    pending_approvals: List[ApprovalRequest]

    # Observability
    agent_execution_log: List[AgentMessage]
```

#### State Flow

```
+-----------+    +------------+    +------------+    +-----------+
|   Input   |--->|   Agent    |--->|   State    |--->|   Next    |
|   State   |    |   Logic    |    |   Update   |    |   Agent   |
+-----------+    +------------+    +------------+    +-----------+
                      |
                      v
              +---------------+
              |   Pydantic    |
              |  Validation   |
              +---------------+
```

---

### ADR-003: LangGraph Workflow Yapisi

#### Context
LangGraph ile cyclic, conditional workflow'lar tasarlamamiz gerekiyor.

#### Decision: Conditional Routing + Checkpointer

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

def create_workflow() -> StateGraph:
    workflow = StateGraph(OrchestratorState)

    # Ana Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("intelligence_squad", intelligence_squad_node)
    workflow.add_node("content_squad", content_squad_node)
    workflow.add_node("sales_ops_squad", sales_ops_squad_node)
    workflow.add_node("human_approval", human_approval_node)

    # Conditional Edges
    workflow.add_conditional_edges(
        "supervisor",
        route_to_squad,
        {
            "intelligence": "intelligence_squad",
            "content": "content_squad",
            "sales_ops": "sales_ops_squad",
            "end": END
        }
    )

    return workflow.compile(checkpointer=SqliteSaver.from_conn_string(":memory:"))
```

#### Workflow Diagram

```
                    +---------------------+
                    |       START         |
                    +----------+----------+
                               |
                               v
                    +----------+----------+
               +--->|     Supervisor      |<------------+
               |    +----------+----------+             |
               |               |                        |
               |    +----------+----------+             |
               |    |   Route Decision    |             |
               |    +----------+----------+             |
               |               |                        |
        +------+-------+-------+-------+--------+       |
        |              |               |                |
        v              v               v                |
+-------+-------+ +----+----+ +--------+-------+        |
|  Intelligence | | Content | |   Sales Ops    |        |
|     Squad     | |  Squad  | |     Squad      |        |
+-------+-------+ +----+----+ +--------+-------+        |
        |              |               |                |
        +--------------+---------------+                |
                       |                                |
                       v                                |
             +---------+---------+                      |
             |  Check Approvals  |                      |
             +---------+---------+                      |
                       |                                |
               +-------+-------+                        |
               |               |                        |
          (needs HITL)    (no approval)                 |
               |               |                        |
               v               +------------------------+
        +------+------+
        |Human Approval|
        |   (PAUSE)    |
        +-------------+
```

---

### ADR-004: CRM Integration Stratejisi (SDK + MCP Hybrid)

#### Context
HubSpot ve Salesforce ile hem okuma (discovery) hem yazma (CRUD) islemleri gerekiyor.

#### Options

**Option A: Sadece SDK**
- (+) Guvenilir, typed
- (-) Discovery icin esnek degil

**Option B: Sadece MCP**
- (+) Esnek, conversational
- (-) CRUD icin guvenilirlik sorunu

**Option C: Hybrid (MCP Read + SDK Write)**
- (+) En iyi kombinasyon
- (-) Iki sistem yonetimi

#### Decision: Option C - Hybrid Approach

```
+-------------------------------------------------------------+
|                    CRM Integration Layer                     |
+--------------------------+----------------------------------+
|      MCP Servers         |         SDK Wrappers             |
|   (Discovery & Read)     |        (CRUD Actions)            |
+--------------------------+----------------------------------+
| hubspot_mcp.py           | hubspot_sdk.py                   |
|   - search_contacts()    |   - create_contact()             |
|   - analyze_pipeline()   |   - update_deal()                |
|   - explore_properties() |   - create_note()                |
|                          |   - create_task()                |
| salesforce_mcp.py        | salesforce_sdk.py                |
|   - query_objects()      |   - update_opportunity()         |
|   - analyze_reports()    |   - create_activity()            |
+--------------------------+----------------------------------+
                              |
                              v
                    +---------+---------+
                    |  Human-in-the-Loop |
                    |    (CRM Writes)    |
                    +-------------------+
```

---

### ADR-005: RAG & Knowledge Base Stratejisi

#### Context
Musteri dokumanlari (Brandvoice PDF, urun kataloglari) ajanlara context olarak sunulmali.

#### Decision: Supabase Vector DB + Chunking Pipeline

```
+-------------------------------------------------------------+
|                       RAG Pipeline                           |
+-------------------------------------------------------------+
|                                                              |
|  +----------+   +----------+   +----------+                  |
|  |  Upload  |-->|  Chunk   |-->|  Embed   |                  |
|  |  (PDF)   |   | (Tokens) |   | (OpenAI) |                  |
|  +----------+   +----------+   +----------+                  |
|                                      |                       |
|                                      v                       |
|                            +---------+----------+            |
|                            |     Supabase       |            |
|                            |    Vector DB       |            |
|                            |    (pgvector)      |            |
|                            +---------+----------+            |
|                                      |                       |
|  +----------+              +---------+----------+            |
|  |  Agent   |<-------------|    Similarity      |            |
|  |  Query   |              |      Search        |            |
|  +----------+              +--------------------+            |
|                                                              |
+-------------------------------------------------------------+
```

#### Chunking Configuration

```python
CHUNK_CONFIG = {
    "chunk_size": 1000,        # tokens
    "chunk_overlap": 200,      # tokens
    "separators": ["\n\n", "\n", ". "],
    "metadata_fields": ["client_id", "doc_type", "source"]
}
```

---

### ADR-006: Human-in-the-Loop (HITL) Approval System

#### Context
CRM guncellemeleri, icerik yayinlama gibi kritik aksiyonlar kullanici onayi beklemeli.

#### Decision: Approval Queue + Webhook Notification

```
+-------------------------------------------------------------+
|                    HITL Approval Flow                        |
+-------------------------------------------------------------+
|                                                              |
|  Agent Action                                                |
|       |                                                      |
|       v                                                      |
|  +---------------+                                           |
|  | Side Effect?  |                                           |
|  +-------+-------+                                           |
|          |                                                   |
|     +----+----+                                               |
|     |         |                                              |
|    YES       NO                                              |
|     |         |                                              |
|     v         v                                              |
|  +--------+  Execute                                         |
|  |Approval|  Immediately                                     |
|  | Queue  |                                                  |
|  +---+----+                                                  |
|      |                                                       |
|      v                                                       |
|  +------------------+   +------------------+                  |
|  | Supabase Table   |-->|    Frontend      |                 |
|  | (approvals)      |   |    Dashboard     |                 |
|  +------------------+   +--------+---------+                 |
|                                  |                           |
|                         User Decision                        |
|                         (Approve/Reject)                     |
|                                  |                           |
|                                  v                           |
|                         +--------+---------+                 |
|                         | Resume Workflow  |                 |
|                         |   (LangGraph)    |                 |
|                         +------------------+                 |
|                                                              |
+-------------------------------------------------------------+
```

#### Approval Types

```python
class ApprovalType(str, Enum):
    CRM_UPDATE = "crm_update"
    CONTENT_PUBLISH = "content_publish"
    EMAIL_SEND = "email_send"
    TASK_CREATE = "task_create"
    MEETING_SCHEDULE = "meeting_schedule"
```

---

### ADR-007: Agent Implementation Pattern

#### Context
15+ ajanin tutarli bir pattern ile implement edilmesi gerekiyor.

#### Decision: Base Agent Class + Pydantic Output Schema

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel

class BaseAgent(ABC):
    """Tum ajanlarin temel sinifi"""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool] = None
    ):
        self.llm = llm
        self.tools = tools or []
        self.output_schema: Type[BaseModel] = None

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Agent'a ozel system prompt"""
        pass

    @abstractmethod
    def process(self, state: OrchestratorState) -> Dict[str, Any]:
        """State'i isle ve guncellemeleri dondur"""
        pass

    def validate_output(self, output: Any) -> BaseModel:
        """Pydantic ile output validation"""
        return self.output_schema.model_validate(output)
```

---

### ADR-008: API Layer Design

#### Context
Frontend ve external systems ile iletisim icin RESTful API gerekiyor.

#### Decision: FastAPI + Async Endpoints

```python
# Workflow Endpoints
POST   /api/v1/workflows                    # Yeni workflow baslat
GET    /api/v1/workflows/{id}               # Workflow durumu
POST   /api/v1/workflows/{id}/resume        # Onay sonrasi devam

# Approval Endpoints
GET    /api/v1/approvals                    # Bekleyen onaylar
POST   /api/v1/approvals/{id}/approve       # Onayla
POST   /api/v1/approvals/{id}/reject        # Reddet

# Document Endpoints (RAG)
POST   /api/v1/documents                    # Dokuman yukle
GET    /api/v1/documents                    # Dokumanlari listele
DELETE /api/v1/documents/{id}               # Dokuman sil
```

---

### ADR-009: Frontend Dashboard Architecture

#### Context
Musterilerin workflow'lari izlemesi ve onaylamasi icin dashboard gerekiyor.

#### Decision: Next.js App Router + Supabase Realtime

```
Frontend Component Structure
============================

src/
+-- app/
|   +-- layout.tsx              # Root layout
|   +-- page.tsx                # Dashboard home
|   +-- workflows/
|   |   +-- page.tsx            # Workflow listesi
|   |   +-- [id]/page.tsx       # Workflow detay
|   +-- approvals/
|   |   +-- page.tsx            # Onay bekleyenler
|   |   +-- [id]/page.tsx       # Onay detay
|   +-- settings/
|       +-- page.tsx            # Ayarlar
|
+-- components/
|   +-- workflow/
|   |   +-- WorkflowCard.tsx
|   |   +-- WorkflowTimeline.tsx
|   |   +-- WorkflowStatus.tsx
|   +-- approval/
|   |   +-- ApprovalCard.tsx
|   |   +-- ApprovalDiff.tsx    # Degisiklik onizleme
|   |   +-- ApprovalActions.tsx
|   +-- common/
|       +-- Header.tsx
|       +-- Sidebar.tsx
|       +-- LoadingState.tsx
|
+-- hooks/
|   +-- useWorkflows.ts         # Workflow data fetching
|   +-- useApprovals.ts         # Approval data fetching
|   +-- useRealtime.ts          # Supabase realtime
|
+-- services/
    +-- api.ts                  # API client
    +-- supabase.ts             # Supabase client
```

---

### ADR-010: Database Schema Design

#### Context
Workflow state, approvals ve vector embeddings icin veritabani semasi gerekiyor.

#### Decision: Supabase PostgreSQL + pgvector

```sql
-- Clients (Musteriler)
CREATE TABLE clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    hubspot_portal_id VARCHAR(50),
    salesforce_org_id VARCHAR(50),
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Workflows (Is Akislari)
CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID REFERENCES clients(id),
    workflow_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    state JSONB NOT NULL,
    checkpointer_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Approvals (Onaylar)
CREATE TABLE approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows(id),
    client_id UUID REFERENCES clients(id),
    approval_type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    payload JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    reviewed_by UUID,
    reviewed_at TIMESTAMPTZ,
    rejection_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Documents (RAG icin)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID REFERENCES clients(id),
    doc_type VARCHAR(50) NOT NULL,
    title VARCHAR(255),
    content TEXT,
    metadata JSONB DEFAULT '{}',
    storage_path VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document Chunks (Vector embeddings)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity search index
CREATE INDEX ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Agent Execution Logs
CREATE TABLE agent_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows(id),
    agent_name VARCHAR(100) NOT NULL,
    input_summary TEXT,
    output_summary TEXT,
    tokens_used INT,
    duration_ms INT,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Row Level Security (Multi-tenant isolation)
ALTER TABLE workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE approvals ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Clients see own workflows"
ON workflows FOR ALL
USING (client_id = auth.uid());

CREATE POLICY "Clients see own approvals"
ON approvals FOR ALL
USING (client_id = auth.uid());
```

---

### ADR-011: Faz Onceliklendirmesi

#### Decision: Faz 1 -> Faz 2 -> Faz 3 (Sirali)

```
+-------------------------------------------------------------+
|                    IMPLEMENTATION ROADMAP                    |
+-------------------------------------------------------------+
|                                                              |
|  Faz 1: Foundation          [====================]  (ONCE)   |
|  +-- FastAPI App Setup                                       |
|  +-- LangGraph Workflow Core                                 |
|  +-- Supabase Connection                                     |
|  +-- Docker Dev Environment                                  |
|  +-- Base Agent Pattern                                      |
|                                                              |
|  Faz 2: Services            [                    ]  (SONRA)  |
|  +-- HubSpot SDK Wrapper                                     |
|  +-- Tavily Search Integration                               |
|  +-- Vector Service (RAG)                                    |
|  +-- Storage Service                                         |
|                                                              |
|  Faz 3: First Squad         [                    ]  (EN SON) |
|  +-- Sales Ops Squad (MVP)                                   |
|                                                              |
+-------------------------------------------------------------+
```

---

### ADR-012: Squad Onceligi

#### Evaluation

| Squad | Karmasiklik | Business Value | Bagimlilik | Skor |
|-------|-------------|----------------|------------|------|
| Intelligence | Orta | Orta | Dusuk | 3/5 |
| Content | Yuksek | Orta | Yuksek | 2/5 |
| Sales Ops | Dusuk-Orta | Yuksek | Dusuk | 5/5 |

#### Decision: Sales Ops Squad Ilk

```
+-------------------------------------------------------------+
|                     SQUAD PRIORITY ORDER                     |
+-------------------------------------------------------------+
|                                                              |
|  1. SALES OPS SQUAD (ILK)                                    |
|  +-----------------------------------------------------+     |
|  | Meeting Notes Analyzer  --- En somut input/output   |     |
|  | Task Extractor          --- CRM'e yazma pratigi     |     |
|  | CRM Updater             --- HITL pattern testi      |     |
|  | Lead Research           --- Tavily + CRM combo      |     |
|  | Email Copilot           --- Content generation      |     |
|  +-----------------------------------------------------+     |
|                                                              |
|  2. INTELLIGENCE SQUAD (IKINCI)                              |
|  +-----------------------------------------------------+     |
|  | Market Research         --- Tavily heavy            |     |
|  | SEO Analysis            --- External API'lar        |     |
|  | Web Analysis            --- Scraping complexity     |     |
|  | Audience Builder        --- CRM data analysis       |     |
|  +-----------------------------------------------------+     |
|                                                              |
|  3. CONTENT SQUAD (SON)                                      |
|  +-----------------------------------------------------+     |
|  | Pipeline Manager        --- Intelligence'a bagimli  |     |
|  | Brandvoice Writer       --- RAG gerekli             |     |
|  | SEO/GEO Optimizer       --- Intelligence output'u   |     |
|  | Social Distributor      --- External API'lar        |     |
|  | Web Publisher           --- En fazla side-effect    |     |
|  +-----------------------------------------------------+     |
|                                                              |
+-------------------------------------------------------------+
```

---

### ADR-013: MVP Scope Tanimi

#### Decision: "Meeting-to-CRM" MVP

```
+-------------------------------------------------------------+
|                         MVP SCOPE                            |
|                   "Meeting Notes -> CRM Actions"             |
+-------------------------------------------------------------+
|                                                              |
|  INPUT                                                       |
|  -----                                                       |
|  - Meeting transcript (text/audio transkript)                |
|  - Client context (HubSpot portal info)                      |
|                                                              |
|  AGENTS (3 tane yeterli)                                     |
|  ------                                                      |
|  +--------------------+                                      |
|  | Meeting Notes      |---> Notlari analiz et                |
|  | Analyzer           |     Action items cikar               |
|  +---------+----------+                                      |
|            |                                                 |
|            v                                                 |
|  +---------+----------+                                      |
|  | Task Extractor     |---> HubSpot task'lari olustur        |
|  |                    |     Due date, assignee belirle       |
|  +---------+----------+                                      |
|            |                                                 |
|            v                                                 |
|  +---------+----------+                                      |
|  | CRM Updater        |---> Contact/Deal notlari guncelle    |
|  |                    |     Activity log ekle                |
|  +--------------------+                                      |
|                                                              |
|  OUTPUT                                                      |
|  ------                                                      |
|  - Meeting summary (structured)                              |
|  - Extracted tasks (pending approval)                        |
|  - CRM updates (pending approval)                            |
|                                                              |
|  HITL APPROVAL                                               |
|  ------------                                                |
|  - Task creation ---> User approval ---> HubSpot             |
|  - CRM update ---> User approval ---> HubSpot                |
|                                                              |
+-------------------------------------------------------------+
```

#### MVP Feature Checklist

```
MUST HAVE (MVP)
===============
[x] FastAPI server running                              # Phase 1 - Complete
[x] LangGraph workflow (3 nodes)                        # Phase 1 - Complete (nodes.py)
[x] Meeting Notes Analyzer agent                        # Phase 3.1 - Complete
[x] Task Extractor agent                                # Phase 3.2 - Complete
[x] CRM Updater agent                                   # Phase 3.3 - Complete
[x] HubSpot SDK integration (read + write)              # Phase 2 - Complete
[x] Approval system (basic)                             # Phase 2 - Complete
[x] Supabase storage (workflow state)                   # Phase 1 - Complete
[x] Single API endpoint: POST /workflows/meeting-analysis # Phase 2 - Complete
[x] Docker compose (dev environment)                    # Phase 1 - Complete

NICE TO HAVE (Post-MVP)
=======================
[ ] Frontend dashboard
[ ] Realtime updates
[ ] Vector DB / RAG
[ ] Email Copilot
[ ] Lead Research
[ ] Multi-CRM (Salesforce)

NOT IN MVP
==========
[x] Intelligence Squad
[x] Content Squad
[x] MCP servers
[x] Production deployment
[x] Auth system (internal use first)
```

**MVP Progress: 10/10 Must-Have items complete (100%)**

---

### ADR-014: CRM Human-in-the-Loop (HITL) Control Mechanism

#### Context
CRM verileri is acisinda kritik oneme sahip. Yapay zekanin kontrolsuz erisimi veri kaybi, iliski hasari, pipeline bozulmasi ve compliance ihlali risklerini tasir.

#### Decision: Multi-Layer Protection System

**3 Katmanli Koruma:**

1. **Operation Classification (Islem Siniflandirma)**
   - READ: Otomatik (onay gerekmez)
   - CREATE: Preview + Approval
   - UPDATE: Preview + Diff + Approval
   - DELETE: BLOCKED (sadece soft-delete, admin only)
   - BULK (>5): Extra confirmation + Rate limit

2. **Risk-Based Tiered Approval (Risk Skorlama)**
   - LOW (0-30): Onay ekrani, hizli onay
   - MEDIUM (31-70): Detayli onizleme, mutlaka onay
   - HIGH (71-100): Detayli onizleme + uyari + yazarak onay

3. **Audit & Rollback**
   - Tum islemler loglanir (before/after state)
   - 30 gun icinde one-click rollback imkani

**Risk Faktorleri:**
- Kayit sayisi (1-2: dusuk, 3-10: orta, 10+: yuksek)
- Alan hassasiyeti (email, telefon, deal_amount, deal_stage, owner_id)
- Deal degeri ($5K alti: dusuk, $5K-$50K: orta, $50K+: yuksek)

**Guvenlik Limitleri:**
- Max 10 operation per workflow run
- 500ms delay between CRM API calls
- Daily limit: 100 operations per client
- Blocked: contact.delete, deal.delete, company.merge, bulk_delete

#### Impact
- Guvenlik: Maksimum - her CRM yazma islemi kontrollu
- Compliance: Tam audit trail, GDPR/KVKK uyumlu
- Rollback: 30 gun icinde geri alma imkani

---

### ADR-015: SDK vs MCP Kullanim Stratejisi

#### Context
HubSpot/Salesforce entegrasyonu icin iki teknoloji secenegi var: SDK (tam kontrol) ve MCP (akilli kesif).

#### Decision: Hybrid Approach (SDK + MCP)

**SDK Kullanim Alanlari (Kontrollu):**
- Tum CRM yazma islemleri (create, update)
- Tum CRM okuma islemleri (get, search)
- Side-effect olan her islem

**MCP Kullanim Alanlari (Akilli Kesif):**
- Web arastirma (Tavily)
- SEO analizi
- Rakip analizi
- Market research
- Lead enrichment (web kaynaklarindan)
- Trend arastirma

**Temel Kural:**
> Side-effect olan hicbir islem MCP ile yapilmaz.

**Phase Bazli Kullanim:**
| Phase | SDK | MCP |
|-------|-----|-----|
| Phase 2 (Services) | HubSpot CRUD | Yok |
| Phase 3 (MVP Agents) | CRM Updater | Yok |
| Phase 4 (Sales Ops+) | CRM islemleri | Lead enrichment (web) |
| Phase 5 (Intelligence) | CRM okuma | Market research, SEO |
| Phase 6 (Content) | Publishing meta | Trend analysis |

#### Impact
- Guvenlik: CRM islemleri tam kontrol altinda
- Esneklik: Arastirma islemleri icin AI ozgurlugu
- Karmasiklik: Yonetilebilir - net ayrim var

---

### ADR-016: Multi-Tenant CRM Integration Architecture

#### Context
Her musteri kendi HubSpot portalina sahip ve kendi API tokenini kullanmali. Mevcut singleton pattern coklu musteri senaryosunda calismaz. Gereksinimler:
- Teknik olmayan musteriler icin en basit baglanti yontemi
- Production-grade guvenlik
- Olceklenebilir mimari

#### Options Evaluated

**Token Toplama Yontemi:**

| Yontem | Musteri Deneyimi | Guvenlik | Karmasiklik |
|--------|------------------|----------|-------------|
| Manual Token Entry | Kotu (Portal'a girip token kopyala) | Dusuk (Token paylasimi) | Dusuk |
| OAuth 2.0 Flow | Mukemmel (Tek tik "Connect") | Yuksek (Token paylasimi yok) | Orta |
| Private App Install | Orta (Admin panelinde kurulum) | Orta | Dusuk |

**Token Saklama Yontemi:**

| Yontem | Guvenlik | Karmasiklik | Maliyet |
|--------|----------|-------------|---------|
| Plain text DB | Cok Dusuk | Cok Dusuk | Yok |
| Encrypted DB (AES-256) | Yuksek | Dusuk | Dusuk |
| Vault/Secrets Manager | Cok Yuksek | Orta | Orta |
| Vault + HSM | Maksimum | Yuksek | Yuksek |

**Client Management Pattern:**

| Pattern | Kod Temizligi | Performans | Test Edilebilirlik |
|---------|---------------|------------|-------------------|
| Token per request | Dusuk | Dusuk (her seferinde yeni) | Orta |
| Client Factory + Cache | Orta | Yuksek (cache) | Iyi |
| Context-Based DI | Yuksek (FastAPI native) | Yuksek | Cok Iyi |

#### Decision: OAuth + Encrypted DB + Context-Based DI

**1. Token Toplama: OAuth 2.0 Flow**
```
+-------------------------------------------------------------+
|                    CUSTOMER ONBOARDING                       |
+-------------------------------------------------------------+
|                                                              |
|  1. Musteri Dashboard'a giris yapar                          |
|                                                              |
|  2. "Connect HubSpot" butonuna tiklar                        |
|     +------------------+                                     |
|     | Connect HubSpot  |  <-- Tek tik, teknik bilgi gerekmez |
|     +------------------+                                     |
|                                                              |
|  3. HubSpot login sayfasina yonlendirilir                    |
|     +------------------+                                     |
|     | HubSpot Login    |  <-- Musteri kendi HubSpot'una girer|
|     | [Email]          |                                     |
|     | [Password]       |                                     |
|     +------------------+                                     |
|                                                              |
|  4. Izin ekrani                                              |
|     +------------------+                                     |
|     | "CRM AI wants:   |                                     |
|     |  - Read contacts |  <-- Musteri izinleri gorur         |
|     |  - Create tasks" |                                     |
|     | [Authorize]      |                                     |
|     +------------------+                                     |
|                                                              |
|  5. Otomatik geri donus + Token saklama                      |
|     Dashboard'da "Connected ✓" gosterilir                    |
|                                                              |
+-------------------------------------------------------------+
```

**2. Token Saklama: Encrypted DB (MVP) + Vault Ready**
```
+-------------------------------------------------------------+
|                    TOKEN STORAGE ARCHITECTURE                |
+-------------------------------------------------------------+
|                                                              |
|  MVP (Phase 2):                                              |
|  +------------------+                                        |
|  | Supabase DB      |                                        |
|  | +--------------- |                                        |
|  | | crm_credentials|                                        |
|  | | - client_id    |                                        |
|  | | - provider     |  (hubspot, salesforce)                 |
|  | | - access_token |  <-- AES-256-GCM encrypted             |
|  | | - refresh_token|  <-- AES-256-GCM encrypted             |
|  | | - expires_at   |                                        |
|  | | - scopes       |                                        |
|  | +--------------- |                                        |
|  +------------------+                                        |
|           |                                                  |
|           v                                                  |
|  +------------------+                                        |
|  | Encryption Key   |  <-- Environment variable              |
|  | (CRM_ENCRYPTION_ |      (Production: Vault/KMS)           |
|  |  KEY)            |                                        |
|  +------------------+                                        |
|                                                              |
|  Production (Phase 7+):                                      |
|  +------------------+      +------------------+               |
|  | AWS Secrets Mgr  | veya | HashiCorp Vault  |              |
|  | / GCP Secret Mgr |      |                  |              |
|  +------------------+      +------------------+               |
|                                                              |
+-------------------------------------------------------------+
```

**3. Multi-Tenant Client Pattern: Context-Based DI**
```python
# HubSpot client artik client_id'ye gore token alir
class HubSpotClientFactory:
    _cache: Dict[str, HubSpotClient] = {}

    async def get_client(self, client_id: str) -> HubSpotClient:
        if client_id in self._cache:
            return self._cache[client_id]

        # Token'i decrypt ederek al
        token = await self._get_decrypted_token(client_id)
        client = HubSpotClient(access_token=token)
        self._cache[client_id] = client
        return client

# FastAPI dependency - otomatik olarak client context'ten alinir
async def get_hubspot_client(
    client: ClientContextDep,
    factory: HubSpotClientFactoryDep
) -> HubSpotClient:
    return await factory.get_client(client.client_id)

# Endpoint kullanimi - temiz ve guvenli
@router.get("/contacts/{contact_id}")
async def get_contact(
    contact_id: str,
    hubspot: HubSpotClientDep  # Otomatik dogru token kullanilir
):
    return await hubspot.get_contact(contact_id)
```

#### Database Schema Addition

```sql
-- CRM Credentials (Multi-tenant token storage)
CREATE TABLE crm_credentials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    provider VARCHAR(20) NOT NULL,  -- 'hubspot', 'salesforce'

    -- OAuth tokens (encrypted)
    access_token_encrypted BYTEA NOT NULL,
    refresh_token_encrypted BYTEA,
    token_iv BYTEA NOT NULL,  -- Initialization vector for AES

    -- Token metadata
    expires_at TIMESTAMPTZ,
    scopes TEXT[],
    portal_id VARCHAR(50),  -- HubSpot portal ID

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMPTZ,
    last_refresh_at TIMESTAMPTZ,

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(client_id, provider)
);

-- Index for fast lookups
CREATE INDEX idx_crm_credentials_client_provider
ON crm_credentials(client_id, provider) WHERE is_active = true;

-- RLS for multi-tenant isolation
ALTER TABLE crm_credentials ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Clients see own credentials"
ON crm_credentials FOR ALL USING (client_id = auth.uid());
```

#### OAuth Flow Endpoints

```python
# OAuth endpoints
GET  /api/v1/oauth/hubspot/authorize     # Redirect to HubSpot
GET  /api/v1/oauth/hubspot/callback      # Handle OAuth callback
POST /api/v1/oauth/hubspot/refresh       # Manual token refresh
DELETE /api/v1/oauth/hubspot/disconnect  # Revoke and delete token
GET  /api/v1/oauth/status                # Check connection status
```

#### Security Measures

1. **Encryption**: AES-256-GCM for tokens at rest
2. **Key Rotation**: Quarterly encryption key rotation support
3. **Token Refresh**: Automatic refresh before expiry
4. **Scope Limitation**: Request minimum required scopes only
5. **Audit Log**: All token operations logged
6. **RLS**: Database-level tenant isolation

#### Implementation Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 2.1.4 | OAuth endpoints + Encrypted storage | [ ] |
| 2.1.5 | HubSpotService multi-tenant refactor | [ ] |
| 2.1.6 | Token refresh background job | [ ] |
| 7.x | Vault/Secrets Manager migration | [ ] |

#### Impact
- Musteri Deneyimi: Tek tik baglanti, teknik bilgi gerekmez
- Guvenlik: AES-256 encryption, OAuth (token paylasimi yok)
- Olceklenebilirlik: Client factory + cache pattern
- Production Ready: Vault migration path hazir

---

### ADR-017: Meeting Notes Input Architecture (Extensible Adapter Pattern)

#### Context
Meeting Notes Analyzer agent'i icin input kaynagi belirlenmesi gerekiyordu. Potansiyel kaynaklar:
- Manuel text input (kullanici yapistirma)
- Dosya yukleme (txt, docx, pdf)
- Takvim entegrasyonlari (Google Calendar, Outlook)
- Transkripsiyon servisleri (Otter.ai, Fireflies.ai, Zoom)

#### Options Evaluated

| Kaynak | MVP Uygunlugu | Karmasiklik | Deger |
|--------|---------------|-------------|-------|
| Manual Text | Yuksek | Dusuk | Hemen kullanilabilir |
| File Upload | Orta | Orta | Parsing gerekli |
| Calendar API | Dusuk | Yuksek | OAuth gerekli |
| Transkripsiyon | Dusuk | Yuksek | External API + Cost |

#### Decision: NormalizedMeetingInput + Adapter Pattern

**MVP:** Manual text input ile basla
**Gelecek:** Extensible adapter pattern ile diger kaynaklari ekle

```
+-------------------------------------------------------------+
|                  MEETING INPUT ARCHITECTURE                  |
+-------------------------------------------------------------+
|                                                              |
|  INPUT SOURCES (Adapter'lar)                                |
|  +------------------+  +------------------+                  |
|  | Manual Text      |  | File Upload      |                 |
|  | (MVP)            |  | (Phase 4)        |                 |
|  +--------+---------+  +--------+---------+                 |
|           |                     |                           |
|           +----------+----------+                           |
|                      |                                      |
|                      v                                      |
|           +----------+----------+                           |
|           | NormalizedMeeting   |  <-- Standard Format      |
|           | Input (Pydantic)    |                           |
|           +----------+----------+                           |
|                      |                                      |
|                      v                                      |
|           +----------+----------+                           |
|           | Meeting Notes       |                           |
|           | Analyzer Node       |                           |
|           +----------+----------+                           |
|                      |                                      |
|                      v                                      |
|           +----------+----------+                           |
|           | MeetingAnalysis     |  <-- Structured Output    |
|           | (Pydantic)          |                           |
|           +---------------------+                           |
|                                                              |
+-------------------------------------------------------------+
```

#### NormalizedMeetingInput Schema

```python
class NormalizedMeetingInput(BaseModel):
    # Source tracking
    source: MeetingInputSource  # manual_text, file_upload, google_calendar, etc.
    source_id: str | None       # External ID from source system

    # Meeting metadata
    title: str | None
    meeting_date: datetime | None
    duration_minutes: int | None
    participants: list[str]
    organizer: str | None

    # Content
    transcript: str             # REQUIRED - main meeting content

    # CRM Context
    deal_id: str | None
    contact_id: str | None
    company_id: str | None
    additional_context: str | None
```

#### MeetingAnalysis Output Schema

```python
class MeetingAnalysis(BaseModel):
    # Summary
    summary: str                           # 2-3 sentence summary
    key_points: list[str]                  # Main discussion points

    # Decisions & Actions
    key_decisions: list[KeyDecision]       # Decisions made
    action_items: list[ActionItem]         # Tasks with assignee/due_date

    # Sentiment
    overall_sentiment: MeetingSentiment    # positive/neutral/negative
    sentiment_explanation: str | None

    # Follow-up
    follow_up_required: bool
    follow_up_reason: str | None

    # CRM Recommendations
    deal_stage_recommendation: DealStageRecommendation | None

    # Additional Insights
    identified_participants: list[str]
    risks_concerns: list[str]
    opportunities: list[str]
    next_steps: list[str]
```

#### Future Adapter Examples

```python
# Phase 4+: File Upload Adapter
class FileUploadAdapter(MeetingInputAdapter):
    async def normalize(self, file: UploadFile) -> NormalizedMeetingInput:
        content = await self._parse_file(file)  # txt, docx, pdf
        return NormalizedMeetingInput(
            source=MeetingInputSource.FILE_UPLOAD,
            transcript=content,
            ...
        )

# Phase 5+: Google Calendar Adapter
class GoogleCalendarAdapter(MeetingInputAdapter):
    async def normalize(self, event_id: str) -> NormalizedMeetingInput:
        event = await self._fetch_event(event_id)
        recording = await self._fetch_recording(event)
        transcript = await self._transcribe(recording)
        return NormalizedMeetingInput(
            source=MeetingInputSource.GOOGLE_CALENDAR,
            source_id=event_id,
            title=event.summary,
            meeting_date=event.start,
            participants=event.attendees,
            transcript=transcript,
            ...
        )
```

#### Impact
- Esneklik: Yeni input kaynaklari kolayca eklenebilir
- Tutarlilik: Tum kaynaklar ayni format'a normalize edilir
- Test Edilebilirlik: Her adapter bagimsiz test edilebilir
- MVP Odakli: Basit basla, gerektiginde genislet

---

## Success Metrics

| Metrik | Hedef | Olcum |
|--------|-------|-------|
| Workflow Completion Rate | > 95% | Tamamlanan / Baslatilan |
| Average Approval Time | < 4 saat | Onay bekleme suresi |
| Token Efficiency | < 50K/workflow | LangSmith tracing |
| Agent Accuracy | > 90% | Pydantic validation pass rate |
| System Uptime | > 99.5% | Monitoring |

---

## Risk Analysis

| Risk | Olasilik | Etki | Mitigasyon |
|------|----------|------|------------|
| LLM Halusinasyon | Yuksek | Yuksek | Pydantic semalari, kaynak zorunlulugu |
| CRM Rate Limiting | Orta | Orta | Queue system, exponential backoff |
| Token Cost Overrun | Orta | Orta | Token budgets, summarization |
| Multi-tenant Data Leak | Dusuk | Cok Yuksek | RLS, client_id validation her yerde |
| Workflow Stuck | Orta | Orta | Timeout handlers, dead letter queue |

---

# TODO LIST - Implementation Roadmap

## PHASE 1: FOUNDATION

### 1.1 FastAPI Application Setup

| # | Task | File | Status |
|---|------|------|--------|
| 1.1.1 | Create Pydantic Settings class with all env variables | `backend/app/core/config.py` | [x] |
| 1.1.2 | Create environment variables template | `.env.example` | [x] |
| 1.1.3 | Create dependency injection setup | `backend/app/core/dependencies.py` | [x] |
| 1.1.4 | Create FastAPI app with CORS, routers, health check | `backend/app/main.py` | [x] |
| 1.1.5 | Create Python container configuration | `backend/Dockerfile` | [x] |
| 1.1.6 | Create dev environment docker-compose | `docker-compose.yml` | [x] |

### 1.2 Database Setup

| # | Task | File | Status |
|---|------|------|--------|
| 1.2.1 | Create async Supabase client | `backend/services/supabase_client.py` | [x] |
| 1.2.2 | Create initial database schema | `backend/migrations/001_initial_schema.sql` | [x] |

### 1.3 LangGraph Core Setup

| # | Task | File | Status |
|---|------|------|--------|
| 1.3.1 | Create StateGraph with checkpointer setup | `backend/graph/workflow.py` | [x] |
| 1.3.2 | Create node function stubs | `backend/graph/nodes.py` | [x] |
| 1.3.3 | Create conditional routing logic | `backend/graph/routers.py` | [x] |

### 1.4 Base Agent Pattern

| # | Task | File | Status |
|---|------|------|--------|
| 1.4.1 | Create BaseAgent ABC class | `backend/agents/base.py` | [x] |
| 1.4.2 | Create base Pydantic schemas | `backend/app/schemas/base.py` | [x] |
| 1.4.3 | Create prompt template management | `backend/prompts/base.py` | [x] |

### 1.5 LLM Service Setup

| # | Task | File | Status |
|---|------|------|--------|
| 1.5.1 | Create Claude/GPT client setup with fallback | `backend/services/llm_service.py` | [x] |

### 1.6 Phase 1 Verification

| # | Task | Criteria | Status |
|---|------|----------|--------|
| 1.6.1 | Test Docker setup | `docker-compose up` runs successfully | [x] |
| 1.6.2 | Test health endpoint | `GET /health` returns 200 OK | [x] |
| 1.6.3 | Test LangGraph | Workflow compiles without errors | [x] |

---

## PHASE 2: SERVICES

### 2.1 HubSpot Integration

| # | Task | File | Status |
|---|------|------|--------|
| 2.1.1 | Create HubSpot client wrapper (contacts, deals, tasks, notes) | `backend/services/hubspot_service.py` | [x] |
| 2.1.2 | Create Pydantic models for HubSpot entities | `backend/app/schemas/hubspot.py` | [x] |
| 2.1.3 | Create LangChain tools for HubSpot operations | `backend/tools/hubspot_tools.py` | [x] |
| 2.1.4 | Create encryption service for token storage | `backend/services/encryption_service.py` | [x] |
| 2.1.5 | Create OAuth flow service for HubSpot | `backend/services/oauth_service.py` | [x] |
| 2.1.6 | Create OAuth REST endpoints | `backend/app/api/v1/oauth.py` | [x] |
| 2.1.7 | Refactor HubSpotService to multi-tenant pattern | `backend/services/hubspot_service.py` | [x] |
| 2.1.8 | Create database migration for crm_credentials table | `backend/migrations/002_crm_credentials.sql` | [x] |

### 2.2 Approval System

| # | Task | File | Status |
|---|------|------|--------|
| 2.2.1 | Create approval service (create, list, approve, reject) | `backend/services/approval_service.py` | [x] |
| 2.2.2 | Create REST endpoints for approvals | `backend/app/api/v1/approvals.py` | [x] |
| 2.2.3 | Create approval request/response models | `backend/app/schemas/approvals.py` | [x] |

### 2.3 Workflow API

| # | Task | File | Status |
|---|------|------|--------|
| 2.3.1 | Create workflow trigger, status, resume endpoints | `backend/app/api/v1/workflow.py` | [x] |
| 2.3.2 | Create workflow orchestration logic | `backend/services/workflow_service.py` | [x] |
| 2.3.3 | Create workflow request/response models | `backend/app/schemas/workflow.py` | [x] |

### 2.4 Phase 2 Verification

| # | Task | Criteria | Status |
|---|------|----------|--------|
| 2.4.1 | Test HubSpot connection | Can fetch contacts from HubSpot | [x] |
| 2.4.2 | Test approval API | CRUD operations work | [x] |
| 2.4.3 | Test workflow API | Can trigger and query workflow | [x] |

---

## PHASE 3: MVP AGENTS

### 3.1 Meeting Notes Analyzer

| # | Task | File | Status |
|---|------|------|--------|
| 3.1.1 | Create Meeting Notes Analyzer node | `backend/graph/nodes.py:meeting_notes_node` | [x] |
| 3.1.2 | Create MeetingAnalysis Pydantic schema | `backend/app/schemas/meeting_notes.py` | [x] |
| 3.1.3 | Create NormalizedMeetingInput schema | `backend/app/schemas/meeting_notes.py` | [x] |
| 3.1.4 | Create system prompt for meeting analysis | `backend/prompts/templates/sales_ops.yaml` | [x] |
| 3.1.5 | Create unit tests for meeting notes agent | `backend/tests/test_meeting_notes_agent.py` | [x] |

**Note:** Phase 3.1 implemented using LangGraph nodes pattern (in `nodes.py`) rather than separate agent files. The prompt is stored in the YAML templates with Jinja2 support. Tests verify: schema validation, LLM integration with mocks, state management, and structured output parsing.

### 3.2 Task Extractor

| # | Task | File | Status |
|---|------|------|--------|
| 3.2.1 | Create Task Extractor node | `backend/graph/nodes.py:task_extractor_node` | [x] |
| 3.2.2 | Create ExtractedTask Pydantic schema | `backend/app/schemas/tasks.py` | [x] |
| 3.2.3 | Create system prompt for task extraction | `backend/prompts/templates/sales_ops.yaml` | [x] |
| 3.2.4 | Create unit tests for task extractor agent | `backend/tests/test_task_extractor_agent.py` | [x] |

**Note:** Phase 3.2 implemented using LangGraph nodes pattern (in `nodes.py`). The task extractor:
- Processes action items from meeting notes
- Uses LLM to enrich tasks with context, calculate due dates, set priorities
- Converts relative dates (e.g., "next week") to absolute YYYY-MM-DD format
- Creates HubSpot-ready CRM task payloads
- Flags tasks needing human review
- Has fallback to basic extraction if LLM fails

### 3.3 CRM Updater

| # | Task | File | Status |
|---|------|------|--------|
| 3.3.1 | Create CRMUpdate Pydantic schema | `backend/app/schemas/crm_updates.py` | [x] |
| 3.3.2 | Create CRM Updater node | `backend/graph/nodes.py:crm_updater_node` | [x] |
| 3.3.3 | Create system prompt for CRM updates | `backend/prompts/templates/sales_ops.yaml` | [x] |
| 3.3.4 | Create unit tests for CRM updater agent | `backend/tests/test_crm_updater_agent.py` | [x] |

**Note:** Phase 3.3 implemented using LangGraph nodes pattern (in `nodes.py`). The CRM Updater:
- Processes pending CRM tasks from state
- Uses LLM to transform tasks into structured CRM operations
- Assesses risk levels per ADR-014 (low/medium/high)
- Creates HITL approval requests with detailed payloads
- Includes deal stage recommendations and meeting notes
- Has fallback mode when LLM fails

### 3.4 Integration & Testing

| # | Task | File | Status |
|---|------|------|--------|
| 3.4.1 | Sales Ops workflow with conditional routing | `backend/graph/workflow.py:build_meeting_analysis_workflow` | [x] |
| 3.4.2 | Create end-to-end workflow test | `backend/tests/test_meeting_workflow_e2e.py` | [x] |
| 3.4.3 | Create unit tests for each agent | `backend/tests/test_*_agent.py` | [x] |

**Note:** Phase 3.4 completed with comprehensive E2E test suite:
- 23 E2E tests covering workflow structure, routing, node integration, and error handling
- Full pipeline test: Meeting transcript → Task extraction → CRM approval request
- Total MVP tests: 100 (all passing)

### 3.5 MVP Release Verification

| # | Task | Criteria | Status |
|---|------|----------|--------|
| 3.5.1 | E2E Test | Meeting transcript -> HubSpot tasks | [x] |
| 3.5.2 | Approval Flow | Approve/Reject works correctly | [x] |
| 3.5.3 | Error Handling | Graceful failure on invalid input | [x] |

**MVP TEST SUMMARY (2026-01-09):**
```
Phase 3.1 - Meeting Notes Agent:     40 tests ✓
Phase 3.2 - Task Extractor Agent:    8 tests ✓
Phase 3.3 - CRM Updater Agent:       29 tests ✓
Phase 3.4 - E2E Integration:         23 tests ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL MVP TESTS:                     100 tests ✓ ALL PASSING
```

---

## POST-MVP PHASES

### Phase 4: Sales Ops Complete

| # | Task | File | Status |
|---|------|------|--------|
| 4.1 | Tavily search integration | `backend/services/tavily_service.py` | [x] |
| 4.2 | Lead Research agent | `backend/graph/nodes.py:lead_research_node` | [x] |
| 4.3 | Email Copilot agent | `backend/graph/nodes.py:email_copilot_node` | [x] |

**Note:** Phase 4.1 completed with comprehensive Tavily service:
- TavilyService with web search, news search, and company research
- Rate limiting (token bucket) and caching (5 min TTL)
- Pydantic schemas: SearchResult, SearchResponse, CompanyResearchResult, NewsSearchResult
- Error handling: TavilyError, RateLimitError, AuthenticationError, QuotaExceededError
- 38 unit tests all passing

**Note:** Phase 4.2 completed with Lead Research agent:
- lead_research_node with Tavily + LLM integration
- Pydantic schemas: LeadResearchInput, LeadResearchResult, CompanyOverview, FundingInfo, KeyPerson, etc.
- Lead qualification scoring (HOT/WARM/COLD/UNQUALIFIED)
- Business signals, pain points, and talking points extraction
- Fallback mode when Tavily or LLM fails
- 30 unit tests all passing

**Note:** Phase 4.3 completed with Email Copilot agent:
- email_copilot_node: Generates personalized sales emails (cold outreach, follow-up, meeting request, post-meeting)
- email_delivery_node: Sends emails via adapter (HubSpot MVP)
- Pydantic schemas: EmailCopilotInput, EmailGenerationResult, EmailDeliveryPayload, EmailApprovalPayload, etc.
- EmailContextBuilder: RAG-ready context aggregation (brandvoice placeholder for Phase 4.5)
- Email adapter pattern: HubSpotEmailAdapter, MockEmailAdapter, extensible for Salesforce/SMTP
- Every email requires HITL approval
- 42 unit tests all passing
- Phase 4 (Sales Ops Complete) is now 100% finished

### Phase 5: Intelligence Squad

| # | Task | File | Status |
|---|------|------|--------|
| 5.1 | Market Research agent | `backend/agents/intelligence/market_research.py` | [ ] |
| 5.2 | SEO Analysis agent | `backend/agents/intelligence/seo_analysis.py` | [ ] |
| 5.3 | Web Analysis agent | `backend/agents/intelligence/web_analysis.py` | [ ] |
| 5.4 | Audience Builder agent | `backend/agents/intelligence/audience_builder.py` | [ ] |

### Phase 6: Content Squad

| # | Task | File | Status |
|---|------|------|--------|
| 6.1 | Pipeline Manager agent | `backend/agents/content/pipeline_manager.py` | [ ] |
| 6.2 | Brandvoice Writer agent | `backend/agents/content/brandvoice_writer.py` | [ ] |
| 6.3 | SEO/GEO Optimizer agent | `backend/agents/content/seo_optimizer.py` | [ ] |
| 6.4 | Social Distributor agent | `backend/agents/content/social_distributor.py` | [ ] |
| 6.5 | Web Publisher agent | `backend/agents/content/web_publisher.py` | [ ] |
| 6.6 | Vector service for RAG | `backend/services/vector_service.py` | [ ] |

### Phase 7: Frontend & DevOps

| # | Task | File | Status |
|---|------|------|--------|
| 7.1 | Next.js project setup | `frontend/package.json` | [ ] |
| 7.2 | Dashboard layout | `frontend/src/app/layout.tsx` | [ ] |
| 7.3 | Workflows page | `frontend/src/app/workflows/page.tsx` | [ ] |
| 7.4 | Approvals page | `frontend/src/app/approvals/page.tsx` | [ ] |
| 7.5 | Realtime subscriptions | `frontend/src/hooks/useRealtime.ts` | [ ] |
| 7.6 | Production Docker setup | `docker-compose.prod.yml` | [ ] |
| 7.7 | CI/CD pipeline | `.github/workflows/ci.yml` | [ ] |

---

## File Structure (Current State)

```
crm-ai-orchestrator/
+-- backend/
|   +-- app/
|   |   +-- __init__.py
|   |   +-- main.py                         # [x] FastAPI entry point
|   |   +-- core/
|   |   |   +-- __init__.py
|   |   |   +-- config.py                   # [x] Pydantic Settings
|   |   |   +-- dependencies.py             # [x] DI setup
|   |   +-- api/
|   |   |   +-- __init__.py
|   |   |   +-- v1/
|   |   |       +-- __init__.py
|   |   |       +-- workflow.py             # [x] Workflow endpoints
|   |   |       +-- approvals.py            # [x] Approval endpoints
|   |   |       +-- oauth.py                # [x] OAuth flow endpoints
|   |   +-- schemas/
|   |       +-- __init__.py                 # [x] Schema exports
|   |       +-- base.py                     # [x] Base Pydantic models
|   |       +-- workflow.py                 # [x] Workflow schemas
|   |       +-- approvals.py                # [x] Approval schemas (with risk assessment)
|   |       +-- hubspot.py                  # [x] HubSpot entity schemas
|   |       +-- meeting_notes.py            # [x] Meeting analysis schemas (Phase 3.1)
|   |       +-- tasks.py                    # [x] Task extraction schemas (Phase 3.2)
|   |       +-- crm_updates.py              # [x] CRM update schemas (Phase 3.3)
|   +-- agents/
|   |   +-- __init__.py
|   |   +-- base.py                         # [x] BaseAgent ABC
|   +-- graph/
|   |   +-- __init__.py
|   |   +-- state.py                        # [x] OrchestratorState TypedDict
|   |   +-- workflow.py                     # [x] StateGraph definitions
|   |   +-- nodes.py                        # [x] ALL agent node functions
|   |   |   +-- meeting_notes_node()        # [x] Phase 3.1 - Complete with LLM
|   |   |   +-- task_extractor_node()       # [x] Phase 3.2 - Complete with LLM
|   |   |   +-- crm_updater_node()          # [x] Phase 3.3 - Complete with LLM
|   |   |   +-- (other squad nodes)         # [ ] Stubs only
|   |   +-- routers.py                      # [x] Conditional routing logic
|   +-- services/
|   |   +-- __init__.py
|   |   +-- supabase_client.py              # [x] Async Supabase client
|   |   +-- llm_service.py                  # [x] Claude/GPT with structured output
|   |   +-- approval_service.py             # [x] Approval CRUD operations
|   |   +-- workflow_service.py             # [x] Workflow orchestration
|   |   +-- hubspot_service.py              # [x] Multi-tenant HubSpot client
|   |   +-- encryption_service.py           # [x] AES-256 token encryption
|   |   +-- oauth_service.py                # [x] OAuth flow management
|   +-- tools/
|   |   +-- __init__.py
|   |   +-- hubspot_tools.py                # [x] LangChain tools for HubSpot
|   +-- prompts/
|   |   +-- __init__.py
|   |   +-- base.py                         # [x] PromptManager with Jinja2
|   |   +-- templates/
|   |       +-- sales_ops.yaml              # [x] Meeting notes prompt (v2.0)
|   +-- migrations/
|   |   +-- 001_initial_schema.sql          # [x] Core tables
|   |   +-- 002_crm_credentials.sql         # [x] Multi-tenant credentials
|   +-- tests/
|   |   +-- __init__.py
|   |   +-- conftest.py                     # [x] Pytest fixtures
|   |   +-- test_hubspot_service.py         # [x] HubSpot tests (Phase 2.4.1)
|   |   +-- test_approval_api.py            # [x] Approval tests (Phase 2.4.2)
|   |   +-- test_workflow_api.py            # [x] Workflow tests (Phase 2.4.3)
|   |   +-- test_meeting_notes_agent.py     # [x] Meeting notes tests (Phase 3.1)
|   |   +-- test_task_extractor_agent.py    # [x] Task extractor tests (Phase 3.2)
|   |   +-- test_crm_updater_agent.py       # [x] CRM updater tests (Phase 3.3)
|   +-- Dockerfile                          # [x] Python container
|   +-- requirements.txt                    # [x] Dependencies (includes nest-asyncio)
|   +-- pytest.ini                          # [x] Pytest config with asyncio
+-- frontend/                               # [ ] Not started (Phase 7)
+-- infrastructure/                         # [ ] Not started
+-- docker-compose.yml                      # [x] Dev environment
+-- .env.example                            # [x] Environment template
+-- ARCHITECTURE_AND_TODO.md                # [x] This file
+-- README.md                               # [x] Project readme
```

**Legend:**
- `[x]` = Implemented and tested
- `[ ]` = Not yet implemented (stub or placeholder)

---

## Quick Reference

### Environment Variables Required

```bash
# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# LLM Providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...  # Optional fallback

# HubSpot
HUBSPOT_ACCESS_TOKEN=pat-na1-...

# Search (Post-MVP)
TAVILY_API_KEY=tvly-...

# Observability (Optional)
LANGSMITH_API_KEY=ls-...
LANGSMITH_PROJECT=crm-ai-orchestrator

# App Config
DEBUG=true
LOG_LEVEL=INFO
```

### Key Commands

```bash
# Development
docker-compose up -d              # Start services
docker-compose logs -f backend    # View logs
docker-compose down               # Stop services

# Testing
pytest backend/tests/ -v          # Run all tests
pytest backend/tests/test_agents.py -v  # Run agent tests

# Database
# Run migrations via Supabase dashboard or CLI
```

---

> Last Updated: 2026-01-09
> Current Phase: Phase 3 - MVP Agents (3.1-3.3 Complete, MVP Ready for E2E Testing)

---

## Progress Summary

### Completed Phases

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| Phase 1 | Foundation (FastAPI, LangGraph, Supabase) | ✅ Complete | ✅ 3/3 |
| Phase 2 | Services (HubSpot, Approvals, Workflow API) | ✅ Complete | ✅ 15/15 |
| Phase 3.1 | Meeting Notes Analyzer Agent | ✅ Complete | ✅ 21/21 |
| Phase 3.2 | Task Extractor Agent | ✅ Complete | ✅ 27/27 |
| Phase 3.3 | CRM Updater Agent | ✅ Complete | ✅ 29/29 |

### Current Phase: 3.4-3.5 Integration & E2E Testing

**Next Steps:**
1. Create Sales Ops squad subgraph integration
2. Create end-to-end workflow test (Meeting → Tasks → CRM Updates)
3. Test HITL approval flow end-to-end
4. Verify error handling across the pipeline

### Test Coverage

```
Total Tests: 92 (Phase 2.4: 15, Phase 3.1: 21, Phase 3.2: 27, Phase 3.3: 29)
All Passing: ✅ Yes
```

### Key Implementation Decisions

1. **Agent Pattern**: LangGraph nodes in `nodes.py` instead of separate agent files
2. **Prompt Storage**: YAML templates with Jinja2 (not separate .py files)
3. **Structured Output**: Using `llm.with_structured_output(PydanticModel)`
4. **Async in Sync**: Using `nest-asyncio` for event loop compatibility
5. **Input Architecture**: Extensible adapter pattern for future input sources
6. **Task Payload Structure**: Nested `hubspot_task` for HubSpot-ready payloads
7. **Risk Assessment**: ADR-014 compliant risk scoring (low/medium/high) for CRM operations
8. **Fallback Mode**: Graceful degradation when LLM fails - basic approval still created
