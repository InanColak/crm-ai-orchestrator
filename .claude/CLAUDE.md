# CRM AI Orchestrator - Project Context

## Project Overview
This project is an end-to-end "Autonomous Growth & Sales Engine" system for HubSpot and Salesforce consulting clients. It autonomously executes 16+ complex tasks from market research to content generation, SEO analysis to CRM updates.

## Tech Stack

### Backend (Python)
- **Orchestration**: LangGraph (Stateful, cyclic multi-agent workflows)
- **API**: FastAPI (Asynchronous, high-performance)
- **Database**: Supabase (PostgreSQL + Vector DB + Storage)
- **CRM SDKs**: HubSpot Python SDK, Salesforce SDK (simple-salesforce)
- **AI Models**: Claude 3.5 Sonnet / GPT-4o via LangChain
- **Search**: Tavily API for real-time web research
- **Observability**: LangSmith for tracing

### Frontend (TypeScript)
- **Framework**: Next.js 14+ (App Router)
- **Styling**: Tailwind CSS
- **Icons**: Lucide Icons
- **Real-time**: Supabase Realtime subscriptions

## Architecture - 3 Agent Squads

### 1. Intelligence Squad (`backend/agents/intelligence/`)
- Market Research Agent
- SEO Analysis Agent (Google/YouTube)
- Web Analysis Agent
- Audience Building Agent

### 2. Content Squad (`backend/agents/content/`)
- Pipeline Manager (Editorial calendar)
- Brandvoice Writer
- SEO/GEO Optimizer
- Social Media Distributor
- Web Publisher

### 3. Sales Ops Squad (`backend/agents/sales_ops/`)
- Lead Research Agent
- Meeting Notes Analyzer
- CRM Updater Agent
- Email Copilot
- Task Extractor

## Critical Files

### State Management (THE CONSTITUTION)
- `backend/graph/state.py` - Central TypedDict, all agents communicate through this state

### Orchestration
- `backend/graph/workflow.py` - LangGraph workflow definitions
- `backend/graph/nodes.py` - Agent node functions

### API Layer
- `backend/app/main.py` - FastAPI entry point
- `backend/app/api/v1/workflow.py` - Workflow trigger endpoints
- `backend/app/api/v1/approvals.py` - Human-in-the-loop approval system

## Development Principles

### 1. State-First Design
All agents exchange data through the `OrchestratorState` TypedDict. State mutation is used instead of direct function calls.

### 2. Human-in-the-Loop
Critical actions (CRM updates, content publishing, email sending) are added to the `pending_approvals` list and await user approval.

### 3. Pydantic Validation
All inputs/outputs are validated with Pydantic schemas. Hallucinations are minimized.

### 4. Observability
Every agent step is traced to LangSmith. `agent_execution_log` is maintained in state.

### 5. RAG-Enabled
Customer documents (Brandvoice PDF, etc.) are stored in Supabase Vector DB and served to agents as context.

## Code Style

- Python: Black formatter, Ruff linter, type hints mandatory
- TypeScript: ESLint + Prettier
- Commits: Conventional Commits (feat:, fix:, docs:, etc.)
- Docstrings: Google style

## Environment Variables
All API keys are kept in the `.env` file. `.env.example` is used as reference.

Required:
- `SUPABASE_URL`, `SUPABASE_KEY`
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
- `HUBSPOT_ACCESS_TOKEN` (obtained via OAuth flow)
- `TAVILY_API_KEY`
- `LANGSMITH_API_KEY` (optional, for tracing)

## Commands
Custom agent commands for project development are defined in the `.claude/commands/` folder:
- `/architect` - Architectural decisions and system design
- `/backend` - FastAPI and Python backend development
- `/frontend` - Next.js frontend development
- `/langgraph` - LangGraph agent development
- `/devops` - Docker, CI/CD, and deployment
- `/qa` - Testing and quality control
