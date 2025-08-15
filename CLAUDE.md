# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000

# Install dependencies first if needed
uv sync
```

### Environment Setup
Create `.env` file in root directory:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Accessing the Application
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) chatbot system** for course materials with a tool-based search architecture.

### Core System Flow
1. **Document Processing**: Course documents (`/docs/*.txt`) are parsed into structured lessons and chunked for vector storage
2. **Query Processing**: User queries trigger a FastAPI endpoint that orchestrates the RAG pipeline
3. **Tool-Based Search**: Claude AI uses `search_course_content` tool to query ChromaDB when needed
4. **Response Generation**: Claude synthesizes answers from search results and conversation context
5. **Session Management**: Conversation history is maintained per session with configurable limits

### Key Architectural Patterns

**Tool-Based AI Architecture**: The system uses Anthropic's tool calling where Claude decides when to search course content based on the query type. This is implemented through:
- `search_tools.py`: Defines the `search_course_content` tool interface
- `ai_generator.py`: Handles tool execution flow with Claude API
- `tool_manager.py`: Manages tool registration and execution

**Document Processing Pipeline**: Course documents follow a specific format and are processed through:
- **Metadata Extraction**: Parses course title, link, and instructor from document headers
- **Lesson Segmentation**: Identifies lesson boundaries using `"Lesson N:"` markers
- **Context-Aware Chunking**: Creates overlapping text chunks with preserved course/lesson context
- **Vector Storage**: Stores chunks in ChromaDB with metadata for filtering

**Session-Based Conversation**: Each user session maintains conversation history through:
- `session_manager.py`: Stores message history per session ID
- Configurable memory limits (default: 2 exchanges = 4 messages)
- Context injection into AI prompts for conversational continuity

### Critical Configuration (backend/config.py)
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks
- `MAX_RESULTS: 5` - Maximum search results returned
- `MAX_HISTORY: 2` - Number of conversation exchanges to remember
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"` - Claude model version
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"` - Sentence transformer for embeddings

### Data Models and Processing

**Document Structure Expected**:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [lesson title]
Lesson Link: [lesson url]
[lesson content...]

Lesson 1: [next lesson]
...
```

**Key Data Models** (`models.py`):
- `Course`: Contains title, link, instructor, and list of lessons
- `Lesson`: Has lesson number, title, and optional link
- `CourseChunk`: Text chunk with course/lesson metadata for vector storage

### Vector Store Architecture

**Two-Collection System** (`vector_store.py`):
- `course_catalog`: Stores course metadata (titles, instructors) for course name resolution
- `course_content`: Stores actual lesson content chunks with embeddings

**Smart Search Features**:
- Course name resolution (partial matches work: "MCP" finds "MCP: Build Rich-Context AI Apps")
- Lesson number filtering for specific lesson queries
- Metadata-preserved search results with course/lesson context

### Frontend-Backend Integration

**API Contract**:
- `POST /api/query`: Processes user queries with optional session context
- `GET /api/courses`: Returns course statistics and titles
- Response includes `answer`, `sources`, and `session_id`

**Frontend Features** (`frontend/`):
- Session persistence across page reloads
- Source attribution display in collapsible sections
- Markdown rendering for AI responses
- Course statistics sidebar with real-time data

### ChromaDB Setup and Persistence
- Database path: `./backend/chroma_db/` (created automatically)
- Collections are created on first run with sentence transformer embeddings
- Documents are loaded automatically from `/docs/` folder on server startup
- Duplicate course detection prevents re-processing existing content

### Development Notes
- No test framework is currently configured
- No linting tools are set up in the project
- The system uses `uv` package manager instead of pip
- Static files are served directly by FastAPI with development no-cache headers
- CORS is configured to allow all origins for development
- Always use uv to run the server. Do not use pip directly
- Make sure to use uv to manage all dependencies
- Use uv to run python files