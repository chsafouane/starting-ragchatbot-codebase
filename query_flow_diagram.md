# RAG System Query Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant FastAPI as FastAPI Server<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant Session as Session Manager<br/>(session_manager.py)
    participant AI as AI Generator<br/>(ai_generator.py)
    participant Tools as Tool Manager<br/>(search_tools.py)
    participant Vector as Vector Store<br/>(vector_store.py)
    participant ChromaDB as ChromaDB<br/>(Embeddings)

    %% User initiates query
    User->>Frontend: Types query & clicks send
    Frontend->>Frontend: Disable input, show loading
    
    %% API Request
    Frontend->>+FastAPI: POST /api/query<br/>{query, session_id}
    
    %% Session handling
    FastAPI->>Session: Create session if needed
    Session-->>FastAPI: session_id
    
    %% RAG orchestration
    FastAPI->>+RAG: query(query, session_id)
    RAG->>Session: get_conversation_history()
    Session-->>RAG: Previous messages context
    
    %% AI processing with tools
    RAG->>+AI: generate_response()<br/>(query, history, tools)
    AI->>AI: Build system prompt<br/>with tool instructions
    AI->>AI: Send to Claude API<br/>with available tools
    
    %% Tool execution (if Claude decides to search)
    Note over AI,ChromaDB: Claude decides to use search_course_content tool
    AI->>+Tools: execute_tool("search_course_content",<br/>query, course_name, lesson_number)
    
    %% Vector search
    Tools->>+Vector: search(query, course_name, lesson_number)
    Vector->>Vector: Resolve course name<br/>if provided
    Vector->>Vector: Build ChromaDB filters
    Vector->>+ChromaDB: query(embeddings, filters)
    ChromaDB-->>-Vector: Matching documents<br/>+ metadata + distances
    Vector-->>-Tools: SearchResults with<br/>documents & metadata
    
    %% Format and track results
    Tools->>Tools: Format results with<br/>course/lesson context
    Tools->>Tools: Store sources in<br/>last_sources[]
    Tools-->>-AI: Formatted search results
    
    %% Final AI response
    AI->>AI: Send search results<br/>back to Claude
    AI->>AI: Generate final answer<br/>based on search results
    AI-->>-RAG: Final response text
    
    %% Session and source management
    RAG->>Tools: get_last_sources()
    Tools-->>RAG: Source list for UI
    RAG->>Tools: reset_sources()
    RAG->>Session: add_exchange(query, response)
    RAG-->>-FastAPI: (response, sources)
    
    %% API Response
    FastAPI-->>-Frontend: JSON Response<br/>{answer, sources, session_id}
    
    %% UI Updates
    Frontend->>Frontend: Remove loading animation
    Frontend->>Frontend: Add assistant message<br/>(with markdown rendering)
    Frontend->>Frontend: Show sources in<br/>collapsible section
    Frontend->>Frontend: Re-enable input controls
    Frontend->>User: Display formatted response<br/>with sources
```

## Architecture Components

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Web Interface<br/>HTML/CSS/JS]
        Chat[Chat Component<br/>Messages & Input]
        Stats[Course Statistics<br/>Sidebar]
    end
    
    subgraph "API Layer"
        FastAPI[FastAPI Server<br/>CORS & Static Files]
        Endpoints["/api/query<br/>/api/courses"]
    end
    
    subgraph "RAG System Core"
        RAG[RAG System<br/>Main Orchestrator]
        Session[Session Manager<br/>Conversation History]
        AI[AI Generator<br/>Claude API Interface]
    end
    
    subgraph "Search & Tools"
        Tools[Tool Manager<br/>Tool Registration]
        Search[Course Search Tool<br/>Query Processing]
        Vector[Vector Store<br/>ChromaDB Interface]
    end
    
    subgraph "Data Processing"
        Processor[Document Processor<br/>Text Chunking]
        Models[Data Models<br/>Course/Lesson/Chunk]
    end
    
    subgraph "Storage Layer"
        ChromaDB[(ChromaDB<br/>Vector Embeddings)]
        Embeddings[Sentence Transformers<br/>Text Embeddings]
        Docs[Course Documents<br/>TXT Files]
    end
    
    %% Connections
    UI --> FastAPI
    Chat --> FastAPI
    Stats --> FastAPI
    
    FastAPI --> RAG
    RAG --> Session
    RAG --> AI
    RAG --> Tools
    
    AI --> Tools
    Tools --> Search
    Search --> Vector
    Vector --> ChromaDB
    Vector --> Embeddings
    
    Processor --> Models
    Processor --> Vector
    Docs --> Processor
    
    %% Styling
    classDef frontend fill:#e1f5fe
    classDef api fill:#f3e5f5  
    classDef core fill:#e8f5e8
    classDef search fill:#fff3e0
    classDef data fill:#fce4ec
    classDef storage fill:#f1f8e9
    
    class UI,Chat,Stats frontend
    class FastAPI,Endpoints api
    class RAG,Session,AI core
    class Tools,Search,Vector search
    class Processor,Models data
    class ChromaDB,Embeddings,Docs storage
```

## Data Flow Summary

1. **User Input** → Frontend captures and validates query
2. **HTTP Request** → POST to `/api/query` with query and session
3. **Session Management** → Create/retrieve conversation context  
4. **RAG Orchestration** → Coordinate AI generation with tools
5. **AI Processing** → Claude decides whether to use search tools
6. **Tool Execution** → Search course content with filters
7. **Vector Search** → Query ChromaDB embeddings for relevant chunks
8. **Result Processing** → Format results with course/lesson context
9. **Response Generation** → Claude synthesizes final answer
10. **Source Tracking** → Collect sources for UI display
11. **Session Update** → Store Q&A exchange in conversation history
12. **Frontend Display** → Render markdown response with sources

## Key Features

- **Tool-Based Search**: AI decides when and how to search
- **Session Persistence**: Maintains conversation context
- **Smart Filtering**: Course name resolution and lesson filtering  
- **Source Attribution**: Tracks which materials were used
- **Contextual Chunking**: Preserves course/lesson context in search results
- **Markdown Rendering**: Rich text formatting in responses
- **Error Handling**: Graceful failure recovery at each layer