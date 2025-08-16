import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
import sys

# Add backend directory to Python path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from vector_store import VectorStore
from document_processor import DocumentProcessor
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk

@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config(temp_chroma_path):
    """Create a test configuration"""
    config = Config()
    config.CHROMA_PATH = temp_chroma_path
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.MAX_RESULTS = 3
    return config

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock a successful text response
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a test response"
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    return mock_client

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)
    
    # Mock search results
    from vector_store import SearchResults
    mock_results = SearchResults(
        documents=["Test content from course 1", "Test content from course 2"],
        metadata=[
            {"course_title": "Test Course", "lesson_number": 1},
            {"course_title": "Test Course", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )
    mock_store.search.return_value = mock_results
    mock_store.get_lesson_link.return_value = "https://example.com/lesson/1"
    
    return mock_store

@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course: Introduction to AI",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson/0"),
            Lesson(lesson_number=1, title="Basics", lesson_link="https://example.com/lesson/1"),
            Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson/2")
        ]
    )

@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is the introduction to artificial intelligence.",
            course_title="Test Course: Introduction to AI",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Here we cover the basic concepts of machine learning.",
            course_title="Test Course: Introduction to AI",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced topics include deep learning and neural networks.",
            course_title="Test Course: Introduction to AI",
            lesson_number=2,
            chunk_index=2
        )
    ]

@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create a CourseSearchTool with mocked dependencies"""
    return CourseSearchTool(mock_vector_store)

@pytest.fixture
def real_vector_store(test_config):
    """Create a real vector store for integration testing"""
    return VectorStore(
        chroma_path=test_config.CHROMA_PATH,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS
    )