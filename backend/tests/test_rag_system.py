import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk

class TestRAGSystem:
    """Test suite for RAG System integration"""

    @pytest.fixture
    def rag_system(self, test_config):
        """Create RAG system with test configuration"""
        return RAGSystem(test_config)

    @pytest.fixture
    def sample_document_path(self, tmp_path):
        """Create a sample course document for testing"""
        doc_content = """Course Title: Test Course: AI Fundamentals
Course Link: https://example.com/ai-fundamentals
Course Instructor: Test Instructor

Lesson 0: Introduction to AI
Lesson Link: https://example.com/lesson/0
This is an introduction to artificial intelligence. AI is a fascinating field that involves creating systems that can perform tasks typically requiring human intelligence.

Lesson 1: Machine Learning Basics
Lesson Link: https://example.com/lesson/1  
Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.

Lesson 2: Neural Networks
Lesson Link: https://example.com/lesson/2
Neural networks are computing systems inspired by biological neural networks. They form the foundation of deep learning."""
        
        doc_path = tmp_path / "test_course.txt"
        doc_path.write_text(doc_content)
        return str(doc_path)

    def test_initialization(self, rag_system, test_config):
        """Test RAG system initialization"""
        assert rag_system.config == test_config
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

    def test_add_course_document_success(self, rag_system, sample_document_path):
        """Test successfully adding a course document"""
        course, chunk_count = rag_system.add_course_document(sample_document_path)
        
        assert course is not None
        assert course.title == "Test Course: AI Fundamentals"
        assert course.instructor == "Test Instructor"
        assert course.course_link == "https://example.com/ai-fundamentals"
        assert len(course.lessons) == 3
        assert chunk_count > 0
        
        # Verify course was added to vector store
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 1
        assert course.title in analytics["course_titles"]

    def test_add_course_document_nonexistent_file(self, rag_system):
        """Test adding non-existent course document"""
        course, chunk_count = rag_system.add_course_document("nonexistent_file.txt")
        
        assert course is None
        assert chunk_count == 0

    def test_add_course_folder_success(self, rag_system, tmp_path):
        """Test adding course documents from folder"""
        # Create test documents
        doc1_content = """Course Title: Course 1
Course Instructor: Instructor 1
Lesson 0: Lesson 1
Content for course 1"""
        
        doc2_content = """Course Title: Course 2  
Course Instructor: Instructor 2
Lesson 0: Lesson 1
Content for course 2"""
        
        (tmp_path / "course1.txt").write_text(doc1_content)
        (tmp_path / "course2.txt").write_text(doc2_content)
        
        courses_added, chunks_added = rag_system.add_course_folder(str(tmp_path))
        
        assert courses_added == 2
        assert chunks_added > 0
        
        # Verify courses were added
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 2
        assert "Course 1" in analytics["course_titles"]
        assert "Course 2" in analytics["course_titles"]

    def test_add_course_folder_nonexistent(self, rag_system):
        """Test adding from non-existent folder"""
        courses_added, chunks_added = rag_system.add_course_folder("nonexistent_folder")
        
        assert courses_added == 0
        assert chunks_added == 0

    def test_add_course_folder_with_clear(self, rag_system, tmp_path, sample_document_path):
        """Test adding course folder with clear existing flag"""
        # First add a course
        rag_system.add_course_document(sample_document_path)
        assert rag_system.get_course_analytics()["total_courses"] == 1
        
        # Create new folder with different course
        new_doc = """Course Title: New Course
Course Instructor: New Instructor  
Lesson 0: New Lesson
New content"""
        
        (tmp_path / "new_course.txt").write_text(new_doc)
        
        # Add folder with clear flag
        courses_added, chunks_added = rag_system.add_course_folder(str(tmp_path), clear_existing=True)
        
        assert courses_added == 1
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 1
        assert "New Course" in analytics["course_titles"]
        assert "Test Course: AI Fundamentals" not in analytics["course_titles"]

    @patch('rag_system.AIGenerator')
    def test_query_without_session(self, mock_ai_generator_class, rag_system, sample_document_path):
        """Test querying without session ID"""
        # Add course data
        rag_system.add_course_document(sample_document_path)
        
        # Mock AI generator response
        mock_ai_generator = Mock()
        mock_ai_generator.generate_response.return_value = "AI is a fascinating field of study."
        mock_ai_generator_class.return_value = mock_ai_generator
        
        # Mock search tool sources
        rag_system.tool_manager.get_last_sources = Mock(return_value=[{
            "display_text": "Test Course - Lesson 0",
            "url": "https://example.com/lesson/0"
        }])
        
        response, sources = rag_system.query("What is artificial intelligence?")
        
        assert response == "AI is a fascinating field of study."
        assert len(sources) == 1
        assert sources[0]["display_text"] == "Test Course - Lesson 0"
        
        # Verify AI generator was called with tools
        mock_ai_generator.generate_response.assert_called_once()
        call_args = mock_ai_generator.generate_response.call_args
        assert "tools" in call_args.kwargs
        assert "tool_manager" in call_args.kwargs

    @patch('rag_system.AIGenerator')
    def test_query_with_session(self, mock_ai_generator_class, rag_system, sample_document_path):
        """Test querying with session ID"""
        # Add course data
        rag_system.add_course_document(sample_document_path)
        
        # Mock AI generator
        mock_ai_generator = Mock()
        mock_ai_generator.generate_response.return_value = "Machine learning is a subset of AI."
        mock_ai_generator_class.return_value = mock_ai_generator
        
        # Mock search tool sources
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        
        # Create session
        session_id = rag_system.session_manager.create_session()
        
        response, sources = rag_system.query("Tell me about machine learning", session_id)
        
        assert response == "Machine learning is a subset of AI."
        
        # Verify conversation history was used
        call_args = mock_ai_generator.generate_response.call_args
        assert call_args.kwargs.get("conversation_history") is not None
        
        # Verify session was updated
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert "Tell me about machine learning" in history
        assert "Machine learning is a subset of AI." in history

    @patch('rag_system.AIGenerator')
    def test_query_api_error(self, mock_ai_generator_class, rag_system):
        """Test query handling when AI generator fails"""
        # Mock AI generator to raise exception
        mock_ai_generator = Mock()
        mock_ai_generator.generate_response.side_effect = Exception("API key invalid")
        mock_ai_generator_class.return_value = mock_ai_generator
        
        # Query should raise the exception
        with pytest.raises(Exception) as exc_info:
            rag_system.query("test query")
        
        assert "API key invalid" in str(exc_info.value)

    def test_get_course_analytics_empty(self, rag_system):
        """Test course analytics with no courses"""
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 0
        assert analytics["course_titles"] == []

    def test_get_course_analytics_with_data(self, rag_system, sample_document_path):
        """Test course analytics with course data"""
        rag_system.add_course_document(sample_document_path)
        
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 1
        assert len(analytics["course_titles"]) == 1
        assert "Test Course: AI Fundamentals" in analytics["course_titles"]

    @patch('rag_system.AIGenerator')
    def test_tool_manager_integration(self, mock_ai_generator_class, rag_system, sample_document_path):
        """Test integration with tool manager and search tools"""
        # Add course data
        rag_system.add_course_document(sample_document_path)
        
        # Mock AI generator to simulate tool use
        mock_ai_generator = Mock()
        mock_ai_generator.generate_response.return_value = "Based on the course content, AI involves..."
        mock_ai_generator_class.return_value = mock_ai_generator
        
        # Mock tool manager behavior
        original_execute_tool = rag_system.tool_manager.execute_tool
        rag_system.tool_manager.execute_tool = Mock(side_effect=original_execute_tool)
        
        response, sources = rag_system.query("What is artificial intelligence?")
        
        # Verify tool manager was passed to AI generator
        call_args = mock_ai_generator.generate_response.call_args
        assert call_args.kwargs["tool_manager"] == rag_system.tool_manager
        
        # Verify tools were provided
        tools = call_args.kwargs["tools"]
        tool_names = [tool["name"] for tool in tools]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_source_tracking_and_reset(self, rag_system, sample_document_path):
        """Test source tracking and reset functionality"""
        # Add course data
        rag_system.add_course_document(sample_document_path)
        
        # Manually execute search tool to generate sources
        result = rag_system.search_tool.execute("artificial intelligence")
        assert len(rag_system.search_tool.last_sources) > 0
        
        # Get sources through tool manager
        sources = rag_system.tool_manager.get_last_sources()
        assert len(sources) > 0
        
        # Reset sources
        rag_system.tool_manager.reset_sources()
        
        # Sources should be cleared
        sources_after_reset = rag_system.tool_manager.get_last_sources()
        assert len(sources_after_reset) == 0

    def test_conversation_flow_multiple_exchanges(self, rag_system):
        """Test multiple conversation exchanges in a session"""
        session_id = rag_system.session_manager.create_session()
        
        # Mock AI responses
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
            mock_generate.side_effect = [
                "Hello! I'm here to help with course materials.",
                "AI is the simulation of human intelligence in machines.",
                "Machine learning is a method of data analysis."
            ]
            
            # First exchange
            response1, _ = rag_system.query("Hello", session_id)
            assert response1 == "Hello! I'm here to help with course materials."
            
            # Second exchange
            response2, _ = rag_system.query("What is AI?", session_id)
            assert response2 == "AI is the simulation of human intelligence in machines."
            
            # Third exchange
            response3, _ = rag_system.query("What about machine learning?", session_id)
            assert response3 == "Machine learning is a method of data analysis."
            
            # Verify conversation history includes all exchanges
            history = rag_system.session_manager.get_conversation_history(session_id)
            assert "Hello" in history
            assert "What is AI?" in history
            assert "What about machine learning?" in history
            assert "Hello! I'm here to help" in history
            assert "AI is the simulation" in history
            assert "Machine learning is a method" in history

    def test_error_handling_in_document_processing(self, rag_system, tmp_path):
        """Test error handling during document processing"""
        # Create invalid document
        invalid_doc = tmp_path / "invalid.txt"
        invalid_doc.write_text("Invalid content without proper headers")
        
        course, chunk_count = rag_system.add_course_document(str(invalid_doc))
        
        # Should handle gracefully
        assert course is None or chunk_count == 0

    @patch('rag_system.AIGenerator')
    def test_empty_course_catalog_behavior(self, mock_ai_generator_class, rag_system):
        """Test system behavior with no courses loaded"""
        # Mock AI generator
        mock_ai_generator = Mock()
        mock_ai_generator.generate_response.return_value = "I don't have access to that information."
        mock_ai_generator_class.return_value = mock_ai_generator
        
        # Query with no courses loaded
        response, sources = rag_system.query("Tell me about AI")
        
        # Should still work, though search tool will return empty results
        assert isinstance(response, str)
        assert isinstance(sources, list)
        
        # Verify AI generator was still called with tools
        call_args = mock_ai_generator.generate_response.call_args
        assert "tools" in call_args.kwargs

    def test_duplicate_course_handling(self, rag_system, tmp_path):
        """Test handling of duplicate course additions"""
        # Create course document
        doc_content = """Course Title: Duplicate Course
Course Instructor: Test Instructor
Lesson 0: Test Lesson
Test content"""
        
        doc_path = tmp_path / "duplicate.txt"
        doc_path.write_text(doc_content)
        
        # Add course twice
        course1, chunks1 = rag_system.add_course_document(str(doc_path))
        course2, chunks2 = rag_system.add_course_document(str(doc_path))
        
        # First addition should succeed
        assert course1 is not None
        assert chunks1 > 0
        
        # Second addition might succeed or be skipped depending on implementation
        # The important thing is it doesn't crash
        assert course2 is not None or chunks2 == 0

    @patch('rag_system.AIGenerator')  
    def test_session_management_integration(self, mock_ai_generator_class, rag_system):
        """Test session management integration with queries"""
        # Mock AI generator
        mock_ai_generator = Mock()
        mock_ai_generator.generate_response.return_value = "Test response"
        mock_ai_generator_class.return_value = mock_ai_generator
        
        # Query without session - should create one automatically via session manager
        response, sources = rag_system.query("test query")
        
        # Should work without errors
        assert response == "Test response"
        
        # Query with explicit session
        session_id = rag_system.session_manager.create_session()
        response2, sources2 = rag_system.query("another query", session_id)
        
        assert response2 == "Test response"
        
        # Verify session has conversation history
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert "another query" in history
        assert "Test response" in history