import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults

class TestCourseSearchTool:
    """Test suite for CourseSearchTool functionality"""

    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is correctly structured"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_execute_basic_query_success(self, course_search_tool, mock_vector_store):
        """Test successful basic query execution"""
        # Setup mock to return successful results
        mock_results = SearchResults(
            documents=["Test content about AI fundamentals"],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"
        
        result = course_search_tool.execute("artificial intelligence")
        
        # Verify search was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="artificial intelligence",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result format
        assert "[AI Course - Lesson 1]" in result
        assert "Test content about AI fundamentals" in result
        assert len(course_search_tool.last_sources) == 1
        assert course_search_tool.last_sources[0]["display_text"] == "AI Course - Lesson 1"
        assert course_search_tool.last_sources[0]["url"] == "https://example.com/lesson/1"

    def test_execute_with_course_name(self, course_search_tool, mock_vector_store):
        """Test query execution with course name filter"""
        mock_results = SearchResults(
            documents=["Course content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": None}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None
        
        result = course_search_tool.execute("test query", course_name="Specific Course")
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Specific Course",
            lesson_number=None
        )
        
        assert "[Specific Course]" in result
        assert "Course content" in result

    def test_execute_with_lesson_number(self, course_search_tool, mock_vector_store):
        """Test query execution with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/3"
        
        result = course_search_tool.execute("lesson content", lesson_number=3)
        
        mock_vector_store.search.assert_called_once_with(
            query="lesson content",
            course_name=None,
            lesson_number=3
        )
        
        assert "[Test Course - Lesson 3]" in result
        assert "Lesson content" in result

    def test_execute_with_both_filters(self, course_search_tool, mock_vector_store):
        """Test query execution with both course name and lesson number"""
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "Target Course", "lesson_number": 2}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/2"
        
        result = course_search_tool.execute(
            "specific content", 
            course_name="Target Course", 
            lesson_number=2
        )
        
        mock_vector_store.search.assert_called_once_with(
            query="specific content",
            course_name="Target Course",
            lesson_number=2
        )
        
        assert "[Target Course - Lesson 2]" in result
        assert "Specific lesson content" in result

    def test_execute_empty_results_no_filters(self, course_search_tool, mock_vector_store):
        """Test handling of empty search results without filters"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = course_search_tool.execute("nonexistent content")
        
        assert result == "No relevant content found."

    def test_execute_empty_results_with_course_filter(self, course_search_tool, mock_vector_store):
        """Test handling of empty search results with course filter"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = course_search_tool.execute("content", course_name="Missing Course")
        
        assert "No relevant content found in course 'Missing Course'." in result

    def test_execute_empty_results_with_lesson_filter(self, course_search_tool, mock_vector_store):
        """Test handling of empty search results with lesson filter"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = course_search_tool.execute("content", lesson_number=99)
        
        assert "No relevant content found in lesson 99." in result

    def test_execute_empty_results_with_both_filters(self, course_search_tool, mock_vector_store):
        """Test handling of empty search results with both filters"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = course_search_tool.execute("content", course_name="Test", lesson_number=5)
        
        assert "No relevant content found in course 'Test' in lesson 5." in result

    def test_execute_search_error(self, course_search_tool, mock_vector_store):
        """Test handling of search errors"""
        error_results = SearchResults(
            documents=[], 
            metadata=[], 
            distances=[], 
            error="Database connection failed"
        )
        mock_vector_store.search.return_value = error_results
        
        result = course_search_tool.execute("test query")
        
        assert result == "Database connection failed"

    def test_execute_multiple_results(self, course_search_tool, mock_vector_store):
        """Test formatting of multiple search results"""
        mock_results = SearchResults(
            documents=[
                "First result content",
                "Second result content",
                "Third result content"
            ],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
                {"course_title": "Course A", "lesson_number": 3}
            ],
            distances=[0.1, 0.2, 0.3]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/a/1",
            "https://example.com/b/2", 
            "https://example.com/a/3"
        ]
        
        result = course_search_tool.execute("test query")
        
        # Check that all results are included
        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 2]" in result
        assert "[Course A - Lesson 3]" in result
        assert "First result content" in result
        assert "Second result content" in result
        assert "Third result content" in result
        
        # Check sources tracking
        assert len(course_search_tool.last_sources) == 3
        assert course_search_tool.last_sources[0]["display_text"] == "Course A - Lesson 1"
        assert course_search_tool.last_sources[1]["display_text"] == "Course B - Lesson 2"
        assert course_search_tool.last_sources[2]["display_text"] == "Course A - Lesson 3"

    def test_execute_missing_metadata(self, course_search_tool, mock_vector_store):
        """Test handling of results with missing metadata"""
        mock_results = SearchResults(
            documents=["Content with missing metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None
        
        result = course_search_tool.execute("test query")
        
        # Should handle missing metadata gracefully
        assert "[unknown]" in result
        assert "Content with missing metadata" in result

    def test_last_sources_reset_on_new_search(self, course_search_tool, mock_vector_store):
        """Test that last_sources is properly updated on new searches"""
        # First search
        mock_results1 = SearchResults(
            documents=["First search result"],
            metadata=[{"course_title": "Course 1", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results1
        mock_vector_store.get_lesson_link.return_value = "https://example.com/1"
        
        course_search_tool.execute("first query")
        assert len(course_search_tool.last_sources) == 1
        assert course_search_tool.last_sources[0]["display_text"] == "Course 1 - Lesson 1"
        
        # Second search with different results
        mock_results2 = SearchResults(
            documents=["Second search result", "Another result"],
            metadata=[
                {"course_title": "Course 2", "lesson_number": 2},
                {"course_title": "Course 3", "lesson_number": 1}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = mock_results2
        mock_vector_store.get_lesson_link.side_effect = ["https://example.com/2", "https://example.com/3"]
        
        course_search_tool.execute("second query")
        
        # Should have new sources, not accumulated
        assert len(course_search_tool.last_sources) == 2
        assert course_search_tool.last_sources[0]["display_text"] == "Course 2 - Lesson 2"
        assert course_search_tool.last_sources[1]["display_text"] == "Course 3 - Lesson 1"


class TestToolManager:
    """Test suite for ToolManager functionality"""

    def test_register_and_execute_tool(self, mock_vector_store):
        """Test tool registration and execution through manager"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Register tool
        manager.register_tool(search_tool)
        
        # Check tool is registered
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
        
        # Mock successful search
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None
        
        # Execute tool
        result = manager.execute_tool("search_course_content", query="test")
        assert "Test content" in result

    def test_execute_nonexistent_tool(self):
        """Test execution of non-registered tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources(self, mock_vector_store):
        """Test source retrieval from tool manager"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute search to generate sources
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"
        
        manager.execute_tool("search_course_content", query="test")
        
        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["display_text"] == "Test Course - Lesson 1"
        assert sources[0]["url"] == "https://example.com/lesson/1"

    def test_reset_sources(self, mock_vector_store):
        """Test source reset functionality"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Generate some sources
        search_tool.last_sources = [{"display_text": "Test", "url": None}]
        
        # Reset sources
        manager.reset_sources()
        
        # Sources should be cleared
        assert search_tool.last_sources == []
        assert manager.get_last_sources() == []