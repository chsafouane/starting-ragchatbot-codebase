import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk

class TestVectorStore:
    """Test suite for VectorStore functionality"""

    def test_initialization(self, temp_chroma_path):
        """Test VectorStore initialization"""
        store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        
        assert store.max_results == 5
        assert store.client is not None
        assert store.embedding_function is not None
        assert store.course_catalog is not None
        assert store.course_content is not None

    def test_add_course_metadata(self, real_vector_store, sample_course):
        """Test adding course metadata to catalog"""
        real_vector_store.add_course_metadata(sample_course)
        
        # Verify course was added
        existing_titles = real_vector_store.get_existing_course_titles()
        assert sample_course.title in existing_titles
        
        # Verify course count
        assert real_vector_store.get_course_count() == 1

    def test_add_course_content(self, real_vector_store, sample_course_chunks):
        """Test adding course content chunks"""
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Verify chunks were added by trying to search
        results = real_vector_store.search("introduction artificial intelligence")
        assert not results.is_empty()
        assert len(results.documents) > 0

    def test_search_basic_query(self, real_vector_store, sample_course, sample_course_chunks):
        """Test basic search functionality"""
        # Add data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search for content
        results = real_vector_store.search("artificial intelligence")
        
        assert not results.is_empty()
        assert results.error is None
        assert len(results.documents) > 0
        assert len(results.metadata) == len(results.documents)
        assert len(results.distances) == len(results.documents)

    def test_search_with_course_filter(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search with course name filtering"""
        # Add data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search with exact course name
        results = real_vector_store.search(
            "machine learning", 
            course_name=sample_course.title
        )
        
        assert not results.is_empty()
        for metadata in results.metadata:
            assert metadata["course_title"] == sample_course.title

    def test_search_with_partial_course_name(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search with partial course name matching"""
        # Add data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search with partial course name
        results = real_vector_store.search(
            "learning", 
            course_name="Introduction to AI"  # Partial match
        )
        
        assert not results.is_empty()
        for metadata in results.metadata:
            assert metadata["course_title"] == sample_course.title

    def test_search_with_lesson_filter(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search with lesson number filtering"""
        # Add data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search for specific lesson
        results = real_vector_store.search(
            "concepts", 
            lesson_number=1
        )
        
        assert not results.is_empty()
        for metadata in results.metadata:
            assert metadata["lesson_number"] == 1

    def test_search_with_both_filters(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search with both course and lesson filters"""
        # Add data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search with both filters
        results = real_vector_store.search(
            "advanced topics", 
            course_name="Test Course",
            lesson_number=2
        )
        
        # Should find the advanced topics content
        assert not results.is_empty()
        for metadata in results.metadata:
            assert metadata["course_title"] == sample_course.title
            assert metadata["lesson_number"] == 2

    def test_search_no_results(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search that returns no results"""
        # Add data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search for non-existent content
        results = real_vector_store.search("quantum computing blockchain")
        
        # May or may not return results depending on semantic similarity
        # But should not error
        assert results.error is None

    def test_search_nonexistent_course(self, real_vector_store, sample_course, sample_course_chunks):
        """Test search with non-existent course name"""
        # Add data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Search with non-existent course
        results = real_vector_store.search(
            "test", 
            course_name="Nonexistent Course"
        )
        
        assert results.error is not None
        assert "No course found matching" in results.error

    def test_resolve_course_name_exact_match(self, real_vector_store, sample_course):
        """Test course name resolution with exact match"""
        real_vector_store.add_course_metadata(sample_course)
        
        resolved = real_vector_store._resolve_course_name(sample_course.title)
        assert resolved == sample_course.title

    def test_resolve_course_name_partial_match(self, real_vector_store, sample_course):
        """Test course name resolution with partial match"""
        real_vector_store.add_course_metadata(sample_course)
        
        # Test various partial matches
        resolved = real_vector_store._resolve_course_name("Introduction")
        assert resolved == sample_course.title
        
        resolved = real_vector_store._resolve_course_name("AI")
        assert resolved == sample_course.title
        
        resolved = real_vector_store._resolve_course_name("Test Course")
        assert resolved == sample_course.title

    def test_resolve_course_name_no_match(self, real_vector_store, sample_course):
        """Test course name resolution with no match"""
        real_vector_store.add_course_metadata(sample_course)
        
        resolved = real_vector_store._resolve_course_name("Completely Different Course")
        # May return None or the closest match depending on similarity threshold
        # The important thing is it doesn't crash

    def test_build_filter_no_filters(self, real_vector_store):
        """Test filter building with no filters"""
        filter_dict = real_vector_store._build_filter(None, None)
        assert filter_dict is None

    def test_build_filter_course_only(self, real_vector_store):
        """Test filter building with course only"""
        filter_dict = real_vector_store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, real_vector_store):
        """Test filter building with lesson only"""  
        filter_dict = real_vector_store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}

    def test_build_filter_both(self, real_vector_store):
        """Test filter building with both filters"""
        filter_dict = real_vector_store._build_filter("Test Course", 2)
        expected = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 2}
            ]
        }
        assert filter_dict == expected

    def test_get_course_link(self, real_vector_store, sample_course):
        """Test retrieving course link"""
        real_vector_store.add_course_metadata(sample_course)
        
        link = real_vector_store.get_course_link(sample_course.title)
        assert link == sample_course.course_link

    def test_get_lesson_link(self, real_vector_store, sample_course):
        """Test retrieving lesson link"""
        real_vector_store.add_course_metadata(sample_course)
        
        link = real_vector_store.get_lesson_link(sample_course.title, 1)
        assert link == "https://example.com/lesson/1"
        
        # Test non-existent lesson
        link = real_vector_store.get_lesson_link(sample_course.title, 99)
        assert link is None

    def test_get_all_courses_metadata(self, real_vector_store, sample_course):
        """Test retrieving all courses metadata"""
        real_vector_store.add_course_metadata(sample_course)
        
        all_metadata = real_vector_store.get_all_courses_metadata()
        assert len(all_metadata) == 1
        
        course_meta = all_metadata[0]
        assert course_meta["title"] == sample_course.title
        assert course_meta["instructor"] == sample_course.instructor
        assert course_meta["course_link"] == sample_course.course_link
        assert "lessons" in course_meta
        assert len(course_meta["lessons"]) == len(sample_course.lessons)

    def test_clear_all_data(self, real_vector_store, sample_course, sample_course_chunks):
        """Test clearing all data"""
        # Add some data
        real_vector_store.add_course_metadata(sample_course)
        real_vector_store.add_course_content(sample_course_chunks)
        
        # Verify data exists
        assert real_vector_store.get_course_count() == 1
        
        # Clear data
        real_vector_store.clear_all_data()
        
        # Verify data is gone
        assert real_vector_store.get_course_count() == 0
        
        # Verify search returns empty
        results = real_vector_store.search("artificial intelligence")
        assert results.is_empty()

    def test_add_empty_chunks_list(self, real_vector_store):
        """Test adding empty chunks list"""
        # Should handle gracefully
        real_vector_store.add_course_content([])
        
        # Should not crash and search should return empty
        results = real_vector_store.search("test")
        assert results.is_empty()

    def test_multiple_courses(self, real_vector_store):
        """Test handling multiple courses"""
        # Create multiple courses
        course1 = Course(
            title="Course 1: Machine Learning",
            course_link="https://example.com/ml",
            instructor="Prof. ML",
            lessons=[Lesson(lesson_number=0, title="ML Intro")]
        )
        
        course2 = Course(
            title="Course 2: Deep Learning", 
            course_link="https://example.com/dl",
            instructor="Prof. DL",
            lessons=[Lesson(lesson_number=0, title="DL Intro")]
        )
        
        chunks1 = [CourseChunk(
            content="Machine learning basics",
            course_title=course1.title,
            lesson_number=0,
            chunk_index=0
        )]
        
        chunks2 = [CourseChunk(
            content="Deep learning fundamentals", 
            course_title=course2.title,
            lesson_number=0,
            chunk_index=0
        )]
        
        # Add both courses
        real_vector_store.add_course_metadata(course1)
        real_vector_store.add_course_metadata(course2)
        real_vector_store.add_course_content(chunks1)
        real_vector_store.add_course_content(chunks2)
        
        # Verify both exist
        assert real_vector_store.get_course_count() == 2
        titles = real_vector_store.get_existing_course_titles()
        assert course1.title in titles
        assert course2.title in titles
        
        # Test course-specific search
        results = real_vector_store.search("basics", course_name="Machine Learning")
        assert not results.is_empty()
        for metadata in results.metadata:
            assert "Machine Learning" in metadata["course_title"]


class TestSearchResults:
    """Test suite for SearchResults class"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key': 'value1'}, {'key': 'value2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
        assert not results.is_empty()

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
        assert results.is_empty()

    def test_from_chroma_missing_data(self):
        """Test creating SearchResults from malformed ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []  
        assert results.distances == []
        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        error_msg = "Database connection failed"
        results = SearchResults.empty(error_msg)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg
        assert results.is_empty()

    def test_is_empty(self):
        """Test is_empty method"""
        # Empty results
        empty_results = SearchResults([], [], [])
        assert empty_results.is_empty()
        
        # Non-empty results
        non_empty_results = SearchResults(['doc'], [{'key': 'value'}], [0.1])
        assert not non_empty_results.is_empty()