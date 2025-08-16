import pytest
from unittest.mock import Mock, MagicMock, patch
from ai_generator import AIGenerator

class TestAIGenerator:
    """Test suite for AIGenerator functionality"""

    def test_initialization(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_basic(self, mock_anthropic_class):
        """Test basic response generation without tools"""
        # Setup mock client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a test response about AI."
        mock_response.stop_reason = "end_turn"
        
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        # Create generator and test
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        result = generator.generate_response("What is artificial intelligence?")
        
        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == "claude-sonnet-4"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"] == [{"role": "user", "content": "What is artificial intelligence?"}]
        assert "You are an AI assistant specialized in course materials" in call_args["system"]
        
        # Verify response
        assert result == "This is a test response about AI."

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test response generation with conversation history"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Follow-up response"
        mock_response.stop_reason = "end_turn"
        
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        history = "User: Hello\nAssistant: Hi there!\nUser: Tell me about AI\nAssistant: AI is..."
        
        result = generator.generate_response(
            "What about machine learning?", 
            conversation_history=history
        )
        
        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]
        assert history in call_args["system"]
        assert result == "Follow-up response"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_class):
        """Test response generation with tools available but not used"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct answer without using tools"
        mock_response.stop_reason = "end_turn"  # No tool use
        
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = generator.generate_response(
            "What is 2+2?", 
            tools=tools
        )
        
        # Verify tools were included in API call
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        # Verify response
        assert result == "Direct answer without using tools"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic_class):
        """Test response generation with tool calling"""
        mock_client = Mock()
        
        # Mock initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        
        # Mock tool use content block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "artificial intelligence"}
        
        mock_initial_response.content = [mock_tool_block]
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Based on the search results, AI is..."
        
        # Setup client to return initial response first, then final response
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: AI fundamentals content"
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = generator.generate_response(
            "Tell me about AI", 
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="artificial intelligence"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
        
        # Verify final response
        assert result == "Based on the search results, AI is..."

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_flow(self, mock_anthropic_class):
        """Test detailed tool execution message flow"""
        mock_client = Mock()
        
        # Mock initial tool use response
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_456"
        mock_tool_block.input = {"query": "machine learning", "course_name": "AI Basics"}
        mock_initial_response.content = [mock_tool_block]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Machine learning is a subset of AI..."
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "ML course content from AI Basics"
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        tools = [{"name": "search_course_content"}]
        
        result = generator.generate_response(
            "Explain machine learning", 
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Check the second API call (final response) message structure
        final_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = final_call_args["messages"]
        
        # Should have original user message, assistant's tool use, and tool result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Explain machine learning"
        
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == [mock_tool_block]
        
        assert messages[2]["role"] == "user"
        tool_result = messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_456"
        assert tool_result["content"] == "ML course content from AI Basics"

    @patch('ai_generator.anthropic.Anthropic')
    def test_multiple_tool_calls(self, mock_anthropic_class):
        """Test handling multiple tool calls in single response"""
        mock_client = Mock()
        
        # Mock response with multiple tool calls
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_1"
        mock_tool_block1.input = {"query": "neural networks"}
        
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use" 
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.id = "tool_2"
        mock_tool_block2.input = {"query": "deep learning"}
        
        mock_initial_response.content = [mock_tool_block1, mock_tool_block2]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Neural networks and deep learning are..."
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager to return different results for different calls
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Neural network content",
            "Deep learning content"
        ]
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        tools = [{"name": "search_course_content"}]
        
        result = generator.generate_response(
            "Compare neural networks and deep learning",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify both tool calls were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="neural networks")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="deep learning")
        
        # Check final message contains both tool results
        final_call_args = mock_client.messages.create.call_args_list[1][1]
        tool_results = final_call_args["messages"][2]["content"]
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[0]["content"] == "Neural network content"
        assert tool_results[1]["tool_use_id"] == "tool_2" 
        assert tool_results[1]["content"] == "Deep learning content"

    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic_class):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API connection failed")
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        
        # Should raise the exception (not handled in current implementation)
        with pytest.raises(Exception) as exc_info:
            generator.generate_response("test query")
        
        assert "API connection failed" in str(exc_info.value)

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_error(self, mock_anthropic_class):
        """Test handling of tool execution errors"""
        mock_client = Mock()
        
        # Mock tool use response
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_error"
        mock_tool_block.input = {"query": "test"}
        mock_initial_response.content = [mock_tool_block]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "I encountered an error but here's what I know..."
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager to return error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search error: Database connection failed"
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        
        result = generator.generate_response(
            "test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Should still get a response even with tool error
        assert result == "I encountered an error but here's what I know..."
        
        # Check that error was passed to model
        final_call_args = mock_client.messages.create.call_args_list[1][1]
        tool_result = final_call_args["messages"][2]["content"][0]
        assert tool_result["content"] == "Search error: Database connection failed"

    def test_system_prompt_structure(self):
        """Test that system prompt contains expected components"""
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        
        system_prompt = generator.SYSTEM_PROMPT
        
        # Check key components are present
        assert "course materials" in system_prompt
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "One tool call per query maximum" in system_prompt
        assert "Brief, Concise and focused" in system_prompt

    @patch('ai_generator.anthropic.Anthropic')  
    def test_conversation_history_integration(self, mock_anthropic_class):
        """Test that conversation history is properly integrated into system prompt"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with context"
        mock_response.stop_reason = "end_turn"
        
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        
        # Test without history
        generator.generate_response("Test query")
        call_without_history = mock_client.messages.create.call_args_list[0][1]
        system_without_history = call_without_history["system"]
        
        # Test with history
        history = "User: Previous question\nAssistant: Previous answer"
        generator.generate_response("Follow-up query", conversation_history=history)
        call_with_history = mock_client.messages.create.call_args_list[1][1]
        system_with_history = call_with_history["system"]
        
        # Verify history integration
        assert "Previous conversation:" not in system_without_history
        assert "Previous conversation:" in system_with_history
        assert history in system_with_history