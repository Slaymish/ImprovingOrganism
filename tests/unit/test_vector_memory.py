import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from src.vector_memory import VectorMemory

class TestVectorMemory(unittest.TestCase):

    @patch('src.vector_memory.weaviate.Client')
    @patch('src.vector_memory.LLMWrapper')
    def setUp(self, MockLLMWrapper, MockWeaviateClient):
        self.mock_llm_wrapper = MockLLMWrapper.return_value
        self.mock_weaviate_client = MockWeaviateClient.return_value
        
        # Configure the mock client to be "ready"
        self.mock_weaviate_client.is_ready.return_value = True
        self.mock_weaviate_client.schema.exists.return_value = False

        self.vector_memory = VectorMemory()
        # Replace the client instance with the mock after initialization
        self.vector_memory.client = self.mock_weaviate_client

    def test_connection(self):
        self.assertIsNotNone(self.vector_memory.client)
        self.mock_weaviate_client.is_ready.assert_called_once()

    def test_ensure_schema_creates_class_if_not_exists(self):
        self.mock_weaviate_client.schema.exists.return_value = False
        self.vector_memory._ensure_schema()
        self.mock_weaviate_client.schema.exists.assert_called_with("MemoryEntry")
        self.mock_weaviate_client.schema.create_class.assert_called_once()

    def test_ensure_schema_does_not_create_class_if_exists(self):
        self.mock_weaviate_client.schema.exists.return_value = True
        self.vector_memory._ensure_schema()
        self.mock_weaviate_client.schema.exists.assert_called_with("MemoryEntry")
        self.mock_weaviate_client.schema.create_class.assert_not_called()

    def test_add_entry(self):
        content = "test content"
        embedding = np.random.rand(768)
        self.mock_llm_wrapper.get_embeddings.return_value = embedding

        self.vector_memory.add_entry(content, "test_type")

        self.mock_llm_wrapper.get_embeddings.assert_called_with(content)
        self.mock_weaviate_client.data_object.create.assert_called_once()
        args, kwargs = self.mock_weaviate_client.data_object.create.call_args
        self.assertEqual(kwargs['class_name'], "MemoryEntry")
        self.assertEqual(kwargs['data_object']['content'], content)
        self.assertEqual(kwargs['vector'], embedding.tolist())

    def test_add_entry_no_embedding(self):
        self.mock_llm_wrapper.get_embeddings.return_value = None
        self.vector_memory.add_entry("test", "test_type")
        self.mock_weaviate_client.data_object.create.assert_not_called()

    def test_search(self):
        query = "search query"
        embedding = np.random.rand(768)
        self.mock_llm_wrapper.get_embeddings.return_value = embedding

        mock_response = {"data": {"Get": {"MemoryEntry": [{"content": "result"}]}}}
        
        # Mock the fluent interface
        mock_query_builder = MagicMock()
        self.mock_weaviate_client.query.get.return_value = mock_query_builder
        mock_query_builder.with_near_vector.return_value = mock_query_builder
        mock_query_builder.with_limit.return_value = mock_query_builder
        mock_query_builder.with_where.return_value = mock_query_builder
        mock_query_builder.do.return_value = mock_response

        results = self.vector_memory.search(query)

        self.mock_llm_wrapper.get_embeddings.assert_called_with(query)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['content'], "result")

    def test_search_no_embedding(self):
        self.mock_llm_wrapper.get_embeddings.return_value = None
        results = self.vector_memory.search("query")
        self.assertEqual(results, [])

if __name__ == '__main__':
    unittest.main()

