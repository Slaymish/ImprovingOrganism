import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Union
import time

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    ML_AVAILABLE = True
    TensorType = torch.Tensor
except ImportError:
    ML_AVAILABLE = False
    torch = None
    F = None
    TensorType = Any  # Fallback type

class LatentWorkspace:
    """
    A latent space workspace for reasoning and thinking without text conversion.
    Preserves rich semantic information and enables complex reasoning operations.
    """
    
    def __init__(self, dim: int = 1024, memory_size: int = 100, num_reasoning_layers: int = 3):
        self.dim = dim
        self.latent_dim = dim  # Alias for compatibility
        self.memory_size = memory_size
        self.num_reasoning_layers = num_reasoning_layers
        # Store torch reference for later use
        self.torch = torch if ML_AVAILABLE else None
        
        if not ML_AVAILABLE:
            print("⚠️  ML dependencies not available. LatentWorkspace running in mock mode.")
            self._init_mock()
            return
            
        # Core workspace tensor - the "thinking space"
        self.workspace = torch.zeros((dim,), dtype=torch.float32)
        
        # Memory components
        self.episodic_memory = deque(maxlen=memory_size)  # Recent experiences
        self.semantic_memory = torch.zeros((memory_size, dim), dtype=torch.float32)  # Long-term knowledge
        self.working_memory = torch.zeros((10, dim), dtype=torch.float32)  # Active reasoning
        
        # Memory management
        self.memory_weights = torch.ones(memory_size) * 0.01  # Importance weights
        self.memory_usage = torch.zeros(memory_size)  # Usage tracking
        self.memory_pointer = 0
        
        # Reasoning state
        self.attention_state = torch.zeros((dim,), dtype=torch.float32)
        self.reasoning_stack = []  # Stack for nested reasoning
        self.thought_chain = []  # Chain of reasoning steps
        self.reasoning_history = []  # History of reasoning operations
        
        # Meta-cognitive components
        self.confidence_state = torch.tensor(0.5)  # Current confidence level
        self.uncertainty_map = torch.zeros((dim,), dtype=torch.float32)  # Uncertainty per dimension
        self.goal_state = torch.zeros((dim,), dtype=torch.float32)  # Current goal representation
        
        # Learning parameters
        self.consolidation_rate = 0.1
        self.attention_decay = 0.95
        self.reasoning_temperature = 1.0
        
    def _init_mock(self):
        """Initialize mock components for development mode"""
        self.workspace = np.zeros(self.dim)
        self.episodic_memory = deque(maxlen=self.memory_size)
        self.semantic_memory = np.zeros((self.memory_size, self.dim))
        self.working_memory = np.zeros((10, self.dim))
        self.memory_weights = np.ones(self.memory_size) * 0.01
        self.memory_usage = np.zeros(self.memory_size)
        self.memory_pointer = 0
        self.attention_state = np.zeros(self.dim)
        self.reasoning_stack = []
        self.thought_chain = []
        self.confidence_state = 0.5
        self.uncertainty_map = np.zeros(self.dim)
        self.goal_state = np.zeros(self.dim)
        
        # Add missing attributes for mock mode
        self.reasoning_history = []
        self.consolidation_rate = 0.1
        self.attention_decay = 0.95
        self.reasoning_temperature = 1.0

    def update(self, embeddings: TensorType, context: str = "", importance: float = 1.0):
        """
        Merge new embeddings into workspace with sophisticated memory consolidation.
        
        Args:
            embeddings: New semantic embeddings to integrate
            context: Textual context for debugging/logging
            importance: Weight for how much to integrate (0.0 to 1.0)
        """
        if not ML_AVAILABLE:
            return self._mock_update(embeddings, context, importance)
            
        # Handle both tensor and numpy input
        if hasattr(embeddings, 'dim'):
            # PyTorch tensor
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
        else:
            # Numpy array - convert to tensor
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            
        # Normalize embeddings to prevent explosion
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # type: ignore
        
        # Compute attention-weighted integration
        attention_weights = self._compute_attention(embeddings)
        weighted_embeddings = embeddings * attention_weights.unsqueeze(-1)
        
        # Update workspace with momentum and attention
        momentum = 0.9
        integration_rate = self.consolidation_rate * importance
        
        for i, emb in enumerate(weighted_embeddings):
            # Selective integration based on novelty and relevance
            novelty = self._compute_novelty(emb)
            relevance = self._compute_relevance(emb)
            
            integration_strength = integration_rate * novelty * relevance
            self.workspace = (momentum * self.workspace + 
                            integration_strength * emb + 
                            (1 - momentum - integration_strength) * self.workspace)
            
            # Store in episodic memory with metadata
            self.episodic_memory.append({
                'embedding': emb.clone(),
                'context': context,
                'timestamp': len(self.episodic_memory),
                'importance': importance,
                'novelty': novelty.item(),
                'relevance': relevance.item()
            })
            
        # Update semantic memory through consolidation
        self._consolidate_to_semantic_memory()
        
        # Update uncertainty estimates
        self._update_uncertainty(embeddings)
        
        # Record thought step
        self.thought_chain.append({
            'type': 'update',
            'workspace_state': self.workspace.clone(),
            'context': context,
            'confidence': self.confidence_state.clone()
        })

    def _mock_update(self, embeddings: Any, context: str = "", importance: float = 1.0):
        """Mock update for development mode"""
        # Handle different input types
        if hasattr(embeddings, 'numpy'):
            # PyTorch tensor (shouldn't happen in mock mode, but just in case)
            mean_emb = embeddings.numpy()
        elif hasattr(embeddings, 'shape'):
            # NumPy array
            mean_emb = embeddings
        else:
            # List or other iterable
            mean_emb = np.array(embeddings)
            
        # Ensure it's 1D and fits workspace size
        if mean_emb.ndim > 1:
            mean_emb = mean_emb.flatten()
        
        # Truncate or pad to workspace size
        workspace_size = len(self.workspace)
        if len(mean_emb) > workspace_size:
            mean_emb = mean_emb[:workspace_size]
        elif len(mean_emb) < workspace_size:
            padded = np.zeros(workspace_size)
            padded[:len(mean_emb)] = mean_emb
            mean_emb = padded
            
        # Update workspace with exponential moving average
        self.workspace = 0.9 * self.workspace + 0.1 * mean_emb
        
        # Store in episodic memory
        self.episodic_memory.append({
            'embedding': mean_emb.copy(),
            'context': context,
            'timestamp': len(self.episodic_memory),
            'importance': importance,
            'novelty': np.random.uniform(0.3, 0.8),  # Mock novelty
            'relevance': np.random.uniform(0.4, 0.9)  # Mock relevance
        })
        
        # Update confidence (simple mock)
        self.confidence_state = min(1.0, self.confidence_state + 0.01 * importance)

    def reason(self, query = "", reasoning_steps: int = 5) -> Dict[str, Any]:
        """
        Perform multi-step reasoning in latent space
        
        Args:
            query: Input query to reason about (string or embedding array)
            reasoning_steps: Number of reasoning iterations
            
        Returns:
            Dict containing reasoning results and metadata
        """
        start_time = time.time()
        
        # Handle both string queries and embedding arrays
        if isinstance(query, str):
            if query:
                query_embedding = self._encode_text(query)
            else:
                # Use zero embedding as neutral query
                if self.torch is None:
                    query_embedding = np.zeros(self.latent_dim)
                else:
                    query_embedding = self.torch.zeros(self.latent_dim)
        else:
            # Assume query is already an embedding array
            query_embedding = query
                
        # Use appropriate reasoning based on mock mode
        if self.torch is None:
            reasoning_result = self._mock_reason(query_embedding, reasoning_steps)
        else:
            reasoning_result = self._real_reason(query_embedding, reasoning_steps)
        
        # Update internal state tracking
        query_summary = query if isinstance(query, str) else f"embedding_array_shape_{getattr(query, 'shape', 'unknown')}"
        self.reasoning_history.append({
            'query': query_summary,
            'timestamp': start_time,
            'steps': reasoning_steps,
            'confidence': float(self.confidence_state) if isinstance(self.confidence_state, (int, float)) else 0.5
        })
        
        # Generate response from reasoning result
        response_text = self._decode_embedding(reasoning_result)
        
        reasoning_time = time.time() - start_time
        
        return {
            'response': response_text,
            'embedding': reasoning_result,  # Include the raw embedding
            'confidence': float(self.confidence_state) if isinstance(self.confidence_state, (int, float)) else 0.5,
            'reasoning_steps': reasoning_steps,
            'reasoning_time': reasoning_time,
            'workspace_state': 'active',
            'memory_usage': len(self.reasoning_history)
        }

    def _real_reason(self, query_embedding, reasoning_steps: int = 5):
        """Real PyTorch-based reasoning"""
        if self.torch is None:
            # Fallback to mock reasoning if torch not available
            return self._mock_reason(query_embedding, reasoning_steps)
            
        # Start with current workspace state - handle both torch and numpy
        try:
            # Try torch tensor methods first
            reasoning_state = self.workspace.clone()
            prev_state = reasoning_state.clone()
        except AttributeError:
            # Fallback to numpy array methods
            reasoning_state = self.workspace.copy()
            prev_state = reasoning_state.copy()
        
        # Multi-step reasoning process
        for step in range(reasoning_steps):
            # Attend to relevant memories
            attended_memories = self._attend_to_memories(reasoning_state)
            
            # Combine current state with attended memories and query
            combined_state = self._combine_states(reasoning_state, attended_memories, query_embedding)
            
            # Perform reasoning step with temporal dynamics
            reasoning_state = self._reasoning_step(combined_state, step)
            
            # Update confidence based on convergence
            if step > 0:
                state_change = self.torch.norm(reasoning_state - prev_state)
                convergence_factor = 1.0 / (1.0 + state_change)
                self.confidence_state = 0.9 * self.confidence_state + 0.1 * convergence_factor
            
            try:
                prev_state = reasoning_state.clone()
            except AttributeError:
                prev_state = reasoning_state.copy()
        
        # Consolidate reasoning into workspace
        self._consolidate_reasoning(reasoning_state)
        
        return reasoning_state

    def _mock_reason(self, query_embedding: Any, reasoning_steps: int = 5):
        """Mock reasoning for development mode"""
        # Handle different input types
        if hasattr(query_embedding, 'numpy'):
            query = query_embedding.numpy()
        elif hasattr(query_embedding, 'shape'):
            query = query_embedding
        else:
            query = np.array(query_embedding)
            
        # Ensure 1D and proper size
        if query.ndim > 1:
            query = query.flatten()
            
        workspace_size = len(self.workspace)
        if len(query) > workspace_size:
            query = query[:workspace_size]
        elif len(query) < workspace_size:
            padded = np.zeros(workspace_size)
            padded[:len(query)] = query
            query = padded
            
        # Simple mock reasoning - iterative blending with workspace
        reasoning_state = self.workspace.copy()
        
        for step in range(reasoning_steps):
            # Blend query with current reasoning state
            blend_factor = 0.3 + 0.1 * step / reasoning_steps
            reasoning_state = (1 - blend_factor) * reasoning_state + blend_factor * query
            
            # Add some noise for variability
            noise = np.random.normal(0, 0.01, reasoning_state.shape)
            reasoning_state += noise
            
            # Normalize to prevent drift
            norm = np.linalg.norm(reasoning_state)
            if norm > 0:
                reasoning_state = reasoning_state / norm
                
            # Update confidence
            self.confidence_state = min(1.0, self.confidence_state + 0.02)
            
        # Update workspace with reasoning result
        self.workspace = 0.7 * self.workspace + 0.3 * reasoning_state
        
        return reasoning_state

    def _reasoning_step(self, combined_state: torch.Tensor, step: int) -> torch.Tensor:
        """
        Single reasoning transformation step.
        This could be replaced with a learned neural network in future versions.
        """
        # Simple but effective reasoning transformation
        # In a more advanced version, this would be a learned neural network
        
        # Ensure combined_state has the right size
        state_dim = self.dim
        if combined_state.size(0) < state_dim:
            # Pad if too small
            padded = torch.zeros(state_dim, dtype=combined_state.dtype, device=combined_state.device)
            padded[:combined_state.size(0)] = combined_state
            combined_state = padded
        elif combined_state.size(0) > state_dim:
            # Truncate if too large
            combined_state = combined_state[:state_dim]
        
        # Simple self-attention mechanism using the state as query, key, and value
        query = combined_state
        key = combined_state
        value = combined_state
        
        # Compute attention scores (dot product attention)
        attention_scores = torch.dot(query, key) / np.sqrt(state_dim)
        attention_weight = torch.sigmoid(attention_scores)  # Single attention weight
        
        # Apply attention (simple scaling)
        attended = attention_weight * value
        
        # Residual connection and normalization
        output = F.normalize(query + attended, p=2, dim=0)
        
        # Add some non-linearity and step-dependent variation
        step_factor = 1.0 + 0.1 * np.sin(step * np.pi / self.num_reasoning_layers)
        output = output * step_factor
        
        return output
        
        return F.normalize(output, p=2, dim=0)

    def _compute_attention(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute attention weights for new embeddings"""
        similarities = F.cosine_similarity(embeddings, self.workspace.unsqueeze(0), dim=1)
        attention = torch.softmax(similarities / self.reasoning_temperature, dim=0)
        return attention

    def _compute_novelty(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute novelty of embedding compared to existing knowledge"""
        if len(self.episodic_memory) == 0:
            return torch.tensor(1.0)
        
        # Compare with recent episodic memories
        recent_embeddings = torch.stack([
            mem['embedding'] for mem in list(self.episodic_memory)[-10:]
        ])
        
        similarities = F.cosine_similarity(embedding.unsqueeze(0), recent_embeddings, dim=1)
        novelty = 1.0 - similarities.max()
        
        return torch.clamp(novelty, 0.0, 1.0)

    def _compute_relevance(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute relevance of embedding to current goal state"""
        if torch.norm(self.goal_state) < 1e-6:
            return torch.tensor(0.5)  # Neutral relevance if no goal set
        
        relevance = F.cosine_similarity(embedding, self.goal_state, dim=0)
        return torch.sigmoid(relevance)  # Ensure positive

    def _attend_to_memories(self, query_state: torch.Tensor) -> torch.Tensor:
        """Compute attention weights over semantic memory"""
        if torch.norm(self.semantic_memory).item() < 1e-6:
            return torch.zeros(self.memory_size)
            
        similarities = F.cosine_similarity(
            query_state.unsqueeze(0), 
            self.semantic_memory, 
            dim=1
        )
        
        # Weight by importance and recency
        attention_scores = similarities * self.memory_weights
        attention_weights = torch.softmax(attention_scores, dim=0)
        
        return attention_weights

    def _gather_attended_memories(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Gather memories based on attention weights"""
        attended = torch.sum(
            attention_weights.unsqueeze(1) * self.semantic_memory, 
            dim=0
        )
        return F.normalize(attended, p=2, dim=0)

    def _consolidate_to_semantic_memory(self):
        """Consolidate episodic memories into semantic memory"""
        if len(self.episodic_memory) < 5:
            return
            
        # Find least used memory slot
        least_used_idx = torch.argmin(self.memory_usage).item()
        
        # Aggregate recent high-importance episodic memories
        recent_important = [
            mem for mem in list(self.episodic_memory)[-20:] 
            if mem['importance'] > 0.5
        ]
        
        if recent_important:
            consolidated = torch.mean(torch.stack([
                mem['embedding'] for mem in recent_important
            ]), dim=0)
            
            # Store in semantic memory
            self.semantic_memory[least_used_idx] = F.normalize(consolidated, p=2, dim=0)
            
            # Update importance weight
            avg_importance = np.mean([mem['importance'] for mem in recent_important])
            self.memory_weights[least_used_idx] = avg_importance
            
            # Reset usage counter
            self.memory_usage[least_used_idx] = 0

    def _update_uncertainty(self, embeddings: torch.Tensor):
        """Update uncertainty estimates based on new information"""
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
            
        # Compute variance across embedding dimensions
        embedding_var = torch.var(embeddings, dim=0)
        
        # Update uncertainty map with exponential moving average
        self.uncertainty_map = 0.9 * self.uncertainty_map + 0.1 * embedding_var

    def set_goal(self, goal_embedding):
        """Set a goal state to guide reasoning and relevance computation"""
        if not ML_AVAILABLE:
            if hasattr(goal_embedding, 'numpy'):
                self.goal_state = goal_embedding.numpy()
            else:
                self.goal_state = np.array(goal_embedding)
            return
        
        # Handle numpy array input even when ML is available
        if hasattr(goal_embedding, 'numpy'):
            # It's already a pytorch tensor
            goal_tensor = goal_embedding
        elif hasattr(goal_embedding, 'ndim'):
            # It's a numpy array, convert to pytorch tensor
            goal_tensor = self.torch.tensor(goal_embedding, dtype=self.torch.float32)
        else:
            # Assume it's already a pytorch tensor
            goal_tensor = goal_embedding
            
        self.goal_state = F.normalize(goal_tensor, p=2, dim=0)
        
        self.thought_chain.append({
            'type': 'goal_set',
            'goal_state': self.goal_state.clone(),
            'workspace_state': self.workspace.clone()
        })

    def introspect(self) -> Dict[str, Any]:
        """
        Introspect current state of the latent workspace.
        Returns rich information about the reasoning state.
        """
        if not ML_AVAILABLE:
            return {
                'workspace_norm': float(np.linalg.norm(self.workspace)),
                'episodic_memory_size': len(self.episodic_memory),
                'confidence': float(self.confidence_state),
                'mode': 'mock'
            }
            
        return {
            'workspace_norm': torch.norm(self.workspace).item(),
            'workspace_entropy': self._compute_entropy(self.workspace),
            'episodic_memory_size': len(self.episodic_memory),
            'semantic_memory_utilization': (self.memory_weights > 0.1).sum().item(),
            'confidence': self.confidence_state.item(),
            'uncertainty_level': torch.mean(self.uncertainty_map).item(),
            'goal_alignment': F.cosine_similarity(self.workspace, self.goal_state, dim=0).item() if torch.norm(self.goal_state) > 1e-6 else 0.0,
            'reasoning_depth': len(self.thought_chain),
            'working_memory_usage': (torch.norm(self.working_memory, dim=1) > 0.1).sum().item(),
            'attention_focus': torch.norm(self.attention_state).item()
        }

    def _combine_states(self, reasoning_state, attended_memories, query_embedding):
        """Combine reasoning state, attended memories, and query into unified state"""
        target_dim = self.latent_dim
        
        if self.torch is None:
            # Mock implementation using numpy - ensure all have same dimensions
            def resize_to_target(arr, target_size):
                if arr.ndim > 1:
                    arr = arr.flatten()
                if len(arr) > target_size:
                    return arr[:target_size]
                elif len(arr) < target_size:
                    padded = np.zeros(target_size)
                    padded[:len(arr)] = arr
                    return padded
                return arr
            
            rs_resized = resize_to_target(reasoning_state, target_dim)
            am_resized = resize_to_target(attended_memories, target_dim)
            qe_resized = resize_to_target(query_embedding, target_dim)
            
            combined = 0.5 * rs_resized + 0.3 * am_resized + 0.2 * qe_resized
            return combined
        else:
            # Real PyTorch implementation - ensure all have same dimensions
            def resize_tensor_to_target(tensor, target_size):
                # Handle both numpy arrays and pytorch tensors
                if hasattr(tensor, 'numpy'):
                    # Convert pytorch tensor to numpy if needed
                    tensor = tensor.numpy()
                    is_numpy = True
                elif hasattr(tensor, 'ndim'):
                    # Already numpy array
                    is_numpy = True
                else:
                    # Assume it's a pytorch tensor
                    is_numpy = False
                
                if is_numpy:
                    if tensor.ndim > 1:
                        tensor = tensor.flatten()
                    if len(tensor) > target_size:
                        result = tensor[:target_size]
                    elif len(tensor) < target_size:
                        result = np.zeros(target_size)
                        result[:len(tensor)] = tensor
                    else:
                        result = tensor
                    # Convert back to torch tensor
                    return self.torch.tensor(result, dtype=self.torch.float32)
                else:
                    # PyTorch tensor handling
                    if tensor.dim() > 1:
                        tensor = tensor.flatten()
                    if tensor.size(0) > target_size:
                        return tensor[:target_size]
                    elif tensor.size(0) < target_size:
                        padded = self.torch.zeros(target_size, dtype=tensor.dtype, device=tensor.device)
                        padded[:tensor.size(0)] = tensor
                        return padded
                    return tensor
            
            rs_resized = resize_tensor_to_target(reasoning_state, target_dim)
            am_resized = resize_tensor_to_target(attended_memories, target_dim)
            qe_resized = resize_tensor_to_target(query_embedding, target_dim)
            
            combined = 0.5 * rs_resized + 0.3 * am_resized + 0.2 * qe_resized
            return F.normalize(combined, p=2, dim=0)

    def _consolidate_reasoning(self, reasoning_result):
        """Consolidate reasoning results into workspace and memory"""
        if self.torch is None:
            # Mock implementation
            self.workspace = 0.8 * self.workspace + 0.2 * reasoning_result
        else:
            # Real implementation
            self.workspace = 0.8 * self.workspace + 0.2 * reasoning_result
            self.workspace = F.normalize(self.workspace, p=2, dim=0)

    def _encode_text(self, text: str):
        """Convert text to embedding (mock implementation)"""
        if self.torch is None:
            # Simple hash-based mock encoding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            # Convert to numbers and normalize
            numbers = [int(c, 16) for c in hash_hex]
            while len(numbers) < self.latent_dim:
                numbers.extend(numbers)
            embedding = np.array(numbers[:self.latent_dim], dtype=np.float32)
            return embedding / np.linalg.norm(embedding)
        else:
            # Mock PyTorch implementation
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            numbers = [int(c, 16) for c in hash_hex]
            while len(numbers) < self.latent_dim:
                numbers.extend(numbers)
            embedding = self.torch.tensor(numbers[:self.latent_dim], dtype=self.torch.float32)
            return F.normalize(embedding, p=2, dim=0)

    def _decode_embedding(self, embedding):
        """Convert embedding back to text (mock implementation)"""
        if hasattr(embedding, 'numpy'):
            embedding = embedding.numpy()
        elif hasattr(embedding, 'detach'):
            embedding = embedding.detach().numpy()
        
        # Simple mock decoding - generate response based on embedding characteristics
        avg_val = float(np.mean(embedding))
        if avg_val > 0.1:
            return "The reasoning process suggests a positive, coherent response with high confidence."
        elif avg_val < -0.1:
            return "The analysis indicates areas requiring deeper consideration and refinement."
        else:
            return "The latent reasoning reveals balanced perspectives requiring synthesis."

    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Compute entropy of a tensor (measure of information content)"""
        probs = torch.softmax(torch.abs(tensor), dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item()

    def read(self) -> torch.Tensor:
        """Read current workspace state"""
        if not ML_AVAILABLE:
            return self.workspace
        return self.workspace.clone()

    def get_thought_chain(self) -> List[Dict[str, Any]]:
        """Get the chain of reasoning steps for analysis"""
        return self.thought_chain.copy()

    def clear_thought_chain(self):
        """Clear the thought chain (for privacy or memory management)"""
        self.thought_chain.clear()

    def reset_workspace(self, preserve_memory: bool = True):
        """Reset workspace while optionally preserving memory"""
        if not ML_AVAILABLE:
            self.workspace = np.zeros_like(self.workspace)
            if not preserve_memory:
                self.episodic_memory.clear()
            return
            
        self.workspace = torch.zeros_like(self.workspace)
        self.attention_state = torch.zeros_like(self.attention_state)
        self.confidence_state = torch.tensor(0.5)
        
        if not preserve_memory:
            self.episodic_memory.clear()
            self.semantic_memory = torch.zeros_like(self.semantic_memory)
            self.memory_weights = torch.ones_like(self.memory_weights) * 0.01
            self.memory_usage = torch.zeros_like(self.memory_usage)
            
        self.thought_chain.clear()
