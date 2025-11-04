import re
from typing import List, Tuple, Dict, Optional
from collections import Counter


class SmartChunker:
    """
    Regex-based smart chunking for documents.
    Detects document structure and creates optimal chunk boundaries.
    """
    
    # Common document structure patterns (generic for any dataset)
    STRUCTURE_PATTERNS = {
        'header': [
            r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS HEADERS
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s*$',  # Title Case: headers
            r'^(SECTION|PART|CHAPTER)\s+\d+',  # Section headers
        ],
        'field': [
            r'^[A-Z][a-z]+(?:\s+[a-z]+)+:\s*',  # Key: Value patterns (e.g., "Purpose for submission:")
            r'^[A-Z][a-z]+(?:\s+[a-z]+)*:\s+',  # Title case fields with values
            r'^[A-Z_]+:\s*',  # UPPER_SNAKE_CASE: fields
            r'^[a-z]+(?:\s+[a-z]+)*:\s*',  # lowercase fields (e.g., "measurand:")
            r'^\d+[\.\)]\s+[A-Z]',  # Numbered items
        ],
        'delimiter': [
            r'^-{3,}',  # Horizontal lines
            r'^={3,}',  # Double lines
            r'^\*{3,}',  # Asterisks
            r'^_{3,}',  # Underscores
        ],
        'paragraph': [
            r'^\s*$',  # Blank lines (paragraph breaks)
            r'\.\s+[A-Z]',  # End of sentence
        ],
    }
    
    def __init__(self):
        self.detected_patterns = {}
        self.strategy = None
    
    def detect_document_structure(self, text: str) -> str:
        """
        Analyze document and detect the best chunking strategy.
        Returns: 'header_based', 'field_based', 'delimiter_based', or 'adaptive'
        """
        lines = text.split('\n')
        pattern_scores = Counter()
        
        # Test generic patterns for any document type
        for pattern_type, patterns in self.STRUCTURE_PATTERNS.items():
            for pattern in patterns:
                for line in lines[:100]:  # Check first 100 lines
                    if re.search(pattern, line, re.IGNORECASE):
                        pattern_scores[pattern_type] += 1
        
        # Determine best strategy based on detected patterns
        if pattern_scores['header'] > 20:
            return 'header_based'
        elif pattern_scores['field'] > 15:
            return 'field_based'
        elif pattern_scores['delimiter'] > 5:
            return 'delimiter_based'
        else:
            return 'adaptive'
    
    def smart_chunk(
        self, 
        text: str, 
        max_chunk_size: int = 100000,
        overlap_size: int = 10000,
        strategy: Optional[str] = None
    ) -> List[str]:
        """
        Create intelligent chunks based on document structure.
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum characters per chunk
            overlap_size: Overlap between chunks
            strategy: Chunking strategy ('header_based', 'field_based', 
                      'delimiter_based', 'adaptive' or None for auto-detect)
        
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        if len(text) <= max_chunk_size:
            return [text]
        
        # Auto-detect strategy if not provided
        if strategy is None:
            strategy = self.detect_document_structure(text)
        
        # Apply strategy-specific chunking
        if strategy == 'header_based':
            chunks = self._header_chunking(text, max_chunk_size, overlap_size)
        elif strategy == 'field_based':
            chunks = self._field_chunking(text, max_chunk_size, overlap_size)
        elif strategy == 'delimiter_based':
            chunks = self._delimiter_chunking(text, max_chunk_size, overlap_size)
        else:  # adaptive or fallback
            chunks = self._adaptive_chunking(text, max_chunk_size, overlap_size)
        
        return chunks if chunks else self._fallback_chunking(text, max_chunk_size, overlap_size)
    
    
    def _header_chunking(
        self, 
        text: str, 
        max_chunk_size: int, 
        overlap_size: int
    ) -> List[str]:
        """Chunk by header boundaries."""
        chunks = []
        lines = text.split('\n')
        header_indices = []
        
        # Find header lines
        for i, line in enumerate(lines):
            if any(re.search(pattern, line) for patterns in [self.STRUCTURE_PATTERNS['header']] for pattern in patterns):
                if i > 0:  # Don't include first line if it's a header
                    header_indices.append(i)
        
        if not header_indices:
            return []
        
        # Create chunks at header boundaries
        for i in range(len(header_indices)):
            start = header_indices[i]
            end = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)
            
            chunk_lines = lines[start:end]
            chunk_text = '\n'.join(chunk_lines)
            
            if len(chunk_text) > max_chunk_size:
                # Recurse on large chunks
                sub_chunks = self._adaptive_chunking(chunk_text, max_chunk_size, overlap_size)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def _field_chunking(
        self, 
        text: str, 
        max_chunk_size: int, 
        overlap_size: int
    ) -> List[str]:
        """Chunk by key-value field boundaries (generic for any dataset)."""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline
            
            # Check if this line starts a new field using generic patterns
            is_field_start = any(
                re.search(pattern, line, re.IGNORECASE) 
                for pattern in self.STRUCTURE_PATTERNS['field']
            )
            
            # If starting a new field and current chunk is getting large
            if is_field_start and current_size > max_chunk_size * 0.7:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk, overlap_size)
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
            else:
                current_chunk.append(line)
                current_size += line_size
                
                # Emergency break if chunk gets too large
                if current_size > max_chunk_size * 1.5:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _delimiter_chunking(
        self, 
        text: str, 
        max_chunk_size: int, 
        overlap_size: int
    ) -> List[str]:
        """Chunk by delimiter lines."""
        chunks = []
        lines = text.split('\n')
        delimiter_indices = []
        
        # Find delimiter lines
        for i, line in enumerate(lines):
            if any(re.search(pattern, line) for pattern in self.STRUCTURE_PATTERNS['delimiter']):
                delimiter_indices.append(i)
        
        if not delimiter_indices:
            return []
        
        # Create chunks between delimiters
        for i in range(len(delimiter_indices) - 1):
            start = delimiter_indices[i]
            end = delimiter_indices[i + 1]
            chunk_lines = lines[start:end]
            chunks.append('\n'.join(chunk_lines))
        
        return chunks
    
    def _adaptive_chunking(
        self, 
        text: str, 
        max_chunk_size: int, 
        overlap_size: int
    ) -> List[str]:
        """Adaptive chunking with sentence awareness."""
        chunks = []
        sentences = re.split(r'(\.\s+)', text)
        
        current_chunk = []
        current_size = 0
        
        for i in range(0, len(sentences), 2):  # Process pairs (sentence + delimiter)
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(''.join(current_chunk))
                
                # Add overlap from previous chunk
                overlap_text = self._get_overlap_text(''.join(current_chunk), overlap_size)
                current_chunk = [overlap_text] if overlap_text else []
                current_size = len(overlap_text)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
    
    def _fallback_chunking(
        self, 
        text: str, 
        max_chunk_size: int, 
        overlap_size: int
    ) -> List[str]:
        """Fallback to simple character-based chunking."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + max_chunk_size // 2:
                end = sentence_end + 1
            
            chunks.append(text[start:end])
            start = end - overlap_size
        
        return chunks
    
    def _get_overlap_lines(self, lines: List[str], overlap_size: int) -> List[str]:
        """Get last N lines for overlap."""
        overlap_lines = []
        size = 0
        
        for line in reversed(lines):
            line_size = len(line) + 1
            if size + line_size <= overlap_size:
                overlap_lines.insert(0, line)
                size += line_size
            else:
                break
        
        return overlap_lines
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get last N characters for overlap."""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
