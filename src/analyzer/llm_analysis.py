from google import genai
import json
import os
from dotenv import load_dotenv
import re
import time
import asyncio
load_dotenv()
from src.embeddings.embedding_pipeline import EmbeddingPipeline

class LLMAnalysis:
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key is None:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        self.model_name = model_name
        self.client = None
        self.embedding_pipeline = EmbeddingPipeline() 



    def _initialize_client(self):
        """Initializes the genai client."""
        if self.api_key is None:
            raise ValueError("API key must be set before calling analyze_lyrics.")
        self.client = genai.Client(api_key=self.api_key)

    def parsed_lyric_callback(self, lyric):
            print(lyric)
            self.embedding_pipeline.enqueue_lyric(lyric)


    def generate_prompt(self, cleaned_lyrics: str) -> str:
        prompt = (
            "Analyze these lyrics line by line. For each line, output in this exact format:\n\n"
            "LINE: [the lyric text]\n"
            "THEMES: [comma-separated, brief (1-3 words each)]\n"
            "EMOTIONS: [comma-separated, brief (1-2 words each)]\n"
            "SUBJECTS:: [comma-separated, brief (1-3 words each)]\n"
            "<<END>>\n\n"
            "Use the <<END>> marker after each line analysis. Keep themes and subjects brief (1-3 words each).\n\n"
            "Song Lyrics:\n"
            f"{cleaned_lyrics}"
        )
        return prompt

    def parse_section(self, section_text):
        """Parse a single section of lyric analysis with the simplified format."""
        section = section_text.strip()
        if not section:
            return None
            
        result = {}
        lines = section.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('LINE:'):
                result['text'] = line[5:].strip()
            elif line.startswith('THEMES:'):
                result['concepts'] = [c.strip() for c in line[7:].strip().split(',')]
            elif line.startswith('EMOTIONS:'):
                result['emotions'] = [e.strip() for e in line[9:].strip().split(',')]
            elif line.startswith('SUBJECTS:'):  # Change to match your prompt
                result['subjects'] = [s.strip() for s in line[9:].strip().split(',')]  # Plural to match expected format
        
        # Only return non-empty items that have text
        if result and 'text' in result:
            return result
        return None


    def process_stream_chunks(self, chunk_stream):
        buffer = ""
        results = []
        first_chunk_received = False
        start_time = time.time()
        
        for chunk in chunk_stream:
            if not first_chunk_received:
                first_chunk_received = True
                elapsed = time.time() - start_time
                print(f"{elapsed:.2f} seconds till first chunk")
                
            buffer += chunk.text
            
            # Check if we have complete sections (marked by <<END>>)
            while "<<END>>" in buffer:
                parts = buffer.split("<<END>>", 1)
                section = parts[0].strip()
                buffer = parts[1]
                
                # Parse the complete section
                if section:
                    result = self.parse_section(section)
                    if result:
                        self.parsed_lyric_callback(result)                        
                        results.append(result)
        
        # Process any remaining content in the buffer
        if buffer.strip():
            result = self.parse_section(buffer.strip())
            if result:
                # print("Analyzed Lyric:", result)
                # Add the lyric to the embedding queue
                results.append(result)
                
        return results



    def analyze_lyrics(self, cleaned_lyrics: str):
        try:
            # Ensure there is an event loop in the current thread
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            self._initialize_client()
            prompt = self.generate_prompt(cleaned_lyrics)
            
            # Record start time before sending prompt
            start_time = time.time()
            print("Sending prompt to model...")
            
            # Use streaming version
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[prompt]
            )
            
            # Process the streaming response and parse sections in real-time
            # We pass start_time implicitly by capturing it in the closure
            analysis_results = self.process_stream_chunks(response_stream)
            
            total_elapsed = time.time() - start_time
            print(f"Analysis completed in {total_elapsed:.2f} seconds")
            
            return analysis_results
        except Exception as e:
            print(f"\nAn error occurred during analysis: {e}")
            return []
