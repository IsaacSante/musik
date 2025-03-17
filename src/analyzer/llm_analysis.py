from google import genai
import json
import os
from dotenv import load_dotenv
import re
import time
import asyncio
load_dotenv()
from queue import Queue
from threading import Thread
from src.analyzer.embedding_processor import global_embedding_processor

MODEL = "gemini-2.0-flash-thinking-exp-01-21"

class LLMAnalysis:
    def __init__(self, model_name: str = MODEL):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key is None:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        self.model_name = model_name
        self.client = None

    def _initialize_client(self):
        """Initializes the genai client."""
        if self.api_key is None:
            raise ValueError("API key must be set before calling analyze_lyrics.")
        self.client = genai.Client(api_key=self.api_key)

    def generate_prompt(self, cleaned_lyrics: str) -> str:
        prompt = (
            "Analyze these lyrics line by line. For each line, output exactly in this format:\n\n"
            "LYRIC: [the lyric text]\n"
            "SUBJECT: [one-word main subject]\n"
            "CONCEPT: [one-word key concept]\n"
            "EMOTION: [one-word primary emotion]\n"
            "<<END>>\n\n"
            "Use the <<END>> marker after each line analysis. Return only one word each for SUBJECT, CONCEPT, and EMOTION.\n\n"
            "Important clustering instructions:\n"
            "- If many lines share a general theme or subject, introduce subtle nuance by choosing slightly varied SUBJECT, CONCEPT, or EMOTION keywords to distinguish finer shades of meaning within the same overarching topic.\n"
            "- Balance specificity and generality: avoid overly broad clusters by capturing subtle semantic variations.\n"
            "- Ensure lyrics that are identical or nearly identical maintain very close semantic tags, while nuanced variations produce slight differences.\n\n"
            "Song Lyrics:\n"
            f"{cleaned_lyrics}"
        )
        return prompt
    
    def parsed_lyric(self, lyric):
        print(lyric)
        global_embedding_processor.enqueue(lyric)



    def parse_section(self, section_text):
        section = section_text.strip()
        if not section:
            return None
            
        result = {}
        lines = section.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('LYRIC:'):
                result['lyric'] = line[6:].strip()
            elif line.startswith('SUBJECT:'):
                result['subject'] = line[8:].strip()
            elif line.startswith('CONCEPT:'):
                result['concept'] = line[8:].strip()
            elif line.startswith('EMOTION:'):
                result['emotion'] = line[8:].strip()
        
        # Only return non-empty items that have lyric
        if result and 'lyric' in result:
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
                        # print("Analyzed Lyric:", result)
                        self.parsed_lyric(result)
                        # Add the lyric to the embedding queue for async processing
                        results.append(result)
        
        # Process any remaining content in the buffer
        if buffer.strip():
            result = self.parse_section(buffer.strip())
            if result:
                self.parsed_lyric(result)
                # print("Analyzed Lyric:", result)
                # Add the lyric to the embedding queue
                self.embedding_queue.put(result)
                results.append(result)
                
        return results

    def analyze_lyrics(self, cleaned_lyrics: str, visualize=True, output_file='embeddings.png'):
        try:
            # Ensure there is an event loop in the current thread
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            self._initialize_client()
            prompt = self.generate_prompt(cleaned_lyrics)
            global_embedding_processor.clear_embeddings()
            
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
            # Wait for embedding queue to finish processing
            global_embedding_processor.embedding_queue.join()
            print("All embeddings processed.")

            # Trigger visualization explicitly here
            if visualize:
                from src.analyzer.embedding_visualizer import EmbeddingVisualizer
                visualizer = EmbeddingVisualizer(global_embedding_processor, output_file)
                visualizer.start()
                visualizer.join()
                print("Embedding visualization completed.")

            
            return analysis_results
        except Exception as e:
            print(f"\nAn error occurred during analysis: {e}")
            return []
