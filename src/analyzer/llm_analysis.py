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

class LLMAnalysis:
    def __init__(self, model_name: str = "gemini-2.0-flash-lite-preview-02-05"):
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
            "Analyze these lyrics line by line. For each line, output in this exact format:\n\n"
            "LINE: [the lyric text]\n"
            "THEMES: [comma-separated list of key concepts/themes]\n"
            "WHO: [comma-separated list of subjects addressed]\n"
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
            elif line.startswith('WHO:'):
                result['subject'] = [s.strip() for s in line[4:].strip().split(',')]
        
        # Only return non-empty items that have text
        if result and 'text' in result:
            return result
        return None

    def process_stream_chunks(self, chunk_stream):
        """
        Process streaming chunks and extract complete lyric analysis sections 
        separated by the <<END>> delimiter in real-time.
        """
        # Create a global embedding pipeline if it doesn't exist
        if not hasattr(self, 'embedding_pipeline'):
            try:
                # Check if we can import UMAP
                try:
                    from umap import UMAP
                    dim_reduction = 'umap'
                    print("UMAP available - will use for dimensionality reduction")
                except (ImportError, AttributeError):
                    dim_reduction = 'pca'
                    print("UMAP not available - will use PCA for dimensionality reduction")
                    
                from src.embedding.lyric_embedding_pipeline import LyricEmbeddingPipeline
                
                # Create with best available options
                self.embedding_pipeline = LyricEmbeddingPipeline(
                    clustering_method='dbscan',
                    dim_reduction=dim_reduction,
                    max_display_lyrics=50  # Adjust based on your preference
                )
                
                # Create a queue for passing data to the embedding thread
                self.embedding_queue = Queue()
                
                # Start a worker thread that processes items from the queue
                def worker():
                    while True:
                        lyric_data = self.embedding_queue.get()
                        if lyric_data is None:  # None is a signal to exit
                            break
                        try:
                            self.embedding_pipeline.add_lyric(lyric_data)
                        except Exception as e:
                            print(f"Error processing lyric: {e}")
                            import traceback
                            traceback.print_exc()  # Print full stack trace for debugging
                        finally:
                            self.embedding_queue.task_done()
                
                self.embedding_thread = Thread(target=worker, daemon=True)
                self.embedding_thread.start()
                print("Started embedding pipeline worker thread")
                
            except Exception as e:
                print(f"Failed to initialize embedding pipeline: {e}")
                import traceback
                traceback.print_exc()
        
        # Rest of the method remains the same...
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
                        print("Analyzed Lyric:", result)
                        # Add the lyric to the embedding queue for async processing
                        self.embedding_queue.put(result)
                        results.append(result)
        
        # Process any remaining content in the buffer
        if buffer.strip():
            result = self.parse_section(buffer.strip())
            if result:
                print("Analyzed Lyric:", result)
                # Add the lyric to the embedding queue
                self.embedding_queue.put(result)
                results.append(result)
                
        return results



    def __del__(self):
        """
        Clean up resources when the object is garbage collected.
        """
        if hasattr(self, 'embedding_queue') and hasattr(self, 'embedding_thread'):
            # Signal the worker thread to exit
            self.embedding_queue.put(None)
            # Wait for the worker thread to finish (with a timeout)
            self.embedding_thread.join(timeout=1.0)
            print("Embedding pipeline worker thread shut down")


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
