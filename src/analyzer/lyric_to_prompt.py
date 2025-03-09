from google import genai
import os
from dotenv import load_dotenv
import time
import asyncio

load_dotenv()

class LyricToImagePrompter:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key is None:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        self.model_name = model_name
        self.client = None

    def _initialize_client(self):
        """Initializes the genai client."""
        if self.api_key is None:
            raise ValueError("API key must be set before calling generate_image_prompts.")
        self.client = genai.Client(api_key=self.api_key)

    def generate_prompt(self, cleaned_lyrics: str) -> str:
        prompt = (
            f"Create concise cinematic image prompts from these song lyrics.\n"
            "For each lyric line, describe a clear, visually filmable scene in exactly 1 concise sentence.\n\n"
            "Each prompt must:\n"
            "- Describe a specific, tangible visual moment (avoid abstract or metaphorical language).\n"
            "- Avoid overly poetic language; use straightforward visual descriptions.\n"
            "- Specify visual details such as clothing, surroundings, expressions, objects, and lighting.\n"
            "- Avoid generic terms like 'woman/man/child'.\n"
            "- Each prompt must be exactly 1 concise sentence.\n\n"
            "Format:\n\n"
            "LINE: [lyric text]\n"
            "PROMPT: [1 clear, specific sentence visually representing the emotion and narrative of the lyric line]\n"
            "<<END>>\n\n"
            "Song Lyrics:\n"
            f"{cleaned_lyrics}\n"
        )
        return prompt

    def parse_section(self, section_text, image_generator=None, song_name=None):
        """Parse a single section of lyric-to-image prompt and generate image if requested."""
        section = section_text.strip()
        if not section:
            return None
            
        result = {}
        lines = section.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('LINE:'):
                result['text'] = line[5:].strip()
            elif line.startswith('PROMPT:'):
                result['prompt'] = line[7:].strip()
        
        # Only return non-empty items that have text
        if result and 'text' in result and 'prompt' in result:
            # Generate image immediately if generator is provided
            if image_generator and song_name:
                image_generator.generate_image(
                    line=result['text'],
                    prompt=result['prompt'],
                    song_name=song_name
                )
            return result
        
        return None

    def process_stream_chunks(self, chunk_stream, image_generator=None, song_name=None):
        """
        Process streaming chunks and extract complete lyric-to-image sections 
        separated by the <<END>> delimiter in real-time.
        Generate images immediately when a section is complete if image_generator is provided.
        """
        buffer = ""
        results = []
        
        for chunk in chunk_stream:
            chunk_text = chunk.text
            print(chunk_text, end='')  # Print chunks as they come in
            
            buffer += chunk_text
            
            # Check if we have complete sections (marked by <<END>>)
            while "<<END>>" in buffer:
                parts = buffer.split("<<END>>", 1)
                section = parts[0].strip()
                buffer = parts[1]
                
                # Parse the complete section and generate image immediately
                if section:
                    result = self.parse_section(section, image_generator, song_name)
                    if result:
                        results.append(result)
        
        # Process any remaining content in the buffer
        if buffer.strip():
            result = self.parse_section(buffer.strip(), image_generator, song_name)
            if result:
                results.append(result)
                
        return results  

    def generate_image_prompts(self, cleaned_lyrics: str, image_generator=None, song_name=None):
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
            # Pass image_generator and song_name to generate images immediately
            prompt_results = self.process_stream_chunks(
                response_stream, 
                image_generator=image_generator, 
                song_name=song_name
            )
            
            total_elapsed = time.time() - start_time
            print(f"\nPrompt generation completed in {total_elapsed:.2f} seconds")
            
            return prompt_results
        except Exception as e:
            print(f"\nAn error occurred during prompt generation: {e}")
            return []
