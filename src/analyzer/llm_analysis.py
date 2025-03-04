from google import genai
import json
import os
from dotenv import load_dotenv
import re
load_dotenv()
import asyncio

def remove_markdown_fences(text):
    # Remove the starting ```json or ``` and the ending ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def parse_and_print_objects(chunk_stream):
    """
    Processes a stream of chunks, filters them with remove_markdown_fences,
    detects complete JSON objects, prints them, and returns a list of parsed objects.
    """
    buffer = ""
    brace_count = 0
    in_string = False
    escape = False
    obj_start = None
    results = []
    
    for chunk in chunk_stream:
        # Filter the chunk text using remove_markdown_fences
        filtered_text = remove_markdown_fences(chunk.text)
        for char in filtered_text:
            # Skip array markers if not currently inside an object
            if char in ['[', ']'] and brace_count == 0:
                continue

            buffer += char

            # Toggle in_string when encountering a non-escaped quote
            if char == '"' and not escape:
                in_string = not in_string

            if not in_string:
                if char == '{':
                    if brace_count == 0:
                        # Mark the beginning of a new JSON object.
                        obj_start = len(buffer) - 1
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and obj_start is not None:
                        # Extract the complete JSON object
                        obj_text = buffer[obj_start:]
                        try:
                            obj = json.loads(obj_text)
                            print("Analyzed Lyric:", obj)
                            results.append(obj)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                        # Reset the buffer after processing a complete object.
                        buffer = ""
                        obj_start = None

            # Handle escape characters inside strings.
            if char == '\\' and in_string:
                escape = True
            else:
                escape = False

    return results

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
            "Analyze the following lyrics line by line and list the key concept(s) or themes present "
            "in each line and the subject being addressed.\n\n"
            "Data Format: Organize the output into a JSON array, where each element is an object representing a line of the lyrics. For example:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"first lyric\",\n"
            "    \"concepts\": [\"concept 1\", \"concept 2\"],\n"
            "    \"subject\": [\"subject 1\", \"subject 2\"]\n"
            "  },\n"
            "  {\n"
            "    \"text\": \"second lyric\",\n"
            "    \"concepts\": [\"concept 1\", \"concept 2\", \"concept 3\"],\n"
            "    \"subject\": [\"subject 1\"]\n"
            "  },\n"
            "  {\n"
            "    \"text\": \"third lyric\",\n"
            "    \"concepts\": [\"concept 1\", \"concept 2\", \"concept 3\"],\n"
            "    \"subject\": [\"subject 1\", \"subject 2\"]\n"
            "  }\n"
            "  // ... additional lines\n"
            "]\n\n"
            "Song Lyrics:\n"
            f"{cleaned_lyrics}"
        )
        return prompt

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
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[prompt]
            )

            analysis_results = parse_and_print_objects(response_stream)
            return analysis_results
        except Exception as e:
            print(f"\nAn error occurred during analysis: {e}")
            return {}
