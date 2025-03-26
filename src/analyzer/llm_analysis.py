from google import genai
import os
from dotenv import load_dotenv
import time
import asyncio
import threading
import traceback
import string # Needed for parse_section if you use normalization there, but storage handles it now
from src.storage.song_analysis_storage import SongAnalysisStorage

load_dotenv()

MODEL = "gemini-1.5-flash-latest"

class LLMAnalysis:
    # --- __init__, _initialize_client, generate_prompt remain the same ---
    def __init__(self, model_name: str = MODEL):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key is None:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        self.model_name = model_name
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        if self.api_key is None:
             raise ValueError("API key must be set before initializing client.")
        if not self.client:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("Google GenAI Client initialized.")
            except Exception as e:
                print(f"Error initializing Google GenAI Client: {e}")
                self.client = None
                raise

    def generate_prompt(self, cleaned_lyrics: str) -> str:
        # --- Prompt remains the same ---
        prompt = (
            "Analyze these lyrics line by line. For each line, output exactly in this format:\n\n"
            "LYRIC: [the lyric text]\n"
            "SUBJECT: [one-word main subject]\n"
            "CONCEPT: [one-word key concept]\n"
            "EMOTION_WORD: [one-word primary emotion]\n"
            "VALENCE: [Score between -1.0 and 1.0]\n"
            "AROUSAL: [Score between -1.0 and 1.0]\n"
            "TENSION: [Score between -1.0 and 1.0]\n"
            "<<END>>\n\n"
            "Use the <<END>> marker after each line analysis.\n"
            "Return only one word each for SUBJECT, CONCEPT, and EMOTION_WORD.\n"
            "Provide numerical scores between -1.0 and 1.0 for VALENCE, AROUSAL, and TENSION.\n\n"
            "Song Lyrics:\n"
            f"{cleaned_lyrics}"
        )
        return prompt

    # --- parse_section remains the same ---
    def parse_section(self, section_text):
        section = section_text.strip()
        if not section:
            return None

        result = {}
        lines = section.split('\n')
        # Keep track of fields requested in the prompt
        missing_fields = {'subject', 'concept', 'emotion_word', 'valence', 'arousal', 'tension'}

        for line in lines:
            line = line.strip()
            try:
                if line.startswith('LYRIC:'):
                    result['lyric'] = line[len('LYRIC:'):].strip()
                elif line.startswith('SUBJECT:'):
                    result['subject'] = line[len('SUBJECT:'):].strip(); missing_fields.discard('subject')
                elif line.startswith('CONCEPT:'):
                    result['concept'] = line[len('CONCEPT:'):].strip(); missing_fields.discard('concept')
                elif line.startswith('EMOTION_WORD:'):
                    result['emotion_word'] = line[len('EMOTION_WORD:'):].strip(); missing_fields.discard('emotion_word')
                elif line.startswith('VALENCE:'):
                    value_str = line[len('VALENCE:'):].strip()
                    try: result['valence'] = max(-1.0, min(1.0, float(value_str)))
                    except ValueError: result['valence'] = 0.0; print(f"Warn: Bad VALENCE: {value_str}")
                    missing_fields.discard('valence')
                elif line.startswith('AROUSAL:'):
                    value_str = line[len('AROUSAL:'):].strip()
                    try: result['arousal'] = max(-1.0, min(1.0, float(value_str)))
                    except ValueError: result['arousal'] = 0.0; print(f"Warn: Bad AROUSAL: {value_str}")
                    missing_fields.discard('arousal')
                elif line.startswith('TENSION:'):
                    value_str = line[len('TENSION:'):].strip()
                    try: result['tension'] = max(-1.0, min(1.0, float(value_str)))
                    except ValueError: result['tension'] = 0.0; print(f"Warn: Bad TENSION: {value_str}")
                    missing_fields.discard('tension')
            except Exception as e:
                 print(f"Warning: Error processing line '{line}': {e}")

        if result and 'lyric' in result:
             # Set defaults only for fields explicitly requested but missing in the response
             if 'valence' in missing_fields: result.setdefault('valence', 0.0)
             if 'arousal' in missing_fields: result.setdefault('arousal', 0.0)
             if 'tension' in missing_fields: result.setdefault('tension', 0.0)
             if 'subject' in missing_fields: result.setdefault('subject', 'Unknown')
             if 'concept' in missing_fields: result.setdefault('concept', 'Unknown')
             if 'emotion_word' in missing_fields: result.setdefault('emotion_word', 'Neutral')
             return result
        elif section:
            # Only warn if the section wasn't just whitespace
            print(f"Warning: Could not parse LYRIC from section:\n---\n{section}\n---")
        return None


    # --- _print_analysis_data remains the same ---
    def _print_analysis_data(self, data: dict):
        """Prints the analyzed data chunk to the console."""
        try:
            print(f"  [Analysis] LYRIC: {data.get('lyric', 'N/A')}")
            print(f"    SUBJECT: {data.get('subject', '?')}, CONCEPT: {data.get('concept', '?')}, EMOTION: {data.get('emotion_word', '?')}")
            print(f"    VALENCE: {data.get('valence', 0.0):.2f}, AROUSAL: {data.get('arousal', 0.0):.2f}, TENSION: {data.get('tension', 0.0):.2f}")
            print("-" * 15)
        except Exception as e:
            print(f"Error printing analysis data chunk: {e}")
            print(f"Problematic data: {data}")

    # --- MODIFIED: Renamed and accepts storage_callback ---
    def _process_stream(self, chunk_stream, storage_callback: callable = None):
        """
        Processes the stream, prints results, and calls the storage_callback
        for each valid parsed data chunk.
        """
        buffer = ""
        total_items_processed = 0
        first_chunk_received = False
        start_time = time.time()

        for chunk in chunk_stream:
            try:
                chunk_text = chunk.text
            except Exception as e:
                print(f"Warning: Error accessing chunk text: {e}")
                chunk_text = ""

            if not first_chunk_received and chunk_text:
                first_chunk_received = True
                elapsed = time.time() - start_time
                print(f"[Analysis Thread] {elapsed:.2f} seconds till first chunk")

            buffer += chunk_text

            while "<<END>>" in buffer:
                try:
                    parts = buffer.split("<<END>>", 1)
                    section = parts[0].strip()
                    buffer = parts[1]

                    if section:
                        result = self.parse_section(section)
                        if result:
                            # 1. Print to console
                            self._print_analysis_data(result)

                            # 2. *** CALL STORAGE CALLBACK ***
                            if storage_callback:
                                try:
                                    storage_callback(result)
                                except Exception as cb_e:
                                    print(f"[Analysis Thread] Error in storage_callback: {cb_e}")
                                    traceback.print_exc()
                            # ********************************

                            total_items_processed += 1
                except Exception as e:
                    print(f"[Analysis Thread] Error processing <<END>> block: {e}")
                    if "<<END>>" in buffer:
                         buffer = buffer.split("<<END>>", 1)[1]
                    else:
                         buffer = ""

        # Process remaining buffer content
        try:
            remaining_section = buffer.strip()
            if remaining_section:
                 if 'LYRIC:' in remaining_section or 'SUBJECT:' in remaining_section:
                    print(f"[Analysis Thread] Processing remaining buffer content...")
                    result = self.parse_section(remaining_section)
                    if result:
                         # 1. Print final part
                         self._print_analysis_data(result)
                         # 2. *** CALL STORAGE CALLBACK (FINAL) ***
                         if storage_callback:
                             try:
                                 storage_callback(result)
                             except Exception as cb_e:
                                 print(f"[Analysis Thread] Error in storage_callback (final): {cb_e}")
                                 traceback.print_exc()
                         # **************************************
                         total_items_processed += 1
        except Exception as e:
            print(f"[Analysis Thread] Error processing final buffer content: {e}")

        return {"total_items_processed": total_items_processed}

    # --- MODIFIED: Accepts and passes storage_callback ---
    def _perform_analysis_thread(self, cleaned_lyrics: str, storage_callback: callable = None):
        """
        Runs the LLM analysis and processes results. Intended for threading.
        Accepts an optional callback for storing results.
        """
        print("[Analysis Thread] Started.")
        thread_start_time = time.time()
        try:
            if not self.client:
                print("[Analysis Thread] Error: Client not initialized.")
                return

            prompt = self.generate_prompt(cleaned_lyrics)
            print("[Analysis Thread] Sending prompt to model...")

            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[prompt]
            )

            # *** PASS storage_callback DOWN ***
            summary_info = self._process_stream(response_stream, storage_callback)
            # **********************************

            total_elapsed = time.time() - thread_start_time
            print(f"[Analysis Thread] Stream completed in {total_elapsed:.2f} seconds. "
                  f"Processed approx {summary_info.get('total_items_processed', 0)} items.")

        except Exception as e:
            print(f"\n[Analysis Thread] An error occurred during analysis: {e}")
            traceback.print_exc()
        finally:
             print("[Analysis Thread] Finished.")


    # --- MODIFIED: Accepts storage_callback and passes it to thread ---
    def analyze_lyrics(self, cleaned_lyrics: str, storage_callback: callable = None):
        """
        Starts the lyrics analysis in a separate background thread.
        Passes an optional callback function (e.g., for storing data)
        to the background thread. Returns immediately with a status message.

        Args:
            cleaned_lyrics: The lyrics text to analyze.
            storage_callback: A callable function that accepts a dict
                              (the analysis data for one line), typically
                              used for storing the result (e.g.,
                              `SongAnalysisStorage.add_analysis_line`).
        """
        if not cleaned_lyrics or cleaned_lyrics.isspace():
             print("LLMAnalysis: No lyrics provided, skipping analysis.")
             return {"status": "No lyrics provided"}

        print("LLMAnalysis: Received request. Starting analysis in background thread...")
        try:
            analysis_thread = threading.Thread(
                target=self._perform_analysis_thread,
                # *** PASS storage_callback TO THREAD TARGET ***
                args=(cleaned_lyrics, storage_callback),
                # *********************************************
                daemon=True
            )
            analysis_thread.start()
            return {"status": "Analysis started in background"}

        except Exception as e:
            print(f"LLMAnalysis: Error starting analysis thread: {e}")
            traceback.print_exc()
            return {"status": "Error starting analysis", "error": str(e)}