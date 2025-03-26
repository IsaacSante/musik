# src/storage/song_analysis_storage.py
import threading
import string
# --- ADD THIS IMPORT ---
from typing import Union, Dict, List, Optional
# --- END ADD IMPORT ---

class SongAnalysisStorage:
    """
    Stores line-by-line analysis data (lyrics, VAT, etc.) for multiple songs,
    optimized for fast lookup by lyric text.
    """
    def __init__(self):
        # Key: song_title (str)
        # Value: dict { normalized_lyric_text (str): analysis_dict }
        # Using typing.Dict for clarity with older Python versions
        self.song_data: Dict[str, Dict[str, Dict]] = {}
        self.current_song_title: Optional[str] = None # Use Optional[str] instead of str | None
        self._lock = threading.Lock()
        print("SongAnalysisStorage initialized (Optimized for Lyric Lookup).")

    def _normalize_lyric(self, lyric_text: str) -> str:
        """Simple normalization: lowercase and remove punctuation/extra whitespace."""
        if not lyric_text:
            return ""
        translator = str.maketrans('', '', string.punctuation)
        normalized = lyric_text.translate(translator)
        normalized = normalized.lower().strip()
        return normalized

    def start_new_song(self, song_title: str):
        """
        Registers a new song title and prepares its storage (as a dictionary).
        Clears previous data for this title.
        """
        if not song_title:
            print("Storage Warning: Attempted to start analysis for an empty song title.")
            return

        with self._lock:
            print(f"Storage: Starting analysis collection for song: '{song_title}'")
            self.current_song_title = song_title
            self.song_data[song_title] = {}

    def add_analysis_line(self, analysis_data: dict):
        """
        Adds analysis data for a single line to the current song's dictionary,
        keyed by the normalized lyric text.
        """
        lyric = analysis_data.get('lyric')
        if not lyric:
            print(f"Storage Warning: Received analysis data without 'lyric' field. Ignored: {analysis_data}")
            return

        normalized_lyric = self._normalize_lyric(lyric)
        if not normalized_lyric:
             print(f"Storage Warning: Lyric '{lyric}' normalized to empty string. Ignored.")
             return

        with self._lock:
            if self.current_song_title is None:
                print(f"Storage Warning: add_analysis_line called but no current song. Lyric ignored: '{lyric}'")
                return

            if self.current_song_title not in self.song_data:
                print(f"Storage Warning: Current song '{self.current_song_title}' dict missing during add. Creating.")
                self.song_data[self.current_song_title] = {}

            # Store using normalized lyric as the key, overwriting duplicates
            self.song_data[self.current_song_title][normalized_lyric] = analysis_data

    # --- CHANGE TYPE HINT HERE ---
    def find_analysis_by_lyric(self, song_title: str, current_lyric_text: str) -> Union[dict, None]:
    # ---  WAS: dict | None ---
        """
        Finds the stored analysis data for a specific lyric within a given song.
        Uses normalized matching.

        Args:
            song_title: The title of the song.
            current_lyric_text: The text of the lyric being currently sung/displayed.

        Returns:
            The analysis dictionary (containing VAT etc.) or None if not found.
        """
        if not song_title or not current_lyric_text:
            return None

        normalized_lookup = self._normalize_lyric(current_lyric_text)
        if not normalized_lookup:
             return None

        with self._lock:
            song_analysis_dict = self.song_data.get(song_title)
            if song_analysis_dict:
                return song_analysis_dict.get(normalized_lookup)
            else:
                return None

    # --- CHANGE TYPE HINT HERE ---
    def get_analysis_dict_for_song(self, song_title: str) -> Union[Dict[str, Dict], None]:
    # --- WAS: dict | None (implicitly) and made more specific ---
         """Retrieves the dictionary {normalized_lyric: analysis_data} for a song."""
         with self._lock:
             # Using .get which naturally returns None if key not found
             return self.song_data.get(song_title)

    # --- CHANGE TYPE HINT HERE ---
    def get_current_song_title(self) -> Union[str, None]:
    # --- WAS: str | None ---
        with self._lock:
            return self.current_song_title

    def get_all_stored_songs(self) -> List[str]: # Use List from typing
        with self._lock:
            return list(self.song_data.keys())