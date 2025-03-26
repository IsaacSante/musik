import threading
import time
import asyncio
import os # Make sure os is imported
from src.info.song_info import SongInfo
# Ensure this path is correct for your project structure
from src.analyzer.llm_analysis import LLMAnalysis

# This helper function remains useful to run the sync analyze_lyrics
# off the main async event loop thread using asyncio.to_thread.
async def analyze_lyrics_async(llm_analyzer: LLMAnalysis, lyrics: str):
    """
    Asynchronously calls LLMAnalysis.analyze_lyrics, which now starts a
    background thread for the actual analysis.
    Returns the status dict from analyze_lyrics.
    """
    # No callback needed here. analyze_lyrics internally handles output.
    start_status = await asyncio.to_thread(llm_analyzer.analyze_lyrics, lyrics)
    return start_status # Returns {"status": "..."}


def main():
    # --- Check for API Key early ---
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        return

    song_info = SongInfo(headless=True) # Or False for debugging
    try:
        # Instantiate LLMAnalysis once
        llm_analyzer = LLMAnalysis()
    except Exception as e:
        print(f"Fatal Error initializing LLMAnalysis: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Loading Spotify lyrics page...")
    try:
        song_info.load_site()
        print("Site loaded. Monitoring for song changes...")
    except Exception as e:
        print(f"Error loading Spotify page with Selenium: {e}")
        print("Check if ChromeDriver is installed and compatible with Chrome.")
        song_info.close()
        return

    stop_event = threading.Event()

    async def monitor_song_title():
        current_song = None
        last_lyrics = ""

        while not stop_event.is_set():
            try:
                updated_title = song_info.update_song_title()

                if updated_title is not None and updated_title != current_song:
                    current_song = updated_title
                    print("\n" + "=" * 40)
                    print("Current song:", current_song)
                    print("=" * 40)

                    time.sleep(1.5) # Allow lyrics page to update

                    lyrics = song_info.get_fullscreen_lyrics()

                    if lyrics and lyrics != last_lyrics:
                        print("Fetching lyrics and initiating background analysis...")
                        last_lyrics = lyrics

                        # --- Call analysis (starts the background thread) ---
                        analysis_status = await analyze_lyrics_async(llm_analyzer, lyrics)

                        # --- MINOR CHANGE HERE: Print the status, not results ---
                        print(f"LLM Analysis Status: {analysis_status.get('status', 'Unknown')}")
                        # The actual line-by-line results will be printed asynchronously
                        # by the background thread started within LLMAnalysis.
                        print("-" * 40)
                        # -------------------------------------------------------

                    elif not lyrics:
                        print("No lyrics found for this song.")
                        last_lyrics = ""
                    else:
                         print("Lyrics unchanged, skipping analysis initiation.")

            except Exception as e:
                 print(f"\nError during song monitoring loop: {e}")
                 import traceback
                 traceback.print_exc()
                 await asyncio.sleep(5) # Wait after error

            await asyncio.sleep(2) # Check periodically

    async def run_monitor():
        await monitor_song_title()

    monitor_thread = threading.Thread(target=lambda: asyncio.run(run_monitor()), daemon=True)
    monitor_thread.start()

    try:
        input("Monitoring Spotify lyrics. Press Enter to exit...\n")
    finally:
        print("Exiting...")
        stop_event.set()
        monitor_thread.join(timeout=5.0) # Wait briefly for thread
        if monitor_thread.is_alive():
             print("Warning: Monitor thread did not exit cleanly.")
        song_info.close()
        print("Browser closed.")


if __name__ == "__main__":
    main()