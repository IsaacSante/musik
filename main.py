import threading
import time
import asyncio
import os
from src.info.song_info import SongInfo
from src.analyzer.llm_analysis import LLMAnalysis

# Global variable to store the latest lyric line
latest_lyric_line = None
latest_lyric_lock = threading.Lock() # To safely update/read the latest lyric

async def analyze_lyrics_async(llm_analyzer: LLMAnalysis, lyrics: str):
    """
    Asynchronously calls LLMAnalysis.analyze_lyrics.
    """
    start_status = await asyncio.to_thread(llm_analyzer.analyze_lyrics, lyrics)
    return start_status

# --- NEW: Callback function for new lyrics ---
def handle_new_lyric(lyric_line: str):
    """
    This function will be called by the monitor_current_lyric thread
    whenever a new lyric line is detected.
    """
    global latest_lyric_line
    with latest_lyric_lock:
        latest_lyric_line = lyric_line
    # Example action: Just print the newly detected lyric line
    print(f"--> Current Lyric: {lyric_line}")
    # You could add more complex logic here, like sending it elsewhere,
    # triggering other actions, etc.


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        return

    song_info = SongInfo(headless=False) # Or False for debugging
    llm_analyzer = None # Initialize later after successful site load if needed
    monitor_thread = None
    lyric_monitor_thread = None # Thread for the new lyric monitor
    # Use separate stop events for clarity, though one could potentially be reused
    title_stop_event = threading.Event()
    lyric_stop_event = threading.Event()

    try:
        print("Loading Spotify lyrics page...")
        # Ensure site loads before initializing things that depend on it
        song_info.load_site()
        print("Site loaded.")

        # Initialize LLMAnalyzer only after site load is confirmed
        try:
            llm_analyzer = LLMAnalysis()
        except Exception as e:
            print(f"Fatal Error initializing LLMAnalysis: {e}")
            import traceback
            traceback.print_exc()
            raise # Re-raise to be caught by the outer try/finally

        print("Starting monitoring...")

        # --- Thread for Song Title and Full Lyrics ---
        async def monitor_song_title_and_lyrics():
            current_song = None
            last_lyrics_analyzed = "" # Track lyrics sent for analysis

            while not title_stop_event.is_set():
                try:
                    updated_title = song_info.update_song_title()

                    if updated_title is not None and updated_title != current_song:
                        current_song = updated_title
                        print("\n" + "=" * 40)
                        print("Current song:", current_song)
                        print("=" * 40)

                        # Reset analyzed lyrics when song changes
                        last_lyrics_analyzed = ""

                        # Allow time for lyrics page to potentially update after song change
                        await asyncio.sleep(1.5)

                        # Fetch full lyrics for analysis (only when song changes)
                        lyrics = song_info.get_fullscreen_lyrics()

                        if lyrics and lyrics != last_lyrics_analyzed:
                            print("Fetching full lyrics and initiating background analysis...")
                            last_lyrics_analyzed = lyrics

                            if llm_analyzer: # Check if analyzer was initialized
                                analysis_status = await analyze_lyrics_async(llm_analyzer, lyrics)
                                print(f"LLM Analysis Status: {analysis_status.get('status', 'Unknown')}")
                                print("-" * 40)
                            else:
                                print("LLM Analyzer not available, skipping analysis.")

                        elif not lyrics:
                            print("No full lyrics found for this song.")
                        else:
                            print("Full lyrics unchanged or already analyzed, skipping analysis initiation.")

                except Exception as e:
                    print(f"\nError during song title/lyrics monitoring loop: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(5) # Wait after error

                # Check periodically for title changes
                await asyncio.sleep(2)
            print("Song title/lyrics monitoring stopped.")


        # Run the async title monitor in its own thread
        monitor_thread = threading.Thread(
            target=lambda: asyncio.run(monitor_song_title_and_lyrics()),
            daemon=True
        )
        monitor_thread.start()

        # --- Thread for Current Lyric Line ---
        # This runs the synchronous, blocking monitor_current_lyric method
        lyric_monitor_thread = threading.Thread(
            target=song_info.monitor_current_lyric,
            args=(handle_new_lyric, lyric_stop_event), # Pass callback and stop event
            daemon=True
        )
        lyric_monitor_thread.start()

        # Keep main thread alive until user presses Enter
        input("Monitoring Spotify. Press Enter to exit...\n")

    except Exception as e:
         print(f"An error occurred during setup or monitoring: {e}")
         # Optional: Add more detailed error logging if needed
         # import traceback
         # traceback.print_exc()

    finally:
        print("Exiting...")

        # Signal threads to stop
        print("Signaling monitoring threads to stop...")
        title_stop_event.set()
        lyric_stop_event.set()

        # Wait for threads to finish
        if monitor_thread and monitor_thread.is_alive():
            print("Waiting for title monitor thread...")
            monitor_thread.join(timeout=5.0)
            if monitor_thread.is_alive():
                print("Warning: Title monitor thread did not exit cleanly.")

        if lyric_monitor_thread and lyric_monitor_thread.is_alive():
            print("Waiting for lyric monitor thread...")
            lyric_monitor_thread.join(timeout=5.0)
            if lyric_monitor_thread.is_alive():
                print("Warning: Lyric monitor thread did not exit cleanly.")

        # Close the browser
        print("Closing browser...")
        song_info.close() # Ensure browser is closed regardless of thread state

        print("Cleanup complete. Exited.")


if __name__ == "__main__":
    main()