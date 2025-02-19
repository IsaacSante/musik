# main.py
import threading
import time
from src.info.song_info import SongInfo

def main():
    # Instantiate with headless=False to see the browser GUI.
    song_info = SongInfo(headless=False)
    
    # Launch the site.
    song_info.load_site()
    
    # Create an event to signal when to stop the monitoring thread.
    stop_event = threading.Event()

    def monitor_song_title():
        while not stop_event.is_set():
            updated_title = song_info.update_song_title()
            if updated_title is not None:
                print("Current song:", updated_title)
                print("-" * 40)
                time.sleep(.2)
                print(song_info.get_fullscreen_lyrics())
                print("-" * 40)
            time.sleep(.5)

    # Start the monitoring loop in a separate thread.
    monitor_thread = threading.Thread(target=monitor_song_title)
    monitor_thread.start()
    
    # Wait for the user to press Enter to exit.
    input("Press Enter to exit and close the browser...")
    
    # Signal the monitoring thread to stop and wait for it to finish.
    stop_event.set()
    monitor_thread.join()
    
    # Close the browser.
    song_info.close()

if __name__ == "__main__":
    main()
