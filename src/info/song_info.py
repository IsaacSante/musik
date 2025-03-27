import os
import getpass
import time
import threading # Import threading for the stop_event type hint
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from typing import Callable # Import Callable for type hinting the callback

class SongInfo:
    def __init__(self, headless=False):
        """
        Initializes the SongInfo instance.
        :param headless: Whether to run Chrome in headless mode.
        """
        self.headless = headless
        self.driver = None
        self.current_song_title = None
        self.song_title_history = []  # Maintains a Python-side history of song titles.
        # --- NEW: Store the assumed active lyric class name ---
        # WARNING: This class name might change with Spotify updates!
        self._active_lyric_class = "EhKgYshvOwpSrTv399Mw"

    def _initialize_driver(self):
        """
        Sets up the Chrome WebDriver with desired options, including a dynamically created user data directory.
        """
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")

        username = getpass.getuser()
        base_profile_dir = os.path.join(os.path.expanduser("~"), ".selenium_profiles")
        os.makedirs(base_profile_dir, exist_ok=True)
        user_profile_dir = os.path.join(base_profile_dir, f"chrome_profile_{username}")
        os.makedirs(user_profile_dir, exist_ok=True)

        chrome_options.add_argument(f"--user-data-dir={user_profile_dir}")
        # Optional: Add argument to allow insecure localhost if needed for certain setups, but be cautious
        # chrome_options.add_argument('--allow-running-insecure-content')
        # chrome_options.add_argument('--ignore-certificate-errors')

        self.driver = webdriver.Chrome(options=chrome_options)

    def load_site(self, url="https://open.spotify.com/lyrics"):
        """
        Loads the given URL using Selenium and returns the page source.

        :param url: The URL to load.
        :return: The page source as a string.
        """
        if self.driver is None:
            self._initialize_driver()
        print(f"Attempting to load URL: {url}")
        try:
            self.driver.get(url)
            print("URL loaded successfully.")
            # Optional: Add a small wait for initial elements to render
            time.sleep(2)
        except Exception as e:
            print(f"Error loading URL {url}: {e}")
            raise # Re-raise the exception to be handled by the caller
        return self.driver.page_source

    def get_song_title(self):
        attempts = 0
        while attempts < 3:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="context-item-link"]')
                return element.text.strip()
            except StaleElementReferenceException:
                attempts += 1
                time.sleep(0.1)  # short wait before retrying
            except NoSuchElementException:
                return None # Element not found
            except Exception as e:
                 print(f"Unexpected error in get_song_title: {e}")
                 return None # Treat other errors as element not found for simplicity here
        return None # Return None after multiple attempts

    def update_song_title(self):
        new_title = self.get_song_title()
        # Check if new_title is None or empty, or if it hasn't changed
        if not new_title or new_title == self.current_song_title:
            return None
        # New title detected.
        self.current_song_title = new_title
        self.song_title_history.append(new_title)
        return new_title

    def clean_lyrics(self, lyrics: str) -> str:
        """
        Cleans the lyrics text by removing unwanted characters, such as music symbols.

        :param lyrics: The original lyrics string.
        :return: The cleaned lyrics string.
        """
        cleaned = lyrics.replace("♪", "")
        cleaned_lines = [line.strip() for line in cleaned.splitlines() if line.strip()] # Also remove empty lines
        return "\n".join(cleaned_lines)

    def get_fullscreen_lyrics(self):
        """
        Finds all divs with data-testid="fullscreen-lyric",
        extracts the text content from their inner div,
        and returns all text as one block with each piece on a new line.
        The returned lyrics are cleaned by removing unwanted symbols.
        """
        lyrics = []
        try:
            # Locate all lyric containers
            lyric_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="fullscreen-lyric"]')
            for elem in lyric_elements:
                try:
                    # Get the inner div that holds the actual lyric text
                    inner_div = elem.find_element(By.XPATH, './div')
                    text = inner_div.text.strip()
                    if text:
                        lyrics.append(text)
                except (NoSuchElementException, StaleElementReferenceException):
                    # Skip if inner div not found or element is stale
                    continue
                except Exception as e:
                     print(f"Minor error getting text from one lyric element: {e}")
                     continue # Skip this element
        except Exception as e:
            print(f"Error finding fullscreen lyric elements: {e}")
            return "" # Return empty string if main selector fails

        # Join the lyrics and clean them before returning
        full_lyrics = "\n".join(lyrics)
        return self.clean_lyrics(full_lyrics)

    # --- NEW METHOD ---
    def monitor_current_lyric(self, new_lyric_callback: Callable[[str], None], stop_event: threading.Event):
        """
        Monitors the lyrics view for the currently active lyric line and calls the callback when a new line becomes active.

        This method runs in a loop until the stop_event is set. It should be run in a separate thread.

        :param new_lyric_callback: A function to call with the text of the newly active lyric line.
        :param stop_event: A threading.Event object used to signal when to stop monitoring.
        """
        if not self.driver:
            print("Error: Driver not initialized. Cannot monitor lyrics.")
            return

        last_active_lyric_text = None
        # More specific selector to find the div that is *both* a lyric line *and* has the active class
        active_lyric_selector = f'div[data-testid="fullscreen-lyric"].{self._active_lyric_class}'

        print("Starting current lyric monitoring...")
        while not stop_event.is_set():
            current_active_lyric_text = None
            try:
                # Find elements that *currently* have the active class
                active_elements = self.driver.find_elements(By.CSS_SELECTOR, active_lyric_selector)

                if active_elements:
                    # Spotify usually highlights only one line, but if multiple match, take the last one found in the DOM.
                    # This is a heuristic that often corresponds to the most recently activated line.
                    target_element = active_elements[-1]
                    try:
                        # Get the inner div's text, same as in get_fullscreen_lyrics
                        inner_div = target_element.find_element(By.XPATH, './div')
                        current_active_lyric_text = inner_div.text.strip()
                    except (NoSuchElementException, StaleElementReferenceException):
                        # Element might have gone stale or structure changed slightly; skip this cycle
                        # print("Debug: Inner div not found or stale for active element.") # Optional debug
                        pass # Continue to next iteration

                # Check if a *new* line has become active (and is not empty)
                if current_active_lyric_text and current_active_lyric_text != last_active_lyric_text:
                    # print(f"DEBUG: New lyric detected: '{current_active_lyric_text}'") # Optional debug print
                    cleaned_lyric = self.clean_lyrics(current_active_lyric_text) # Clean the single line
                    if cleaned_lyric: # Ensure it's not empty after cleaning
                        try:
                            new_lyric_callback(cleaned_lyric) # Call the provided callback
                        except Exception as cb_err:
                            print(f"Error executing new_lyric_callback: {cb_err}")
                        last_active_lyric_text = current_active_lyric_text # Update the last known active lyric

                # If no active lyric is found currently, but there *was* one before, reset.
                elif not current_active_lyric_text and last_active_lyric_text is not None:
                    # print("DEBUG: Active lyric lost.") # Optional debug
                    last_active_lyric_text = None # Reset state

            except StaleElementReferenceException:
                 # print("Debug: Stale element reference during active lyric search, retrying.") # Optional debug
                 pass # Page structure might be changing, just try again next iteration
            except NoSuchElementException:
                 # This might happen if the lyrics view itself is gone.
                 # print("Debug: Lyric container likely not present.") # Optional debug
                 last_active_lyric_text = None # Reset state
                 time.sleep(1) # Wait a bit longer if the container isn't there
            except Exception as e:
                # Catch other potential Selenium or unexpected errors
                print(f"Unexpected error in lyric monitoring loop: {e}")
                # import traceback # Optional: uncomment for detailed stack trace
                # traceback.print_exc()
                last_active_lyric_text = None # Reset state on error
                time.sleep(1) # Wait a bit longer after an unexpected error

            # --- Polling Interval ---
            # Adjust if needed. Shorter is more responsive but uses more CPU.
            # Longer is less resource-intensive but might miss very quick changes.
            time.sleep(0.3) # Check roughly 3 times per second

        print("Current lyric monitoring stopped.")


    def close(self):
        """
        Closes the Selenium WebDriver.
        """
        if self.driver:
            try:
                self.driver.quit()
                print("Browser closed via SongInfo.close().")
            except Exception as e:
                print(f"Error during driver quit: {e}")
            finally:
                self.driver = None