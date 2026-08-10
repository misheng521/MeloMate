import sys
import os
import time

import edge_tts
from loguru import logger
from .tts_interface import TTSInterface

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# Check out doc at https://github.com/rany2/edge-tts
# Use `edge-tts --list-voices` to list all available voices


class TTSEngine(TTSInterface):
    RETRY_DELAYS = (0.45, 1.2)
    MIN_AUDIO_BYTES = 128

    def __init__(self, voice="en-US-AvaMultilingualNeural"):
        self.voice = voice

        self.temp_audio_file = "temp"
        self.file_extension = "mp3"
        self.new_audio_dir = "cache"

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

    def generate_audio(self, text, file_name_no_ext=None):
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext: str
            name of the file without extension


        Returns:
        str: the path to the generated audio file

        """
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)
        last_error = None

        for attempt in range(len(self.RETRY_DELAYS) + 1):
            try:
                if os.path.exists(file_name):
                    os.remove(file_name)
                communicate = edge_tts.Communicate(text, self.voice)
                communicate.save_sync(file_name)
                if not os.path.exists(file_name) or os.path.getsize(file_name) < self.MIN_AUDIO_BYTES:
                    raise RuntimeError("Edge TTS returned an empty or incomplete audio file.")
                if attempt:
                    logger.info(f"Edge TTS recovered after {attempt + 1} attempts.")
                return file_name
            except Exception as error:
                last_error = error
                if os.path.exists(file_name):
                    try:
                        os.remove(file_name)
                    except OSError:
                        pass
                if attempt < len(self.RETRY_DELAYS):
                    delay = self.RETRY_DELAYS[attempt]
                    logger.warning(
                        f"Edge TTS attempt {attempt + 1} failed: {error}. "
                        f"Retrying in {delay:.2f}s."
                    )
                    time.sleep(delay)

        logger.critical(f"\nError: edge-tts unable to generate audio after retries: {last_error}")
        logger.critical("The Edge speech service may be temporarily unavailable or blocked.")
        return None


# en-US-AvaMultilingualNeural
# en-US-EmmaMultilingualNeural
# en-US-JennyNeural
