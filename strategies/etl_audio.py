"""Estrategias de transcripción de audio/video.

Todas retornan la misma forma: List[{"start": float, "end": float,
"text": str}]. Para video, las estrategias que usan una API externa
extraen primero la pista de audio (definitions.extract_audio_track) ya
que las APIs solo garantizan soporte de contenedores de audio puro;
faster-whisper local decodifica el video directamente, sin ese paso.
"""
from typing import Any, Dict, List

from strategies.base import AudioTranscriptionStrategy


class FasterWhisperLocalStrategy(AudioTranscriptionStrategy):
    """faster-whisper local (ya en uso). Gratis, decodifica video
    directamente sin extracción de audio previa."""

    async def transcribe(self, file_path: str) -> List[Dict[str, Any]]:
        import definitions

        return await definitions.transcribe_audio_or_video(file_path)


class OpenAIWhisperAPIStrategy(AudioTranscriptionStrategy):
    """OpenAI Whisper API (whisper-1), con timestamps por segmento."""

    def __init__(self):
        from settings import settings

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY no está configurado. Requerido para la "
                "estrategia de audio 'whisper_api'."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def transcribe(self, file_path: str) -> List[Dict[str, Any]]:
        import asyncio
        import definitions

        audio_path = file_path
        if definitions.classify_file_type(file_path) == "video":
            audio_path = str(definitions.extract_audio_track(file_path))

        def _call():
            with open(audio_path, "rb") as f:
                result = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            return [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in (result.segments or [])
            ]

        return await asyncio.get_event_loop().run_in_executor(None, _call)


class DeepgramStrategy(AudioTranscriptionStrategy):
    """Deepgram — transcripción de alta velocidad con diarización de
    hablantes (cada hablante se antepone al texto de su intervención)."""

    def __init__(self):
        from settings import settings

        if not settings.DEEPGRAM_API_KEY:
            raise ValueError(
                "DEEPGRAM_API_KEY no está configurado. Requerido para la "
                "estrategia de audio 'deepgram'."
            )
        from deepgram import DeepgramClient

        self._client = DeepgramClient(api_key=settings.DEEPGRAM_API_KEY)

    async def transcribe(self, file_path: str) -> List[Dict[str, Any]]:
        import asyncio
        import definitions

        audio_path = file_path
        if definitions.classify_file_type(file_path) == "video":
            audio_path = str(definitions.extract_audio_track(file_path))

        def _call():
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            response = self._client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model="nova-2",
                smart_format=True,
                diarize=True,
                utterances=True,
            )
            utterances = (response.results.utterances or []) if response.results else []
            segments = []
            for u in utterances:
                text = u.transcript.strip()
                if u.speaker is not None:
                    text = f"[Hablante {u.speaker}] {text}"
                segments.append({"start": u.start, "end": u.end, "text": text})
            return segments

        return await asyncio.get_event_loop().run_in_executor(None, _call)
