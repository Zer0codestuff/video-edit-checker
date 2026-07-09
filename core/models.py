"""Tipi e schema condivisi per gli errori di montaggio."""

from __future__ import annotations

from dataclasses import dataclass


ERROR_TYPES = {
    "black_screen": "⬛ Schermo nero",
    "frozen_frame": "🧊 Frame congelato",
    "missed_cut": "✂️ Taglio mancato",
    "repeated_phrase": "🔁 Frase ripetuta",
    "audio_glitch": "🔇 Problema audio",
    "other": "⚠️ Altro",
}

# Tipi rilevabili da un modello che vede solo i frame / la clip (no audio).
VISUAL_ERROR_TYPES = frozenset({
    "black_screen", "frozen_frame", "missed_cut", "other",
})


@dataclass
class EditError:
    type: str
    start: float
    end: float
    description: str
    confidence: float

    @property
    def label(self) -> str:
        return ERROR_TYPES.get(self.type, ERROR_TYPES["other"])


# Schema JSON imposto al modello via grammatica llama.cpp.
# Mantenuto semplice per non confondere i modelli piccoli.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(ERROR_TYPES.keys())},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "description": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["type", "start", "end", "description", "confidence"],
            },
        },
    },
    "required": ["errors"],
}
