"""Costanti di dominio condivise tra moduli."""

from __future__ import annotations

# Sotto i 5 secondi uno schermo nero e' quasi sempre una transizione voluta.
BLACK_MIN_DURATION_SECONDS = 5.0

# Timeout di sicurezza oltre il timeout HTTP della singola finestra:
# evita che un future bloccato fermi l'intera analisi.
WINDOW_FUTURE_TIMEOUT_SECONDS = 930.0
