# cbers_colorize/progress.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import sys
import time


ProgressCallback = Callable[[float, str], None]
# float: 0..1, str: mensagem


@dataclass
class Progress:
    """
    Progresso simples para CLI e para integração futura com QGIS.

    - Você pode passar `callback` para o plugin atualizar barra/mensagens.
    - Se callback=None, imprime no stderr (se `enabled=True`).
    """
    enabled: bool = True
    callback: Optional[ProgressCallback] = None
    throttle_s: float = 0.2

    _last_emit: float = 0.0

    def emit(self, frac01: float, msg: str = "") -> None:
        if not self.enabled:
            return

        frac01 = 0.0 if frac01 < 0 else 1.0 if frac01 > 1 else float(frac01)
        now = time.time()
        if (now - self._last_emit) < self.throttle_s and frac01 < 1.0:
            return
        self._last_emit = now

        if self.callback is not None:
            self.callback(frac01, msg)
            return

        # fallback CLI
        pct = int(frac01 * 100.0 + 0.5)
        line = f"[PROG] {pct:3d}%"
        if msg:
            line += f" | {msg}"
        print(line, file=sys.stderr, flush=True)

    def step(self, i: int, n: int, msg: str = "") -> None:
        if n <= 0:
            self.emit(0.0, msg)
        else:
            self.emit(i / float(n), msg)

    def done(self, msg: str = "done") -> None:
        self.emit(1.0, msg)