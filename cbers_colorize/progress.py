from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import sys
import time


ProgressCallback = Callable[[float, str], None]
# frac01 em 0..1, mensagem opcional


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass
class Progress:
    """
    Progresso simples para CLI e integração futura com QGIS/plugin.

    - Se callback for fornecido, delega ao callback.
    - Caso contrário, imprime no stderr.
    - throttle_s evita spam excessivo.
    """
    enabled: bool = True
    callback: Optional[ProgressCallback] = None
    throttle_s: float = 0.2
    prefix: str = "[PROG]"

    _last_emit: float = field(default=0.0, init=False)
    _last_frac: float = field(default=-1.0, init=False)
    _last_msg: str = field(default="", init=False)

    def emit(self, frac01: float, msg: str = "") -> None:
        if not self.enabled:
            return

        frac01 = _clip01(frac01)
        msg = str(msg or "")

        now = time.time()
        same_payload = (abs(frac01 - self._last_frac) < 1e-9) and (msg == self._last_msg)
        throttled = ((now - self._last_emit) < float(self.throttle_s)) and frac01 < 1.0

        if same_payload:
            return
        if throttled:
            return

        self._last_emit = now
        self._last_frac = frac01
        self._last_msg = msg

        if self.callback is not None:
            self.callback(frac01, msg)
            return

        pct = int(frac01 * 100.0 + 0.5)
        line = f"{self.prefix} {pct:3d}%"
        if msg:
            line += f" | {msg}"
        print(line, file=sys.stderr, flush=True)

    def step(self, i: int, n: int, msg: str = "") -> None:
        if n <= 0:
            self.emit(0.0, msg)
            return
        self.emit(float(i) / float(n), msg)

    def phase(self, phase_idx: int, num_phases: int, local_frac01: float = 0.0, msg: str = "") -> None:
        if num_phases <= 0:
            self.emit(0.0, msg)
            return

        phase_idx = max(0, min(int(phase_idx), int(num_phases)))
        local_frac01 = _clip01(local_frac01)

        base = float(phase_idx) / float(num_phases)
        span = 1.0 / float(num_phases)
        frac = base + span * local_frac01
        self.emit(frac, msg)

    def subrange(self, start_frac01: float, end_frac01: float) -> "SubProgress":
        return SubProgress(parent=self, start_frac01=start_frac01, end_frac01=end_frac01)

    def done(self, msg: str = "done") -> None:
        self.emit(1.0, msg)


@dataclass
class SubProgress:
    """
    Mapeia progresso local 0..1 para uma faixa do progresso pai.
    """
    parent: Progress | "SubProgress"
    start_frac01: float
    end_frac01: float

    def _map(self, frac01: float) -> float:
        frac01 = _clip01(frac01)
        a = _clip01(self.start_frac01)
        b = _clip01(self.end_frac01)
        if b < a:
            a, b = b, a
        return a + (b - a) * frac01

    def emit(self, frac01: float, msg: str = "") -> None:
        self.parent.emit(self._map(frac01), msg)

    def step(self, i: int, n: int, msg: str = "") -> None:
        if n <= 0:
            self.emit(0.0, msg)
            return
        self.emit(float(i) / float(n), msg)

    def subrange(self, start_frac01: float, end_frac01: float) -> "SubProgress":
        """
        Permite encadear subfaixas sobre uma subfaixa existente.
        """
        a = self._map(start_frac01)
        b = self._map(end_frac01)
        return SubProgress(parent=self.parent, start_frac01=a, end_frac01=b)

    def done(self, msg: str = "done") -> None:
        self.emit(1.0, msg)