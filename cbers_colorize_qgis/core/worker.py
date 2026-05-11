# -*- coding: utf-8 -*-
from __future__ import annotations

from qgis.PyQt.QtCore import QObject, QProcess, QTimer, pyqtSignal


class DockerProcessWorker(QObject):
    log_line = pyqtSignal(str)
    progress_changed = pyqtSignal(int)
    finished_ok = pyqtSignal(dict)
    finished_error = pyqtSignal(int, dict)
    elapsed_changed = pyqtSignal(int)
    started_ok = pyqtSignal()
    start_failed = pyqtSignal(str)

    def __init__(self, parser, parent=None):
        super().__init__(parent)
        self.parser = parser
        self.process: QProcess | None = None
        self.timer: QTimer | None = None
        self.elapsed_seconds = 0
        self._running = False
        self._last_progress = 0

    def start(self, cmd: list[str]):
        if self._running:
            self.start_failed.emit("Já existe um processo em execução.")
            return

        self.elapsed_seconds = 0
        self._last_progress = 0

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_ready_read)
        self.process.finished.connect(self._on_finished)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self.process.start(cmd[0], cmd[1:])

        if not self.process.waitForStarted(3000):
            self.process = None
            self.timer = None
            self._running = False
            self.start_failed.emit("Falha ao iniciar o processo Docker.")
            return

        self._running = True
        self.timer.start(1000)
        self.progress_changed.emit(0)
        self.started_ok.emit()

    def cancel(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.log_line.emit("[PLUGIN] Cancelando processo...")
            self.process.kill()
            self.process.waitForFinished(3000)

        if self.timer:
            self.timer.stop()

        self._running = False

    def is_running(self) -> bool:
        return self._running

    def _on_timer_tick(self):
        self.elapsed_seconds += 1
        self.elapsed_changed.emit(self.elapsed_seconds)

    def _on_ready_read(self):
        if not self.process:
            return

        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return

        for line in data.splitlines():
            self.log_line.emit(line)
            self.parser.feed_line(line)

            p = int(self.parser.progress)
            if p != self._last_progress:
                self._last_progress = p
                self.progress_changed.emit(p)

    def _on_finished(self, exit_code, exit_status):
        if self.timer:
            self.timer.stop()

        self._running = False
        summary = self.parser.summary()

        if exit_code == 0:
            self.progress_changed.emit(100)
            self.finished_ok.emit(summary)
        else:
            self.finished_error.emit(exit_code, summary)