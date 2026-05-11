# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)


LOAD_MODE_NONE = "Não carregar"
LOAD_MODE_MASTER = "Carregar master"
LOAD_MODE_COG = "Carregar COG"
LOAD_MODE_BOTH = "Carregar ambos"


class CBERSColorizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CBERS Colorize")
        self.resize(1120, 820)
        self.setMinimumSize(1040, 760)
        self._build_ui()

    def _set_row_widget_height(self, *widgets, h: int = 30):
        for w in widgets:
            w.setMinimumHeight(h)
            w.setMaximumHeight(h)

    def _make_browse_button(self) -> QPushButton:
        btn = QPushButton("...")
        btn.setFixedSize(40, 30)
        return btn

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ============================================================
        # ÁREA SUPERIOR ROLÁVEL
        # ============================================================
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QScrollArea.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        # =========================
        # Ambiente
        # =========================
        grp_env = QGroupBox("Ambiente")
        env_layout = QGridLayout(grp_env)
        env_layout.setContentsMargins(8, 8, 8, 8)
        env_layout.setHorizontalSpacing(10)
        env_layout.setVerticalSpacing(6)

        self.lbl_env_docker = QLabel("Docker: não verificado")
        self.lbl_env_cuda = QLabel("CUDA: não verificado")
        self.lbl_env_image = QLabel("Imagem: não verificada")
        self.lbl_env_io = QLabel("I/O: não verificado")

        for lbl in (self.lbl_env_docker, self.lbl_env_cuda, self.lbl_env_image, self.lbl_env_io):
            lbl.setWordWrap(False)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            lbl.setMinimumHeight(22)

        self.btn_check_env = QPushButton("Verificar ambiente")
        self.btn_check_env.setMinimumSize(190, 32)
        self.btn_check_env.setMaximumHeight(32)

        env_layout.addWidget(self.lbl_env_docker, 0, 0)
        env_layout.addWidget(self.lbl_env_cuda, 0, 1)
        env_layout.addWidget(self.btn_check_env, 0, 2, 2, 1)
        env_layout.addWidget(self.lbl_env_image, 1, 0, 1, 2)
        env_layout.addWidget(self.lbl_env_io, 2, 0, 1, 3)

        env_layout.setColumnStretch(0, 3)
        env_layout.setColumnStretch(1, 2)
        env_layout.setColumnStretch(2, 0)

        scroll_layout.addWidget(grp_env)

        # =========================
        # Autodetecção
        # =========================
        grp_auto = QGroupBox("Autodetecção")
        auto_layout = QGridLayout(grp_auto)
        auto_layout.setContentsMargins(8, 8, 8, 8)
        auto_layout.setHorizontalSpacing(8)
        auto_layout.setVerticalSpacing(6)

        self.le_autodetect_dir = QLineEdit()
        self._set_row_widget_height(self.le_autodetect_dir)

        btn_autodetect_dir = self._make_browse_button()
        self.btn_autodetect = QPushButton("Auto detectar BAND0/BAND1/BAND2/BAND3")
        self.btn_autodetect.setMinimumHeight(32)

        btn_autodetect_dir.clicked.connect(self._pick_autodetect_directory)
        self.btn_autodetect.clicked.connect(self._auto_detect_bands)

        auto_layout.addWidget(QLabel("Pasta para autodetecção"), 0, 0)
        auto_layout.addWidget(self.le_autodetect_dir, 0, 1)
        auto_layout.addWidget(btn_autodetect_dir, 0, 2)
        auto_layout.addWidget(self.btn_autodetect, 1, 1, 1, 2)
        auto_layout.setColumnStretch(1, 1)

        scroll_layout.addWidget(grp_auto)

        # =========================
        # Entradas
        # =========================
        grp_inputs = QGroupBox("Entradas")
        inputs_layout = QGridLayout(grp_inputs)
        inputs_layout.setContentsMargins(8, 8, 8, 8)
        inputs_layout.setHorizontalSpacing(8)
        inputs_layout.setVerticalSpacing(6)

        self.le_pan = QLineEdit()
        self.le_blue = QLineEdit()
        self.le_green = QLineEdit()
        self.le_red = QLineEdit()

        self._set_row_widget_height(self.le_pan, self.le_blue, self.le_green, self.le_red)

        btn_pan = self._make_browse_button()
        btn_blue = self._make_browse_button()
        btn_green = self._make_browse_button()
        btn_red = self._make_browse_button()

        btn_pan.clicked.connect(lambda: self._pick_file(self.le_pan, "Selecione a Band0 (PAN)"))
        btn_blue.clicked.connect(lambda: self._pick_file(self.le_blue, "Selecione a Band1 (Blue)"))
        btn_green.clicked.connect(lambda: self._pick_file(self.le_green, "Selecione a Band2 (Green)"))
        btn_red.clicked.connect(lambda: self._pick_file(self.le_red, "Selecione a Band3 (Red)"))

        inputs_layout.addWidget(QLabel("Band0 (PAN)"), 0, 0)
        inputs_layout.addWidget(self.le_pan, 0, 1)
        inputs_layout.addWidget(btn_pan, 0, 2)

        inputs_layout.addWidget(QLabel("Band1 (Blue)"), 1, 0)
        inputs_layout.addWidget(self.le_blue, 1, 1)
        inputs_layout.addWidget(btn_blue, 1, 2)

        inputs_layout.addWidget(QLabel("Band2 (Green)"), 2, 0)
        inputs_layout.addWidget(self.le_green, 2, 1)
        inputs_layout.addWidget(btn_green, 2, 2)

        inputs_layout.addWidget(QLabel("Band3 (Red)"), 3, 0)
        inputs_layout.addWidget(self.le_red, 3, 1)
        inputs_layout.addWidget(btn_red, 3, 2)

        inputs_layout.setColumnStretch(1, 1)

        scroll_layout.addWidget(grp_inputs)

        # =========================
        # Saída
        # =========================
        grp_output = QGroupBox("Saída")
        output_layout = QGridLayout(grp_output)
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.setHorizontalSpacing(8)
        output_layout.setVerticalSpacing(6)

        self.le_outdir = QLineEdit()
        self._set_row_widget_height(self.le_outdir)

        btn_outdir = self._make_browse_button()
        self.btn_open_outdir = QPushButton("Abrir pasta de saída")
        self.btn_open_outdir.setMinimumHeight(32)

        btn_outdir.clicked.connect(self._pick_directory)

        output_layout.addWidget(QLabel("Diretório de saída"), 0, 0)
        output_layout.addWidget(self.le_outdir, 0, 1)
        output_layout.addWidget(btn_outdir, 0, 2)
        output_layout.addWidget(self.btn_open_outdir, 1, 1, 1, 2)
        output_layout.setColumnStretch(1, 1)

        scroll_layout.addWidget(grp_output)

        # =========================
        # Processamento
        # =========================
        grp_proc = QGroupBox("Processamento")
        proc_layout = QGridLayout(grp_proc)
        proc_layout.setContentsMargins(8, 8, 8, 8)
        proc_layout.setHorizontalSpacing(8)
        proc_layout.setVerticalSpacing(6)

        self.le_image = QLineEdit("cbers-colorize:gpu")
        self._set_row_widget_height(self.le_image)

        self.btn_detect_image = QPushButton("Detectar imagem")
        self.btn_copy_command = QPushButton("Copiar comando Docker")
        self.btn_detect_image.setMinimumHeight(32)
        self.btn_copy_command.setMinimumHeight(32)

        self.chk_export_cog = QCheckBox("Exportar COG")
        self.chk_export_cog.setChecked(True)

        self.chk_keep_tmp = QCheckBox("Manter temporários")
        self.chk_keep_tmp.setChecked(False)

        self.chk_verbose = QCheckBox("Verbose")
        self.chk_verbose.setChecked(True)

        proc_layout.addWidget(QLabel("Imagem Docker"), 0, 0)
        proc_layout.addWidget(self.le_image, 0, 1)
        proc_layout.addWidget(self.btn_detect_image, 0, 2)

        proc_layout.addWidget(self.chk_export_cog, 1, 0)
        proc_layout.addWidget(self.chk_keep_tmp, 1, 1)
        proc_layout.addWidget(self.chk_verbose, 1, 2)

        proc_layout.addWidget(self.btn_copy_command, 2, 1, 1, 2)
        proc_layout.setColumnStretch(1, 1)

        scroll_layout.addWidget(grp_proc)

        # =========================
        # Pós-processamento no QGIS
        # =========================
        grp_post = QGroupBox("Pós-processamento no QGIS")
        post_layout = QGridLayout(grp_post)
        post_layout.setContentsMargins(8, 8, 8, 8)
        post_layout.setHorizontalSpacing(8)
        post_layout.setVerticalSpacing(6)

        self.cb_load_mode = QComboBox()
        self.cb_load_mode.addItems([
            LOAD_MODE_NONE,
            LOAD_MODE_MASTER,
            LOAD_MODE_COG,
            LOAD_MODE_BOTH,
        ])
        self.cb_load_mode.setCurrentText(LOAD_MODE_MASTER)
        self._set_row_widget_height(self.cb_load_mode)

        post_layout.addWidget(QLabel("Ao finalizar"), 0, 0)
        post_layout.addWidget(self.cb_load_mode, 0, 1, 1, 2)
        post_layout.setColumnStretch(1, 1)

        scroll_layout.addWidget(grp_post)

        # =========================
        # Execução
        # =========================
        grp_exec = QGroupBox("Execução")
        exec_outer_layout = QVBoxLayout(grp_exec)
        exec_outer_layout.setContentsMargins(8, 8, 8, 8)
        exec_outer_layout.setSpacing(6)

        exec_top = QHBoxLayout()
        exec_top.setSpacing(8)

        self.lbl_elapsed = QLabel("Tempo decorrido: 00:00:00")
        self.lbl_status = QLabel("Status: parado")

        self.btn_validate = QPushButton("Validar entradas")
        self.btn_run = QPushButton("Executar")
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)

        for btn in (self.btn_validate, self.btn_run, self.btn_cancel):
            btn.setMinimumHeight(32)

        exec_top.addWidget(self.lbl_elapsed)
        exec_top.addStretch()
        exec_top.addWidget(self.lbl_status)
        exec_top.addWidget(self.btn_validate)
        exec_top.addWidget(self.btn_run)
        exec_top.addWidget(self.btn_cancel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(24)

        self.lbl_progress = QLabel("Progresso: 0%")

        exec_outer_layout.addLayout(exec_top)
        exec_outer_layout.addWidget(self.progress_bar)
        exec_outer_layout.addWidget(self.lbl_progress)

        scroll_layout.addWidget(grp_exec)
        scroll_layout.addStretch(1)

        self.scroll_area.setWidget(scroll_content)

        # ============================================================
        # PARTE INFERIOR FIXA
        # ============================================================
        bottom_splitter = QSplitter(Qt.Vertical)

        grp_log = QGroupBox("Log")
        log_layout = QVBoxLayout(grp_log)
        log_layout.setContentsMargins(8, 8, 8, 8)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(170)
        self.txt_log.setMaximumBlockCount(5000)
        log_layout.addWidget(self.txt_log)

        grp_summary = QGroupBox("Resumo TIME / QA / Saídas")
        summary_layout = QVBoxLayout(grp_summary)
        summary_layout.setContentsMargins(8, 8, 8, 8)

        self.tabs_summary = QTabWidget()

        tab_times = QWidget()
        tab_times_layout = QVBoxLayout(tab_times)
        tab_times_layout.setContentsMargins(0, 0, 0, 0)

        self.tbl_times = QTableWidget(0, 2)
        self.tbl_times.setHorizontalHeaderLabels(["Etapa", "Tempo"])
        self.tbl_times.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_times.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_times.verticalHeader().setVisible(False)
        self.tbl_times.setAlternatingRowColors(True)
        tab_times_layout.addWidget(self.tbl_times)

        tab_qa = QWidget()
        tab_qa_layout = QVBoxLayout(tab_qa)
        tab_qa_layout.setContentsMargins(0, 0, 0, 0)

        self.tbl_qa = QTableWidget(0, 2)
        self.tbl_qa.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.tbl_qa.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_qa.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_qa.verticalHeader().setVisible(False)
        self.tbl_qa.setAlternatingRowColors(True)
        tab_qa_layout.addWidget(self.tbl_qa)

        tab_outputs = QWidget()
        tab_outputs_layout = QVBoxLayout(tab_outputs)
        tab_outputs_layout.setContentsMargins(0, 0, 0, 0)

        self.tbl_outputs = QTableWidget(0, 2)
        self.tbl_outputs.setHorizontalHeaderLabels(["Campo", "Valor"])
        self.tbl_outputs.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_outputs.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_outputs.verticalHeader().setVisible(False)
        self.tbl_outputs.setAlternatingRowColors(True)
        tab_outputs_layout.addWidget(self.tbl_outputs)

        self.tabs_summary.addTab(tab_times, "TIME")
        self.tabs_summary.addTab(tab_qa, "QA")
        self.tabs_summary.addTab(tab_outputs, "Saídas")

        summary_layout.addWidget(self.tabs_summary)

        bottom_splitter.addWidget(grp_log)
        bottom_splitter.addWidget(grp_summary)
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 2)
        bottom_splitter.setSizes([230, 220])

        # ============================================================
        # LAYOUT FINAL
        # ============================================================
        root.addWidget(self.scroll_area, 3)
        root.addWidget(bottom_splitter, 2)

    def _pick_file(self, target_lineedit: QLineEdit, title: str):
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "Raster (*.tif *.tiff *.vrt);;Todos (*.*)"
        )
        if path:
            target_lineedit.setText(path)

    def _pick_directory(self):
        path = QFileDialog.getExistingDirectory(self, "Selecione o diretório de saída")
        if path:
            self.le_outdir.setText(path)

    def _pick_autodetect_directory(self):
        path = QFileDialog.getExistingDirectory(self, "Selecione a pasta das bandas")
        if path:
            self.le_autodetect_dir.setText(path)
            self._suggest_output_dir_from_autodetect(path)

    def _suggest_output_dir_from_autodetect(self, autodetect_dir: str):
        if self.le_outdir.text().strip():
            return
        p = Path(autodetect_dir)
        suggested = p / "output_colorized"
        self.le_outdir.setText(str(suggested))

    def _auto_detect_bands(self):
        folder = self.le_autodetect_dir.text().strip()
        if not folder:
            QMessageBox.warning(self, "Autodetecção", "Selecione a pasta para autodetecção.")
            return

        d = Path(folder)
        if not d.exists():
            QMessageBox.warning(self, "Autodetecção", "A pasta informada não existe.")
            return

        tif_files = list(d.glob("*.tif")) + list(d.glob("*.tiff"))
        tif_files = sorted(set(tif_files))

        def find_band(band_suffix: str):
            for p in tif_files:
                name = p.name.upper()
                if band_suffix in name:
                    return p
            return None

        pan = find_band("BAND0")
        blue = find_band("BAND1")
        green = find_band("BAND2")
        red = find_band("BAND3")

        missing = []
        if pan is None:
            missing.append("BAND0")
        if blue is None:
            missing.append("BAND1")
        if green is None:
            missing.append("BAND2")
        if red is None:
            missing.append("BAND3")

        if missing:
            QMessageBox.warning(
                self,
                "Autodetecção",
                "Não foi possível localizar:\n- " + "\n- ".join(missing)
            )
            return

        self.le_pan.setText(str(pan))
        self.le_blue.setText(str(blue))
        self.le_green.setText(str(green))
        self.le_red.setText(str(red))

        QMessageBox.information(self, "Autodetecção", "Bandas detectadas com sucesso.")

    def current_load_mode(self) -> str:
        return self.cb_load_mode.currentText()

    def append_log(self, text: str):
        self.txt_log.appendPlainText(text)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self):
        self.txt_log.clear()

    def clear_summary(self):
        self.tbl_times.setRowCount(0)
        self.tbl_qa.setRowCount(0)
        self.tbl_outputs.setRowCount(0)

    def set_elapsed_seconds(self, seconds: int):
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        self.lbl_elapsed.setText(f"Tempo decorrido: {h:02d}:{m:02d}:{s:02d}")

    def set_progress(self, value: int):
        value = max(0, min(100, int(value)))
        self.progress_bar.setValue(value)
        self.lbl_progress.setText(f"Progresso: {value}%")

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_validate.setEnabled(not running)
        self.btn_autodetect.setEnabled(not running)
        self.btn_check_env.setEnabled(not running)
        self.btn_open_outdir.setEnabled(not running)
        self.btn_detect_image.setEnabled(not running)
        self.btn_copy_command.setEnabled(not running)

        self.lbl_status.setText("Status: executando" if running else "Status: parado")

        if not running and self.progress_bar.value() < 100:
            self.set_progress(0)

    def set_environment_status(self, docker_text: str, cuda_text: str, image_text: str, io_text: str):
        self.lbl_env_docker.setText(docker_text)
        self.lbl_env_cuda.setText(cuda_text)
        self.lbl_env_image.setText(image_text)
        self.lbl_env_io.setText(io_text)

    def copy_text_to_clipboard(self, text: str):
        QApplication.clipboard().setText(text)

    def set_summary(self, summary: dict):
        self.clear_summary()

        times = summary.get("times", {})
        total = summary.get("total_time", None)

        time_rows = list(times.items())
        if total is not None:
            time_rows.append(("TOTAL", total))

        self.tbl_times.setRowCount(len(time_rows))
        for i, (k, v) in enumerate(time_rows):
            self.tbl_times.setItem(i, 0, QTableWidgetItem(str(k)))
            self.tbl_times.setItem(i, 1, QTableWidgetItem(str(v)))

        qa_rows = []
        for key in [
            "qa2_status", "qa2_n", "qa2_grad_mean", "qa2_lap_mean", "qa2_leak_mean",
            "qa2_skipped_low_valid", "qa2_skipped_small_mask", "qa2_text",
            "qa3_status", "qa3_n", "qa3_grad_mean", "qa3_lap_mean", "qa3_leak_mean",
            "qa3_skipped_low_valid", "qa3_skipped_small_mask", "qa3_text",
        ]:
            value = summary.get(key, None)
            if value not in (None, ""):
                qa_rows.append((key, value))

        self.tbl_qa.setRowCount(len(qa_rows))
        for i, (k, v) in enumerate(qa_rows):
            self.tbl_qa.setItem(i, 0, QTableWidgetItem(str(k)))
            self.tbl_qa.setItem(i, 1, QTableWidgetItem(str(v)))

        output_rows = []
        for key in ["device_used", "master_path", "cog_path"]:
            value = summary.get(key, None)
            if value not in (None, ""):
                output_rows.append((key, value))

        self.tbl_outputs.setRowCount(len(output_rows))
        for i, (k, v) in enumerate(output_rows):
            self.tbl_outputs.setItem(i, 0, QTableWidgetItem(str(k)))
            self.tbl_outputs.setItem(i, 1, QTableWidgetItem(str(v)))

        self.tbl_times.resizeRowsToContents()
        self.tbl_qa.resizeRowsToContents()
        self.tbl_outputs.resizeRowsToContents()