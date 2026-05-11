# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import os
import subprocess

from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.core import QgsProject, QgsRasterLayer

from .ui.cbers_colorize_dialog import (
    CBERSColorizeDialog,
    LOAD_MODE_MASTER,
    LOAD_MODE_COG,
    LOAD_MODE_BOTH,
)
from .core.docker_runner import (
    DockerRunConfig,
    build_docker_command,
    detect_docker,
    detect_cuda_support,
    detect_image_exists,
    detect_preferred_image,
    test_output_dir_writable,
    test_input_files_readable,
    find_docker_exe,
    get_docker_search_diagnostics,
)
from .core.log_parser import LogSummaryParser
from .core.worker import DockerProcessWorker
from .core.presets import PRESETS


SETTINGS_PREFIX = "CBERSColorize"


class CBERSColorizePlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = Path(__file__).resolve().parent
        self.action = None
        self.dlg = None
        self.worker = None
        self.parser = None
        self._last_device_used = None
        self._last_built_cmd = None

    def initGui(self):
        icon_path = str(self.plugin_dir / "icon.png")
        self.action = QAction(QIcon(icon_path), "CBERS Colorize", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu("&CBERS Colorize", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        if self.action:
            self.iface.removePluginMenu("&CBERS Colorize", self.action)
            self.iface.removeToolBarIcon(self.action)
        self.action = None

    def run(self):
        if self.dlg is None:
            self.dlg = CBERSColorizeDialog()

        self._connect_dialog_signals_once()
        self._load_settings_into_dialog()
        self._reset_dialog_runtime_state()
        self._auto_fill_preferred_image()
        self._check_environment()
        self.dlg.show()
        self.dlg.raise_()
        self.dlg.activateWindow()

    def _connect_dialog_signals_once(self):
        if getattr(self.dlg, "_signals_connected", False):
            return

        self.dlg.btn_run.clicked.connect(self._on_run_clicked)
        self.dlg.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.dlg.btn_validate.clicked.connect(self._on_validate_clicked)
        self.dlg.btn_check_env.clicked.connect(self._check_environment)
        self.dlg.btn_open_outdir.clicked.connect(self._open_output_folder)
        self.dlg.btn_detect_image.clicked.connect(self._auto_fill_preferred_image)
        self.dlg.btn_copy_command.clicked.connect(self._copy_last_command)
        self.dlg._signals_connected = True

    def _reset_dialog_runtime_state(self):
        self.dlg.clear_log()
        self.dlg.append_log("Plugin pronto.")
        self.dlg.set_running(False)
        self.dlg.set_elapsed_seconds(0)
        self.dlg.set_progress(0)
        self.dlg.clear_summary()

    def _settings(self) -> QSettings:
        return QSettings()

    def _load_settings_into_dialog(self):
        s = self._settings()
        self.dlg.le_pan.setText(s.value(f"{SETTINGS_PREFIX}/pan", "", type=str))
        self.dlg.le_blue.setText(s.value(f"{SETTINGS_PREFIX}/blue", "", type=str))
        self.dlg.le_green.setText(s.value(f"{SETTINGS_PREFIX}/green", "", type=str))
        self.dlg.le_red.setText(s.value(f"{SETTINGS_PREFIX}/red", "", type=str))
        self.dlg.le_outdir.setText(s.value(f"{SETTINGS_PREFIX}/outdir", "", type=str))
        self.dlg.le_image.setText(s.value(f"{SETTINGS_PREFIX}/image", "cbers-colorize:gpu", type=str))
        self.dlg.le_autodetect_dir.setText(s.value(f"{SETTINGS_PREFIX}/autodetect_dir", "", type=str))

        self.dlg.chk_export_cog.setChecked(s.value(f"{SETTINGS_PREFIX}/export_cog", True, type=bool))
        self.dlg.chk_keep_tmp.setChecked(s.value(f"{SETTINGS_PREFIX}/keep_tmp", False, type=bool))
        self.dlg.chk_verbose.setChecked(s.value(f"{SETTINGS_PREFIX}/verbose", True, type=bool))

        load_mode = s.value(f"{SETTINGS_PREFIX}/load_mode", "Carregar master", type=str)
        idx_load = self.dlg.cb_load_mode.findText(load_mode)
        if idx_load >= 0:
            self.dlg.cb_load_mode.setCurrentIndex(idx_load)

    def _save_settings_from_dialog(self):
        s = self._settings()
        s.setValue(f"{SETTINGS_PREFIX}/pan", self.dlg.le_pan.text().strip())
        s.setValue(f"{SETTINGS_PREFIX}/blue", self.dlg.le_blue.text().strip())
        s.setValue(f"{SETTINGS_PREFIX}/green", self.dlg.le_green.text().strip())
        s.setValue(f"{SETTINGS_PREFIX}/red", self.dlg.le_red.text().strip())
        s.setValue(f"{SETTINGS_PREFIX}/outdir", self.dlg.le_outdir.text().strip())
        s.setValue(f"{SETTINGS_PREFIX}/image", self.dlg.le_image.text().strip())
        s.setValue(f"{SETTINGS_PREFIX}/autodetect_dir", self.dlg.le_autodetect_dir.text().strip())
        s.setValue(f"{SETTINGS_PREFIX}/export_cog", self.dlg.chk_export_cog.isChecked())
        s.setValue(f"{SETTINGS_PREFIX}/keep_tmp", self.dlg.chk_keep_tmp.isChecked())
        s.setValue(f"{SETTINGS_PREFIX}/verbose", self.dlg.chk_verbose.isChecked())
        s.setValue(f"{SETTINGS_PREFIX}/load_mode", self.dlg.current_load_mode())

    def _fixed_current_preset(self) -> dict:
        return PRESETS.get("Atual") or next(iter(PRESETS.values()))

    def _auto_fill_preferred_image(self):
        current = self.dlg.le_image.text().strip()
        ok, _ = detect_image_exists(current) if current else (False, "")
        if ok:
            return

        image, msg = detect_preferred_image()
        if image:
            self.dlg.le_image.setText(image)
            self.dlg.append_log(f"[PLUGIN] {msg}")
        else:
            self.dlg.append_log(f"[PLUGIN] {msg}")

    def _validate_inputs(self) -> tuple[bool, str]:
        paths = {
            "Band0 (PAN)": self.dlg.le_pan.text().strip(),
            "Band1 (Blue)": self.dlg.le_blue.text().strip(),
            "Band2 (Green)": self.dlg.le_green.text().strip(),
            "Band3 (Red)": self.dlg.le_red.text().strip(),
            "Diretório de saída": self.dlg.le_outdir.text().strip(),
        }

        for label, value in paths.items():
            if not value:
                return False, f"Campo obrigatório não preenchido: {label}"

        for label in ["Band0 (PAN)", "Band1 (Blue)", "Band2 (Green)", "Band3 (Red)"]:
            if not Path(paths[label]).exists():
                return False, f"Arquivo não encontrado: {paths[label]}"

        outdir = Path(paths["Diretório de saída"])
        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"Não foi possível criar/acessar o diretório de saída:\n{outdir}\n\n{repr(e)}"

        return True, ""

    def _on_validate_clicked(self):
        ok, msg = self._validate_inputs()
        if not ok:
            QMessageBox.warning(self.dlg, "Validação", msg)
            return

        read_ok, read_msg = test_input_files_readable([
            self.dlg.le_pan.text().strip(),
            self.dlg.le_blue.text().strip(),
            self.dlg.le_green.text().strip(),
            self.dlg.le_red.text().strip(),
        ])
        write_ok, write_msg = test_output_dir_writable(self.dlg.le_outdir.text().strip())

        final_msg = [
            "Entradas básicas: OK",
            read_msg,
            write_msg,
        ]

        if read_ok and write_ok:
            QMessageBox.information(self.dlg, "Validação", "\n\n".join(final_msg))
        else:
            QMessageBox.warning(self.dlg, "Validação", "\n\n".join(final_msg))

    def _check_environment(self):
        image_name = self.dlg.le_image.text().strip() or "cbers-colorize:gpu"

        docker_ok, docker_msg = detect_docker()
        docker_exe = find_docker_exe()

        cuda_ok = detect_cuda_support() if docker_ok else False
        image_ok, image_msg = detect_image_exists(image_name) if docker_ok else (False, "Imagem não verificada.")

        io_text = "I/O: não verificado"
        if self.dlg.le_outdir.text().strip():
            io_ok, io_msg = test_output_dir_writable(self.dlg.le_outdir.text().strip())
            io_text = f"I/O: {'OK' if io_ok else 'FALHA'} | {io_msg}"

        docker_text = f"Docker: {'OK' if docker_ok else 'FALHA'}"
        if docker_msg:
            docker_text += f" | {docker_msg}"

        cuda_text = f"CUDA: {'OK' if cuda_ok else 'indisponível'}"

        image_text = f"Imagem: {'OK' if image_ok else 'FALHA'}"
        if image_msg:
            image_text += f" | {image_msg}"

        self.dlg.set_environment_status(docker_text, cuda_text, image_text, io_text)

        self.dlg.append_log("[PLUGIN] Ambiente verificado.")
        if docker_exe:
            self.dlg.append_log(f"[PLUGIN] Docker localizado em: {docker_exe}")
        else:
            self.dlg.append_log("[PLUGIN] Docker não localizado.")
            self.dlg.append_log("[PLUGIN] Diagnóstico de descoberta do Docker:")
            for line in get_docker_search_diagnostics().splitlines():
                self.dlg.append_log(f"[PLUGIN] {line}")

    def _open_output_folder(self):
        outdir = self.dlg.le_outdir.text().strip()
        if not outdir:
            QMessageBox.warning(self.dlg, "Saída", "Nenhum diretório de saída informado.")
            return

        p = Path(outdir)
        if not p.exists():
            QMessageBox.warning(self.dlg, "Saída", "O diretório de saída ainda não existe.")
            return

        try:
            os.startfile(str(p))
        except Exception:
            try:
                subprocess.Popen(["explorer", str(p)])
            except Exception as e:
                QMessageBox.warning(self.dlg, "Saída", f"Não foi possível abrir a pasta:\n{repr(e)}")

    def _build_current_command(self):
        use_cuda = detect_cuda_support()
        device = "cuda" if use_cuda else "cpu"

        preset = self._fixed_current_preset()
        cfg = DockerRunConfig(
            image=self.dlg.le_image.text().strip() or "cbers-colorize:gpu",
            pan=self.dlg.le_pan.text().strip(),
            blue=self.dlg.le_blue.text().strip(),
            green=self.dlg.le_green.text().strip(),
            red=self.dlg.le_red.text().strip(),
            outdir=self.dlg.le_outdir.text().strip(),
            device=device,
            export_cog=self.dlg.chk_export_cog.isChecked(),
            keep_tmp=self.dlg.chk_keep_tmp.isChecked(),
            verbose=self.dlg.chk_verbose.isChecked(),
            preset_name=preset["name"],
            preset_args=preset["args"],
        )
        return build_docker_command(cfg), device

    def _copy_last_command(self):
        if self._last_built_cmd:
            txt = " ".join(self._last_built_cmd)
            self.dlg.copy_text_to_clipboard(txt)
            QMessageBox.information(self.dlg, "Comando Docker", "Último comando copiado para a área de transferência.")
            return

        ok, msg = self._validate_inputs()
        if not ok:
            QMessageBox.warning(self.dlg, "Comando Docker", f"Não foi possível montar o comando:\n{msg}")
            return

        try:
            cmd, _device = self._build_current_command()
        except Exception as e:
            QMessageBox.warning(self.dlg, "Comando Docker", f"Não foi possível montar o comando:\n{repr(e)}")
            return

        self._last_built_cmd = cmd
        self.dlg.copy_text_to_clipboard(" ".join(cmd))
        QMessageBox.information(self.dlg, "Comando Docker", "Comando atual copiado para a área de transferência.")

    def _on_run_clicked(self):
        ok, msg = self._validate_inputs()
        if not ok:
            QMessageBox.warning(self.dlg, "Validação", msg)
            return

        docker_ok, docker_msg = detect_docker()
        if not docker_ok:
            QMessageBox.critical(self.dlg, "Docker", docker_msg)
            return

        self._save_settings_from_dialog()

        try:
            cmd, device = self._build_current_command()
        except Exception as e:
            QMessageBox.critical(self.dlg, "Execução", f"Falha ao montar comando Docker:\n{repr(e)}")
            return

        self._last_device_used = device
        self._last_built_cmd = cmd

        self.dlg.clear_log()
        self.dlg.clear_summary()
        self.dlg.append_log("[PLUGIN] Preset fixo em uso: Atual")
        self.dlg.append_log(f"Device selecionado: {device}")
        self.dlg.append_log("Comando:")
        self.dlg.append_log(" ".join(cmd))
        self.dlg.append_log("-" * 80)

        self.parser = LogSummaryParser()
        self.worker = DockerProcessWorker(self.parser, self.dlg)
        self.worker.log_line.connect(self.dlg.append_log)
        self.worker.elapsed_changed.connect(self.dlg.set_elapsed_seconds)
        self.worker.progress_changed.connect(self.dlg.set_progress)
        self.worker.started_ok.connect(self._on_worker_started)
        self.worker.start_failed.connect(self._on_worker_start_failed)
        self.worker.finished_ok.connect(self._on_worker_finished_ok)
        self.worker.finished_error.connect(self._on_worker_finished_error)

        self.dlg.set_running(True)
        self.worker.start(cmd)

    def _on_cancel_clicked(self):
        if self.worker and self.worker.is_running():
            self.worker.cancel()
        self.dlg.set_running(False)

    def _on_worker_started(self):
        self.dlg.append_log("[PLUGIN] Processo iniciado com sucesso.")

    def _on_worker_start_failed(self, msg: str):
        self.dlg.set_running(False)
        QMessageBox.critical(self.dlg, "Execução", msg)

    def _on_worker_finished_ok(self, summary: dict):
        self.dlg.set_running(False)

        outdir = Path(self.dlg.le_outdir.text().strip())
        master_path = outdir / "pan_2m_color_guided.tif"
        cog_path = outdir / "pan_2m_color_guided.cog.tif"

        summary["device_used"] = self._last_device_used or ""
        summary["master_path"] = str(master_path) if master_path.exists() else ""
        summary["cog_path"] = str(cog_path) if cog_path.exists() else ""

        self.dlg.set_summary(summary)
        self.dlg.set_progress(100)

        loaded = []
        load_mode = self.dlg.current_load_mode()

        if load_mode in (LOAD_MODE_MASTER, LOAD_MODE_BOTH) and master_path.exists():
            if self._add_raster_to_map(master_path, "CBERS Guided Master"):
                loaded.append(master_path.name)

        if load_mode in (LOAD_MODE_COG, LOAD_MODE_BOTH) and cog_path.exists():
            if self._add_raster_to_map(cog_path, "CBERS Guided COG"):
                loaded.append(cog_path.name)

        msg = "Processamento concluído."
        if loaded:
            msg += "\n\nCamadas adicionadas ao mapa:\n- " + "\n- ".join(loaded)

        QMessageBox.information(self.dlg, "CBERS Colorize", msg)

    def _on_worker_finished_error(self, exit_code: int, summary: dict):
        self.dlg.set_running(False)

        outdir = Path(self.dlg.le_outdir.text().strip())
        master_path = outdir / "pan_2m_color_guided.tif"
        cog_path = outdir / "pan_2m_color_guided.cog.tif"

        summary["device_used"] = self._last_device_used or ""
        summary["master_path"] = str(master_path) if master_path.exists() else ""
        summary["cog_path"] = str(cog_path) if cog_path.exists() else ""

        self.dlg.set_summary(summary)
        QMessageBox.warning(
            self.dlg,
            "CBERS Colorize",
            f"Processo finalizado com código {exit_code}."
        )

    def _add_raster_to_map(self, raster_path: Path, layer_name: str) -> bool:
        layer = QgsRasterLayer(str(raster_path), layer_name)
        if not layer.isValid():
            self.dlg.append_log(f"[PLUGIN] Falha ao carregar camada no mapa: {raster_path}")
            return False

        QgsProject.instance().addMapLayer(layer)
        self.dlg.append_log(f"[PLUGIN] Camada adicionada ao mapa: {raster_path.name}")
        return True