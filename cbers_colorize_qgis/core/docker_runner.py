# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import sys


@dataclass
class DockerRunConfig:
    image: str
    pan: str
    blue: str
    green: str
    red: str
    outdir: str
    device: str
    export_cog: bool
    keep_tmp: bool
    verbose: bool
    preset_name: str
    preset_args: list[str]


def find_docker_exe() -> str | None:
    """
    Localiza o executável do Docker de forma robusta, especialmente no Windows/QGIS,
    onde o PATH do processo pode não refletir o PATH do terminal do usuário.
    """
    docker = shutil.which("docker")
    if docker:
        return str(Path(docker).resolve())

    candidates: list[str] = []

    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

        candidates.extend([
            os.path.join(program_files, "Docker", "Docker", "resources", "bin", "docker.exe"),
            os.path.join(program_files, "Docker", "Docker", "resources", "bin", "docker"),
            os.path.join(program_files_x86, "Docker", "Docker", "resources", "bin", "docker.exe"),
            os.path.join(program_files_x86, "Docker", "Docker", "resources", "bin", "docker"),
        ])

        if local_appdata:
            candidates.extend([
                os.path.join(local_appdata, "Programs", "Docker", "Docker", "resources", "bin", "docker.exe"),
                os.path.join(local_appdata, "Programs", "Docker", "Docker", "resources", "bin", "docker"),
            ])

    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p.resolve())

    return None


def get_docker_search_diagnostics() -> str:
    which_docker = shutil.which("docker")
    lines = [
        f"sys.platform={sys.platform}",
        f"PATH={os.environ.get('PATH', '')}",
        f"shutil.which('docker')={which_docker!r}",
    ]

    found = find_docker_exe()
    lines.append(f"find_docker_exe()={found!r}")
    return "\n".join(lines)


def _run_subprocess(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def detect_docker() -> tuple[bool, str]:
    docker = find_docker_exe()
    if not docker:
        return False, "Docker não encontrado no PATH do QGIS nem em caminhos padrão do Windows."

    try:
        cp = _run_subprocess([docker, "--version"], timeout=10)
        if cp.returncode != 0:
            stderr = (cp.stderr or "").strip()
            stdout = (cp.stdout or "").strip()
            detail = stderr or stdout or "sem detalhes"
            return False, f"Falha ao executar docker --version via:\n{docker}\n\n{detail}"
    except Exception as e:
        return False, f"Erro ao testar Docker via:\n{docker}\n\n{repr(e)}"

    version = (cp.stdout or "").strip() or "Docker encontrado."
    return True, f"{version} | executável: {docker}"


def detect_cuda_support() -> bool:
    docker = find_docker_exe()
    if not docker:
        return False

    test_cmd = [
        docker,
        "run",
        "--rm",
        "--gpus",
        "all",
        "nvidia/cuda:12.3.2-base-ubuntu22.04",
        "nvidia-smi",
    ]

    try:
        cp = _run_subprocess(test_cmd, timeout=40)
        return cp.returncode == 0
    except Exception:
        return False


def detect_image_exists(image_name: str) -> tuple[bool, str]:
    docker = find_docker_exe()
    if not docker:
        return False, "Docker não encontrado."

    try:
        cp = _run_subprocess([docker, "image", "inspect", image_name], timeout=15)
        if cp.returncode == 0:
            return True, f"Imagem encontrada: {image_name}"
        return False, f"Imagem não encontrada localmente: {image_name}"
    except Exception as e:
        return False, f"Erro ao verificar imagem Docker via:\n{docker}\n\n{repr(e)}"


def detect_preferred_image(candidates: list[str] | None = None) -> tuple[str | None, str]:
    if candidates is None:
        candidates = [
            "cbers-colorize:gpu",
            "cbers-colorize:latest",
            "cbers-colorize",
            "cbers-colorize:cpu",
        ]

    for image in candidates:
        ok, _msg = detect_image_exists(image)
        if ok:
            return image, f"Imagem sugerida: {image}"

    return None, "Nenhuma imagem padrão do projeto foi encontrada localmente."


def test_output_dir_writable(outdir: str) -> tuple[bool, str]:
    try:
        p = Path(outdir)
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".cbers_colorize_write_test.tmp"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok\n")
        test_file.unlink(missing_ok=True)
        return True, f"Diretório gravável: {p}"
    except Exception as e:
        return False, f"Falha de escrita no diretório de saída:\n{repr(e)}"


def test_input_files_readable(paths: list[str]) -> tuple[bool, str]:
    missing = []
    unreadable = []

    for p in paths:
        pp = Path(p)
        if not pp.exists():
            missing.append(str(pp))
            continue
        try:
            with open(pp, "rb") as f:
                f.read(16)
        except Exception:
            unreadable.append(str(pp))

    if missing:
        return False, "Arquivos inexistentes:\n- " + "\n- ".join(missing)
    if unreadable:
        return False, "Arquivos sem leitura:\n- " + "\n- ".join(unreadable)
    return True, "Arquivos de entrada legíveis."


def _common_input_dir(paths: list[str]) -> Path:
    resolved = [str(Path(p).resolve()) for p in paths]
    return Path(os.path.commonpath(resolved))


def _container_rel_path(file_path: str, mounted_root: Path) -> str:
    p = Path(file_path).resolve()
    rel = p.relative_to(mounted_root)
    return "/data/" + rel.as_posix()


def build_docker_command(cfg: DockerRunConfig) -> list[str]:
    docker = find_docker_exe()
    if not docker:
        raise RuntimeError(
            "Docker não encontrado no PATH do QGIS nem em caminhos padrão do Windows."
        )

    input_root = _common_input_dir([cfg.pan, cfg.blue, cfg.green, cfg.red])
    outdir = Path(cfg.outdir).resolve()

    pan_in = _container_rel_path(cfg.pan, input_root)
    blue_in = _container_rel_path(cfg.blue, input_root)
    green_in = _container_rel_path(cfg.green, input_root)
    red_in = _container_rel_path(cfg.red, input_root)

    cmd = [docker, "run", "--rm", "-i"]

    if cfg.device == "cuda":
        cmd += ["--gpus", "all"]
        cmd += ["-e", "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8"]

    cmd += ["-v", f"{str(input_root)}:/data:ro"]
    cmd += ["-v", f"{str(outdir)}:/out"]
    cmd += ["-v", "cbers_work:/work"]

    cmd += [cfg.image, "run"]
    cmd += ["--pan", pan_in]
    cmd += ["--blue", blue_in]
    cmd += ["--green", green_in]
    cmd += ["--red", red_in]
    cmd += ["--outdir", "/out"]
    cmd += ["--device", cfg.device]

    cmd += cfg.preset_args

    if cfg.export_cog:
        cmd += ["--export_cog", "--cog_overviews"]

    if cfg.keep_tmp:
        cmd += ["--keep_tmp"]

    if cfg.verbose:
        cmd += ["--verbose"]

    return cmd