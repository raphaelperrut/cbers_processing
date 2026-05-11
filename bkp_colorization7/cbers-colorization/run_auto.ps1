param(
    [string]$InputDir = "C:\Users\DGEO2CGEO\Documents\Colorizacao\cbers\teste\input",
    [string]$OutputDir = "C:\Users\DGEO2CGEO\Documents\Colorizacao\cbers\teste\output",

    [string]$PanFile = "CBERS_4A_WPM_20250807_208_134_L4_BAND0.tif",
    [string]$BlueFile = "CBERS_4A_WPM_20250807_208_134_L4_BAND1.tif",
    [string]$GreenFile = "CBERS_4A_WPM_20250807_208_134_L4_BAND2.tif",
    [string]$RedFile = "CBERS_4A_WPM_20250807_208_134_L4_BAND3.tif",

    [string]$ContainerInputDir = "/data",
    [string]$ContainerOutputDir = "/out",

    [string]$DockerImage = "cbers-colorize:gpu",
    [string]$DockerVolumeWork = "cbers_work",

    [bool]$ExportCOG = $true,
    [bool]$KeepTmp = $false,
    [bool]$Verbose = $false
)

$ErrorActionPreference = "Stop"

# =========================
# Helpers
# =========================

function Test-Docker {
    try {
        docker version | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Test-NvidiaSmiHost {
    try {
        $null = Get-Command nvidia-smi -ErrorAction Stop
        nvidia-smi | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Test-DockerGpu {
    try {
        docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Assert-PathExists([string]$PathValue, [string]$Label) {
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label não encontrado: $PathValue"
    }
}

function Join-ContainerPath([string]$Dir, [string]$Leaf) {
    $d = $Dir.TrimEnd("/")
    return "$d/$Leaf"
}

# =========================
# Validações
# =========================

if (-not (Test-Docker)) {
    throw "Docker não está disponível neste host."
}

Assert-PathExists -PathValue $InputDir -Label "Diretório de entrada"

if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$panHost   = Join-Path $InputDir $PanFile
$blueHost  = Join-Path $InputDir $BlueFile
$greenHost = Join-Path $InputDir $GreenFile
$redHost   = Join-Path $InputDir $RedFile

Assert-PathExists -PathValue $panHost   -Label "Band0/PAN"
Assert-PathExists -PathValue $blueHost  -Label "Band1/BLUE"
Assert-PathExists -PathValue $greenHost -Label "Band2/GREEN"
Assert-PathExists -PathValue $redHost   -Label "Band3/RED"

$panInContainer   = Join-ContainerPath $ContainerInputDir $PanFile
$blueInContainer  = Join-ContainerPath $ContainerInputDir $BlueFile
$greenInContainer = Join-ContainerPath $ContainerInputDir $GreenFile
$redInContainer   = Join-ContainerPath $ContainerInputDir $RedFile

# =========================
# Detecta GPU/CPU
# =========================

$UseGpu = $false

if ((Test-NvidiaSmiHost) -and (Test-DockerGpu)) {
    $UseGpu = $true
}

# =========================
# Garante volume de trabalho
# =========================

docker volume create $DockerVolumeWork | Out-Null

# =========================
# Preset de produção
# =========================

$PipelineArgs = @(
    "run",
    "--pan", $panInContainer,
    "--blue", $blueInContainer,
    "--green", $greenInContainer,
    "--red", $redInContainer,
    "--outdir", $ContainerOutputDir,

    "--tile", "512",
    "--overlap", "32",
    "--io_block", "1024",
    "--out_block", "1024",
    "--compress", "ZSTD",
    "--guide_norm", "per_band",

    "--chroma_strength", "0.62",
    "--sat", "0.84",
    "--cr_bias", "-0.010",
    "--cb_bias", "-0.004",

    "--veg_exg_th", "0.10",
    "--veg_sat", "0.78",
    "--veg_chroma", "0.86",
    "--veg_cr_bias", "-0.014",
    "--veg_cb_bias", "0.004",

    "--neutral_mag_thr", "0.060",
    "--neutral_strength", "0.85",

    "--shadow_y_lo", "0.08",
    "--shadow_y_hi", "0.28",
    "--shadow_strength", "0.28",
    "--shadow_cb_bias", "-0.006",
    "--shadow_cr_bias", "0.004",
    "--shadow_chroma", "1.0",

    "--urban_neutral_thr", "0.12",
    "--urban_y_lo", "0.18",
    "--urban_y_hi", "0.90",
    "--urban_blue_reduction", "0.08",
    "--urban_warmth", "0.03",

    "--hi_y", "0.82",
    "--hi_desat", "0.78",
    "--gamut_gain", "6.0",

    "--luma_rolloff_knee", "0.80",
    "--luma_rolloff_strength", "0.55",
    "--luma_gamma", "0.96"
)

if ($ExportCOG) {
    $PipelineArgs += @("--export_cog", "--cog_overviews")
}

if ($KeepTmp) {
    $PipelineArgs += @("--keep_tmp")
}

if ($Verbose) {
    $PipelineArgs += @("--verbose")
}

if ($UseGpu) {
    $Device = "cuda"
    Write-Host "[AUTO] GPU NVIDIA detectada. Processamento com CUDA." -ForegroundColor Green
}
else {
    $Device = "cpu"
    Write-Host "[AUTO] GPU NVIDIA indisponível. Processamento em CPU." -ForegroundColor Yellow
}

$PipelineArgs += @("--device", $Device)

# =========================
# Monta docker run
# =========================

$DockerArgs = @(
    "run",
    "--rm",
    "-it"
)

if ($UseGpu) {
    $DockerArgs += @("--gpus", "all")
}

$DockerArgs += @(
    "-e", "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8",
    "-v", "${InputDir}:${ContainerInputDir}:ro",
    "-v", "${OutputDir}:${ContainerOutputDir}",
    "-v", "${DockerVolumeWork}:/work",
    $DockerImage
)

$DockerArgs += $PipelineArgs

# =========================
# Log amigável
# =========================

Write-Host ""
Write-Host "========== CBERS Colorization ==========" -ForegroundColor Cyan
Write-Host "InputDir : $InputDir"
Write-Host "OutputDir: $OutputDir"
Write-Host "Band0    : $PanFile"
Write-Host "Band1    : $BlueFile"
Write-Host "Band2    : $GreenFile"
Write-Host "Band3    : $RedFile"
Write-Host "Imagem   : $DockerImage"
Write-Host "Modo     : $Device"
Write-Host "COG      : $ExportCOG"
Write-Host "KeepTmp  : $KeepTmp"
Write-Host "Verbose  : $Verbose"
Write-Host "======================================="
Write-Host ""

# =========================
# Executa
# =========================

& docker @DockerArgs
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "O processamento falhou com código de saída $exitCode."
}

Write-Host ""
Write-Host "[AUTO] Processamento concluído com sucesso." -ForegroundColor Green
Write-Host "Master esperado: $(Join-Path $OutputDir 'pan_2m_color_guided.tif')"
if ($ExportCOG) {
    Write-Host "COG esperado   : $(Join-Path $OutputDir 'pan_2m_color_guided.cog.tif')"
}