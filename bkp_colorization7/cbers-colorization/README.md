# CBERS Colorization

Pipeline para geração de imagem colorida em **2 m** a partir de uma cena **CBERS-4A WPM** com as quatro entradas separadas:

- **BAND0**: PAN (2 m)
- **BAND1**: Blue (8 m)
- **BAND2**: Green (8 m)
- **BAND3**: Red (8 m)

O projeto foi organizado para rodar em **Docker** com CPU ou GPU, mantendo a pancromática como referência espacial e usando as bandas multiespectrais como **guia cromático**. Na configuração final atual, o fluxo principal é um **color guided fusion** com injeção de luminância PAN e crominância derivada do guia RGB reamostrado e alinhado.

## 1. Objetivo

O objetivo do projeto é produzir um raster RGB de alta resolução espacial que:

- preserve o detalhe da **PAN**;
- transfira cor de forma estável a partir do RGB 8 m;
- reduza vazamento cromático em bordas;
- reduza excesso de azul em áreas urbanas e excesso de vermelho em solo exposto;
- gere um **GeoTIFF master** e, opcionalmente, derivados leves como **COG** e **ECW**.

## 2. Estrutura do projeto

```text
cbers-colorization/
├─ run_auto.ps1
├─ cbers_colorize/
│  ├─ __init__.py
│  ├─ cli.py
│  ├─ pipeline.py
│  ├─ progress.py
│  ├─ ops_color.py
│  ├─ ops_gdal.py
│  ├─ ops_sr.py
│  ├─ rsinet/
│  │  ├─ __init__.py
│  │  ├─ net_common.py
│  │  └─ net.py
│  └─ tools/
│     ├─ __init__.py
│     ├─ infer_geotiff_color.py
│     └─ infer_geotiff_sr.py
├─ docker/
│  ├─ Dockerfile
│  └─ entrypoint.sh
├─ models/
│  └─ color.pkl
├─ compose.gpu.yaml
├─ compose.yaml
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

## 3. Visão geral do fluxo

### 3.1 Entradas

O comando principal recebe quatro rasters separados:

- `--pan`: banda pancromática (BAND0)
- `--blue`: banda azul (BAND1)
- `--green`: banda verde (BAND2)
- `--red`: banda vermelha (BAND3)

Embora o sensor seja fornecido como B, G, R, o pipeline converte tudo internamente para a convenção **RGB**, isto é:

- banda interna 1 = **R**
- banda interna 2 = **G**
- banda interna 3 = **B**

### 3.2 Etapas

#### Etapa 0 — PAN para RGB 3 bandas `float01`

A PAN é normalizada por percentis (`pan_p_lo`, `pan_p_hi`) e replicada em 3 bandas. Isso cria uma base geométrica com a mesma resolução final e pronta para a etapa de fusão.

Saída temporária típica:

- `_work/pan_3band_f01.tif`

#### Etapa 1 — Construção do VRT RGB LR alinhado ao PAN

As bandas Blue, Green e Red são alinhadas ao grid do PAN, porém na resolução LR correspondente (`PAN / scale`, atualmente `scale=4`). Depois, é montado um VRT na ordem `[RED, GREEN, BLUE]`.

Saída temporária típica:

- `_work/rgb_lr_8m_aligned.vrt`

#### Etapa 1b — Normalização do guia LR

O VRT LR é convertido para GeoTIFF temporário e normalizado para `0..1`.

Modos suportados:

- `per_band`: percentis independentes por banda;
- `joint_y`: percentis conjuntos usando a luminância Y.

Preset recomendado:

- `--guide_norm per_band`

#### Etapa 2 — Reamostragem do guia RGB para 2 m

O RGB LR é reamostrado para a grade da PAN com `gdalwarp` e interpolação cúbica. Em seguida, é novamente normalizado para `0..1`.

Saída temporária típica:

- `_work/rgb_sensor_hr_2m_01.tif`

Esse raster é o **guia cromático final**.

#### Etapa 3 — Fusão PAN + guia RGB

O script `infer_geotiff_color.py` é chamado com:

- entrada PAN 3 bandas já em `float01`;
- guia RGB 2 m alinhado;
- parâmetros de ajuste cromático.

O fluxo final aprovado usa principalmente:

- `guide_mode = pan_luminance_injection`
- `fusion_mode = ycbcr`
- `guide_only = True`

Na prática:

- a **luminância** final vem da PAN;
- a **crominância** vem do guia RGB;
- ajustes locais refinam vegetação, zonas urbanas, sombras, highlights e gamut.

Saída master:

- `pan_2m_color_guided.tif`

#### Etapa 4 — QA espacial

Depois da fusão, o pipeline executa verificações automáticas comparando gradiente e laplaciano da saída com a PAN, além de medir vazamento de textura cromática.

Os logs aparecem como:

- `QA2`: no script de colorização
- `QA3`: no pipeline, contra a PAN bruta

#### Etapa 5 — Derivados opcionais

O pipeline pode gerar:

- **VIS**: GeoTIFF leve com stretch visual
- **COG**: Cloud Optimized GeoTIFF em JPEG
- **ECW**: opcional, se o driver ECW estiver disponível no GDAL

Recomendação operacional atual:

- usar `pan_2m_color_guided.tif` como **master**;
- usar `pan_2m_color_guided.cog.tif` como derivado leve;
- desligar `VIS` quando não houver necessidade;
- manter `ECW` apenas como opcional.

#### Etapa 6 — Limpeza automática

Se `--keep_tmp` não for usado, o pipeline remove:

- VRTs temporários `_tmp_*`
- TIFFs temporários intermediários
- o VRT LR final
- a pasta `_work`

## 4. Produtos gerados

### 4.1 Master técnico

- `pan_2m_color_guided.tif`

Características:

- GeoTIFF em `float32`
- 3 bandas RGB
- alinhado à PAN
- preserva georreferenciamento
- produto principal para análise, arquivamento e pós-processamento

### 4.2 COG leve opcional

- `pan_2m_color_guided.cog.tif`

Características:

- COG com compressão JPEG
- mais leve para distribuição e visualização
- não substitui o master técnico

### 4.3 VIS opcional

- `pan_2m_color_guided_vis.tif`

Características:

- GeoTIFF com stretch visual e gamma
- útil para abrir diretamente no QGIS com boa visualização inicial
- opcional em produção

### 4.4 ECW opcional

- `pan_2m_color_guided.ecw`

Observação:

- só é exportado se o **driver ECW** estiver disponível no GDAL do ambiente.

## 5. Execução em Docker

O projeto pode ser executado de duas formas principais:

1. com `docker run` manual, quando se deseja controle total dos parâmetros;
2. com `run_auto.ps1`, quando se deseja uma forma mais prática de operação em Windows, com preferência por CUDA e fallback para CPU.

### 5.1 Pré-requisitos

- Docker instalado e funcional
- imagem Docker do projeto já construída, por exemplo:
  - `cbers-colorize:gpu`
- arquivos de entrada disponíveis em uma pasta local
- pasta local de saída com espaço suficiente

Observação prática:

- para uma cena completa do CBERS, reserve **pelo menos 100 GB livres**;
- após o processamento, os temporários podem ser removidos automaticamente se `--keep_tmp` não for usado.

### 5.2 Execução manual com `docker run`

Exemplo de uso em PowerShell com GPU:

```powershell

docker build -f docker\Dockerfile --target cpu -t cbers-colorize:cpu .     

docker build -f docker\Dockerfile --target gpu -t cbers-colorize:gpu .       

docker volume create cbers_work | Out-Null

docker run --rm -it --gpus all -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8" -v "C:\Users\DGEO2CGEO\Documents\Colorizacao\cbers\teste\input:/data:ro" -v "C:\Users\DGEO2CGEO\Documents\Colorizacao\cbers\teste\output:/out" -v cbers_work:/work cbers-colorize:gpu run --pan "/data/CBERS_4A_WPM_20250807_208_134_L4_BAND0.tif" --blue "/data/CBERS_4A_WPM_20250807_208_134_L4_BAND1.tif" --green "/data/CBERS_4A_WPM_20250807_208_134_L4_BAND2.tif" --red "/data/CBERS_4A_WPM_20250807_208_134_L4_BAND3.tif" --outdir "/out" --device cuda --tile 512 --overlap 32 --io_block 1024 --out_block 1024 --compress ZSTD --guide_norm per_band --chroma_strength 0.62 --sat 0.84 --cr_bias -0.010 --cb_bias -0.004 --veg_exg_th 0.10 --veg_sat 0.78 --veg_chroma 0.86 --veg_cr_bias -0.014 --veg_cb_bias 0.004 --neutral_mag_thr 0.060 --neutral_strength 0.85 --shadow_y_lo 0.08 --shadow_y_hi 0.28 --shadow_strength 0.28 --shadow_cb_bias -0.006 --shadow_cr_bias 0.004 --shadow_chroma 1.0 --urban_neutral_thr 0.12 --urban_y_lo 0.18 --urban_y_hi 0.90 --urban_blue_reduction 0.08 --urban_warmth 0.03 --hi_y 0.82 --hi_desat 0.78 --gamut_gain 6.0 --luma_rolloff_knee 0.80 --luma_rolloff_strength 0.55 --luma_gamma 0.96 --export_cog --cog_overviews

### 5.3 Execução simplificada com run_auto.ps1

O arquivo run_auto.ps1 deve ficar na raiz do projeto, ao lado de arquivos como:

  README.md
  pyproject.toml
  requirements.txt
  pasta cbers_colorize
  pasta docker

Ele serve como um launcher PowerShell para:

  detectar se o host possui suporte a CUDA;
  preferir GPU quando disponível;
  fazer fallback automático para CPU quando CUDA não estiver disponível;
  montar o comando Docker com menos risco de erro operacional;
  permitir ao usuário informar dinamicamente as bandas e as pastas de entrada e saída.

### 5.4 Como rodar o run_auto.ps1

No PowerShell, dentro da pasta do projeto:


  .\run_auto.ps1

Se o PowerShell bloquear scripts:

  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\run_auto.ps1

### 5.5 Exemplo de uso dinâmico do run_auto.ps1

Cena completa:

  .\run_auto.ps1 `
    -InputDir "D:\CBERS\entrada" `
    -OutputDir "D:\CBERS\saida" `
    -PanFile "CBERS_4A_WPM_20250807_208_134_L4_BAND0.tif" `
    -BlueFile "CBERS_4A_WPM_20250807_208_134_L4_BAND1.tif" `
    -GreenFile "CBERS_4A_WPM_20250807_208_134_L4_BAND2.tif" `
    -RedFile "CBERS_4A_WPM_20250807_208_134_L4_BAND3.tif"

Cena de teste recortada:

  .\run_auto.ps1 `
    -InputDir "C:\Users\SEU_USUARIO\Documents\Colorizacao\cbers\teste\input" `
    -OutputDir "C:\Users\SEU_USUARIO\Documents\Colorizacao\cbers\teste\output" `
    -PanFile "cbers_band0_cut.tif" `
    -BlueFile "cbers_band1_cut.tif" `
    -GreenFile "cbers_band2_cut.tif" `
    -RedFile "cbers_band3_cut.tif"

### 5.6 O que o usuário precisa ajustar

Ao usar run_auto.ps1, o usuário normalmente só precisa ajustar:

  InputDir
  OutputDir
  PanFile
  BlueFile
  GreenFile
  RedFile

Ou seja, os nomes das bandas e os diretórios não precisam ficar fixos no script.

### 5.7 Quando usar cada abordagem

Use run_auto.ps1 quando:

  o objetivo for operação rotineira;
  o usuário não quiser montar manualmente um docker run grande;
  a execução precisar decidir automaticamente entre GPU e CPU.

Use docker run manual quando:

  você estiver depurando;
  quiser testar parâmetros específicos;
  precisar documentar exatamente a linha de comando executada.

### 5.8 Saídas recomendadas

Na prática operacional, a recomendação é:

  master técnico: pan_2m_color_guided.tif
  derivado leve: pan_2m_color_guided.cog.tif

O VIS pode ser desligado quando não houver necessidade, e o ECW permanece opcional, dependendo da disponibilidade do driver GDAL correspondente.


## 6. O que faz cada arquivo Python

### `cbers_colorize/cli.py`

É a entrada de linha de comando.

Responsabilidades:

- define a CLI;
- cria o subcomando `run`;
- resolve `device=cpu/cuda/auto`;
- resolve AMP;
- valida ranges de parâmetros;
- chama `run_pipeline(...)`.

### `cbers_colorize/pipeline.py`

É o orquestrador principal.

Responsabilidades:

- define `PipelineConfig`;
- executa o fluxo ponta a ponta;
- cria PAN 3 bandas;
- monta VRT RGB LR alinhado;
- normaliza guia;
- reamostra guia para 2 m;
- chama `infer_geotiff_color.py`;
- roda QA espacial;
- exporta VIS/COG/ECW opcionais;
- limpa `_work` automaticamente.

### `cbers_colorize/progress.py`

Camada simples de progresso para CLI e futura integração com plugin QGIS.

Responsabilidades:

- emitir progresso em `0..1`;
- imprimir em stderr;
- aceitar callback externo.

### `cbers_colorize/ops_gdal.py`

Utilitários de alinhamento espacial e subprocessos GDAL.

Responsabilidades:

- configurar ambiente GDAL;
- chamar `gdalwarp` e `gdalbuildvrt`;
- construir o VRT RGB LR alinhado ao PAN;
- limpar temporários VRT;
- reamostrar para um grid de referência.

### `cbers_colorize/ops_color.py`

Utilitários auxiliares para a etapa de cor.

Responsabilidades:

- garantir `PYTHONPATH` correto para scripts em `tools/`;
- gerar um guia `uint8` com stretch por percentis;
- montar chamadas para o tool de colorização.

### `cbers_colorize/ops_sr.py`

Módulo de apoio para super-resolução RGB.

Responsabilidades:

- validar coerência dimensional LR/PAN;
- chamar `infer_geotiff_sr.py`;
- expor gancho de progresso para futura interface QGIS.

Observação: o fluxo principal atual está concentrado na fusão guiada por cor, então o módulo de SR está mais como infraestrutura e evolução futura.

### `cbers_colorize/rsinet/net_common.py`

Implementa blocos básicos reutilizáveis da rede, substituindo parcialmente dependências externas como `mmcv`.

Responsabilidades:

- `ConvModule`
- `Mish`
- `Default_Conv`
- `ChannelAttention`
- `SpatialAttention`
- `ConvUpsampler`
- `involution`

### `cbers_colorize/rsinet/net.py`

Implementa a arquitetura do modelo neural.

Responsabilidades:

- definir blocos (`MRB`, `IMUB_Head`, `IDB`, `IMUB`, `DAB`);
- montar a `UNet_Plus_Residual`;
- fornecer a fábrica `Kong(...)` para carregamento dos checkpoints.

### `cbers_colorize/tools/infer_geotiff_sr.py`

Script executável de super-resolução.

Responsabilidades:

- ler RGB LR;
- carregar checkpoint SR;
- dividir em tiles;
- inferir com overlap e feathering;
- gravar acumuladores e saída final em GeoTIFF;
- lidar com normalização, pós-processamento e fallback de memória.

### `cbers_colorize/tools/infer_geotiff_color.py`

Script executável de colorização/fusão cromática.

Responsabilidades:

- carregar checkpoint de cor;
- executar inferência ou modo `guide_only`;
- usar `fusion_mode=ycbcr` ou `ratio`;
- ajustar vegetação, neutros, sombras, urbano, highlights e gamut;
- gravar estatísticas embutidas no GeoTIFF;
- executar QA2 ao final.

## 7. Dependências e revisão do `requirements.txt`

### Dependências diretas realmente usadas pelos arquivos Python

As dependências Python diretas observadas no código são:

- `numpy`
- `rasterio`
- `torch`

Além disso, o ambiente precisa de utilitários GDAL instalados no sistema:

- `gdalwarp`
- `gdalbuildvrt`
- `gdal_translate`
- `gdaladdo`
- `gdalinfo`

### Situação do `requirements.txt`

O `requirements.txt` atual contém:

- `numpy`
- `rasterio`
- `tqdm`
- `packaging`
- `typing_extensions`

Para os arquivos Python enviados, `tqdm`, `packaging` e `typing_extensions` **não aparecem como dependências diretas atualmente**. Já `torch` **é usado diretamente** por `cli.py`, `net_common.py`, `net.py`, `infer_geotiff_sr.py` e `infer_geotiff_color.py`.

### Recomendação prática

Se o `Dockerfile` já instala `torch` separadamente conforme CPU/GPU, o `requirements.txt` atual pode continuar funcionando dentro do container.

Se você quiser que `pip install -r requirements.txt` também monte um ambiente Python funcional fora do Docker, a recomendação é ajustar para algo assim:

```txt
numpy==1.26.4
rasterio==1.3.10
# torch deve ser instalado conforme o ambiente (CPU ou CUDA)
# Exemplo CPU:
# torch==2.2.2
```

Ou, se preferir manter o arquivo mínimo e alinhado ao Docker:

```txt
numpy==1.26.4
rasterio==1.3.10
```

## 8. `pyproject.toml`

O `pyproject.toml` define:

- nome do pacote: `cbers-colorization`
- Python mínimo: `>=3.11`
- dependências base: `numpy`, `rasterio`
- extra opcional `cpu` com `torch>=2.1`
- script de console: `cbers-colorize = cbers_colorize.cli:main`

Isso está coerente com a estratégia de instalar `torch` de forma diferente em CPU e CUDA.

## 9. Próximos passos naturais

A estrutura atual já está pronta para a próxima fase:

1. consolidar o pacote Docker final;
2. testar a cena completa do CBERS;
3. criar um **plugin QGIS** com interface simples para:
   - escolher `band0`, `band1`, `band2`, `band3`;
   - escolher a pasta de saída;
   - disparar o processamento;
   - exibir logs/progresso.

## 10. Resumo operacional

- **Produto master:** `pan_2m_color_guided.tif`
- **Derivado leve recomendado:** `pan_2m_color_guided.cog.tif`
- **VIS:** opcional
- **ECW:** opcional e dependente do driver GDAL
- **Temporários em `_work`:** removidos automaticamente, salvo `--keep_tmp`
- **Preset final aprovado:** `guide_norm per_band` com o conjunto de parâmetros cromáticos incorporado na configuração atual
- **Espaço mínimo em disco** pelo menos 100 GB para processar 1 cena completa do CBERS (após finalizado processamento, cerca de 52 GB de arquivos temporários são descartados)
