# CBERS Colorize Plugin for QGIS

Plugin QGIS para executar o pipeline **CBERS Colorization** via **Docker**, sem exigir que o usuário digite manualmente comandos longos no terminal.

## Estado atual

Versão documentada neste arquivo: **V1.5**

O plugin já oferece uma interface operacional utilizável para execução do processamento em Docker, com seleção explícita das bandas, acompanhamento de execução, resumo QA/TIME e integração com o mapa do QGIS.

---

## 1. Objetivo do plugin

O plugin foi criado para servir como **interface gráfica do pipeline CBERS Colorization** dentro do QGIS.

Em vez de o usuário precisar montar manualmente um comando `docker run` com dezenas de parâmetros, o plugin permite:

- selecionar as bandas CBERS de entrada;
- selecionar a pasta de saída;
- escolher um preset de processamento;
- executar o pipeline no Docker;
- acompanhar o processamento por log, cronômetro e progresso estimado;
- visualizar o resumo estatístico do processamento;
- carregar automaticamente o resultado no mapa do QGIS.

---

## 2. O que o plugin faz atualmente

Na V1.5, o plugin já permite:

- selecionar explicitamente:
  - `Band0 (PAN)`
  - `Band1 (Blue)`
  - `Band2 (Green)`
  - `Band3 (Red)`
- escolher o diretório de saída;
- autodetectar as bandas a partir de uma pasta;
- sugerir uma pasta de saída a partir da pasta das bandas;
- escolher se exporta COG;
- escolher se mantém temporários;
- escolher se roda em modo verbose;
- detectar Docker;
- detectar suporte a CUDA;
- detectar automaticamente uma imagem Docker do projeto;
- testar leitura dos arquivos de entrada;
- testar escrita na pasta de saída;
- montar o comando Docker automaticamente;
- copiar o comando Docker para a área de transferência;
- executar o processamento;
- mostrar:
  - log em tempo real;
  - cronômetro;
  - barra de progresso estimada;
  - percentual de progresso;
- exibir resumo final com:
  - `TIME BREAKDOWN`
  - `QA2`
  - `QA3`
  - device usado
  - caminho do master
  - caminho do COG
- abrir a pasta de saída;
- carregar automaticamente no mapa:
  - o master
  - o COG
  - ambos
  - ou nenhum
- salvar/restaurar configurações com `QSettings`.

---

## 3. Estrutura atual do plugin

```text
cbers_colorize_qgis/
├─ __init__.py
├─ metadata.txt
├─ icon.png
├─ resources.py
├─ resources.qrc
├─ cbers_colorize_plugin.py
├─ ui/
│  └─ cbers_colorize_dialog.py
├─ core/
│  ├─ docker_runner.py
│  ├─ log_parser.py
│  ├─ presets.py
│  └─ worker.py
└─ README_plugin.md

---

## 4. O que faz cada arquivo

__init__.py

    Arquivo de entrada do plugin para o QGIS.
    Responsabilidade:
    expor classFactory(...).

metadata.txt

    Metadados do plugin reconhecidos pelo QGIS.
    Responsabilidade:
    nome;
    descrição;
    versão;
    autor;
    categoria;
    ícone;
    compatibilidade.


icon.png

    Ícone do plugin no menu/barra de ferramentas do QGIS.

resources.qrc

    Arquivo de definição de recursos Qt.
    Atualmente pode conter apenas o icon.png.

resources.py

    Arquivo de recursos compilados do Qt.
    Na V1.5 ele ainda pode ficar como placeholder, porque o ícone está sendo usado por caminho direto.

cbers_colorize_plugin.py

    Arquivo principal do plugin.
    Responsabilidades:

        registrar ação no QGIS;
        abrir a janela do plugin;
        coordenar validação;
        verificar ambiente Docker/CUDA/imagem;
        montar o comando;
        iniciar o worker;
        tratar fim da execução;
        adicionar saídas ao mapa;
        salvar/restaurar preferências via QSettings.

ui/cbers_colorize_dialog.py

    Define a interface gráfica do plugin.
    Responsabilidades:

        campos de entrada e saída;
        autodetecção por pasta;
        seleção de preset;
        checkboxes de opções;
        painel de ambiente;
        cronômetro;
        barra de progresso;
        área de log;
        tabelas de resumo;
        ações auxiliares como abrir pasta e copiar texto.

core/presets.py

    Centraliza os presets de parâmetros do pipeline.
    Responsabilidades:

        armazenar presets:
        Atual
        Conservadora
        Viva
        manter a UI limpa;
        facilitar futuras expansões.

core/docker_runner.py

    Camada de integração com Docker.
    Responsabilidades:

        detectar Docker;
        detectar CUDA;
        detectar se a imagem Docker existe;
        sugerir imagem padrão;
        validar leitura/escrita de arquivos e pastas;
        montar o comando docker run.

core/worker.py

    Executa o processo de forma assíncrona.
    Responsabilidades:

        iniciar o processo Docker;
        capturar stdout/stderr;
        emitir sinais de log;
        emitir sinais de progresso;
        emitir cronômetro;
        permitir cancelamento;
        emitir resumo no final.

core/log_parser.py

    Interpreta o log do pipeline para montar o resumo exibido na interface.
    Responsabilidades:

        extrair tempos por etapa;
        extrair tempo total;
        extrair QA2 e QA3;
        separar métricas como:

            status
            n
            grad_mean
            lap_mean
            leak_mean
            skipped_low_valid
            skipped_small_mask
        estimar progresso com base em marcos do log.

## 5. Requisitos para uso

    O plugin não executa o processamento por conta própria. Ele apenas orquestra o Docker.
    O computador do usuário precisa ter:
        QGIS instalado;
        Docker instalado e funcionando;
        imagem Docker do projeto disponível localmente, por exemplo:
            cbers-colorize:gpu
    Se for usar GPU:
        driver NVIDIA instalado no host;
        NVIDIA Container Toolkit configurado;
        Docker com suporte a --gpus all.

## 6. Onde instalar o plugin no QGIS

    No Windows, normalmente a pasta do plugin fica em algo como:

        C:\Users\SEU_USUARIO\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\cbers_colorize_qgis\

    Depois:

        abrir o QGIS;
        ir em Plugins > Manage and Install Plugins;
        localizar o plugin entre os instalados;
        habilitá-lo.

## 7. Fluxo de uso

### Passo 1 — verificar ambiente

Na abertura, o plugin pode verificar:

- Docker
- CUDA
- imagem Docker
- I/O da pasta de saída

O usuário também pode clicar em:

**Verificar ambiente**

### Passo 2 — autodetectar ou selecionar bandas

O usuário pode:

- selecionar manualmente:
  - `Band0`
  - `Band1`
  - `Band2`
  - `Band3`

ou:

- escolher uma pasta e clicar em:

**Auto detectar BAND0/BAND1/BAND2/BAND3**

O plugin procura arquivos contendo:

- `BAND0`
- `BAND1`
- `BAND2`
- `BAND3`

no nome do arquivo.

### Passo 3 — escolher a pasta de saída

O usuário escolhe a pasta de saída manualmente ou aceita a sugestão automática:

`<PASTA_DAS_BANDAS>\output_colorized`

### Passo 4 — escolher preset e opções

O usuário define:

- preset:
  - **Atual**
  - **Conservadora**
  - **Viva**
- exportar COG
- manter temporários
- verbose
- o que carregar no mapa ao final

### Passo 5 — validar entradas

O botão:

**Validar entradas**

executa verificações de:

- existência dos arquivos
- leitura dos arquivos
- escrita da pasta de saída

### Passo 6 — executar

Ao clicar em:

**Executar**

o plugin:

- detecta CUDA
- escolhe `cuda` ou `cpu`
- monta o comando Docker
- guarda o comando
- inicia o worker
- atualiza:
  - log
  - cronômetro
  - barra de progresso
  - percentual

### Passo 7 — acompanhar execução

Durante o processamento, a interface mostra:

- tempo decorrido
- status
- barra de progresso estimada
- log em tempo real

### Passo 8 — conclusão

Ao final, o plugin exibe:

- `TIME BREAKDOWN`
- `QA2`
- `QA3`
- device usado
- caminho do master
- caminho do COG

Dependendo da opção escolhida, ele também pode carregar automaticamente no mapa:

- master
- COG
- ambos
- nenhum

## 8. Painel de ambiente

Na V1.5, o plugin possui um painel de ambiente com:

- Docker
- CUDA
- Imagem
- I/O

### Docker

Verifica se o executável `docker` está acessível.

### CUDA

Tenta executar um container de teste com `nvidia-smi`.

### Imagem

Verifica se a imagem definida no campo **Imagem Docker** existe localmente.

### I/O

Testa se a pasta de saída pode ser criada e escrita.

## 9. Presets

Os presets atuais são definidos em `core/presets.py`.

### Atual

Preset principal para uso geral.

### Conservadora

Variação ligeiramente mais conservadora nas cores.

### Viva

Variação um pouco mais viva, sem alterar radicalmente o comportamento geral.

> **Observação**
> Na prática, diferenças visuais entre os presets podem ser sutis dependendo da cena.

## 10. Progresso exibido ao usuário

O plugin exibe uma barra de progresso com percentual.

### Importante

Na V1.5, esse progresso é estimado por marcos do log, não por progresso interno exato do pipeline.

Exemplos de marcos usados:

- `STEP0 iniciado/concluído`
- `STEP1 concluído`
- `STEP1b concluído`
- `STEP2 concluído`
- `STEP7 concluído`
- `QA3 concluído`
- `exportação COG`
- `TOTAL`

Isso já fornece uma percepção operacional útil do andamento.

## 11. Resumo QA na V1.5

O parser tenta extrair do log:

### QA2

- `qa2_status`
- `qa2_n`
- `qa2_grad_mean`
- `qa2_lap_mean`
- `qa2_leak_mean`
- `qa2_skipped_low_valid`
- `qa2_skipped_small_mask`
- `qa2_text`

### QA3

- `qa3_status`
- `qa3_n`
- `qa3_grad_mean`
- `qa3_lap_mean`
- `qa3_leak_mean`
- `qa3_skipped_low_valid`
- `qa3_skipped_small_mask`
- `qa3_text`

Esses campos dependem do formato do log emitido pelo pipeline.