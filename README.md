# LittleBit on a Steam Deck memory budget

**Target: Llama-2-7B. Executed CPU experiment: TinyStories-260K.** The 7B runner uses SamsungLabs' implementation. The CPU experiment is an independent, scaled-down reproduction with working packed checkpoints. Only the latter has been run here.

## Локальный запуск Qwen3-8B на Steam Deck

Цель — измерить пиковую RAM, время до первого токена и скорость генерации готовой модели **Qwen3-8B LittleBit-2 (0,55 BPW)**. Тест работает на CPU в Desktop Mode. Обучение, CUDA, ROCm и SSH не нужны. Веса занимают около 3 ГБ на диске. Для загрузки, Python и зависимостей желательно иметь 8 ГБ свободного места. Фактический пик RAM покажет тест.

**1. Установка.** Переключись в Desktop Mode, открой Konsole и выполни отдельной командой `bash`. Следующие блоки рассчитаны на Bash. Команды ставят Python и пакеты в домашний каталог без sudo. Если репозиторий уже клонирован, вместо первых двух строк перейди в его каталог и выполни `git pull --ff-only`.

```bash
git clone https://github.com/ILayZ/littlebit-steamdeck.git
cd littlebit-steamdeck
df -h .
free -h
command -v uv >/dev/null || curl --proto '=https' -fsSL https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 .venv-deck
source .venv-deck/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'
uv pip install 'transformers==4.51.3' 'accelerate==1.6.0' 'numpy==2.2.4' 'huggingface-hub==0.30.2' 'safetensors==0.5.3'
```

**2. Скачай модель один раз.** Команда загрузит закреплённую публичную версию в кеш Hugging Face. Аккаунт не требуется. После обрыва сети можно повторить команду.

```bash
python deck_qwen.py --download-only
```

**3. Запусти короткий тест.** Закрой игру перед запуском и запусти блок в той же Konsole. В новой сессии сначала выполни `bash`, перейди в каталог репозитория и активируй `.venv-deck`, как выше.

```bash
set -o pipefail
python -u deck_qwen.py --offline --threads 4 --tokens 16 2>&1 | tee deck-generation.log
```

Тест задаёт вопрос о столице Испании и печатает ответ модели. В конце лога будут `load_s` (загрузка модели), `first_token_s` (обработка запроса до первого токена), `decode_tokens_s` (последующие токены в секунду) и `peak_rss_mib` (пиковая RAM всего процесса, включая загрузку). Если модель сразу выдаст конец ответа, скорость последующих токенов будет `null`. Чтобы сравнить 4 и 8 потоков, повтори команду с `--threads 8` и другим именем лога. Для сравнения используй одинаковый лимит мощности и один режим питания.

**4. При желании проверь качество на WikiText-2.** Для первого запуска нужен интернет, чтобы скачать тестовую выборку. Сначала оцени длительность одного блока из 2048 токенов. Параметр `--offline` относится к весам модели.

```bash
uv pip install 'datasets==3.5.0'
python -u deck_qwen.py --offline --ppl-blocks 1 2>&1 | tee deck-ppl-one.log
```

Для сопоставления с опубликованным результатом запусти полный тест. Короткая проверка одного блока даёт другую выборку. Издатель чекпойнта указывает **24,48** на полном WikiText-2. Наш CPU-загрузчик и вычисления BF16 могут дать другое значение.

```bash
python -u deck_qwen.py --offline --ppl-blocks 0 2>&1 | tee deck-ppl-full.log
```

**Если тест прервётся.** При ошибке загрузки проверь свободное место через `df -h ~/.cache`. Если процесс завершится сообщением `Killed`, посмотри сообщения ядра через `journalctl -k -b | grep -Ei 'out of memory|oom|killed process' | tail`. Для разбора достаточно файла `deck-generation.log` либо последних строк ошибки, вывода `free -h` и, если запускал оценку, лога PPL.

Загрузчик в `deck_qwen.py` создаёт каркас Qwen3 без выделения памяти под исходные 8 млрд плотных весов. Знаки факторов остаются упакованными между вызовами слоёв и распаковываются для текущего матричного умножения. Это переносимая реализация для замера на CPU. Проверены разбор битов, обработка дополнения строк и интерфейс командной строки. Полная модель на физическом Steam Deck ещё не проверена.

[Карточка и метрики модели](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw) · [Формат бинарных факторов SamsungLabs](https://github.com/SamsungLabs/LittleBit/blob/0ceadcfa842b2c26e14a6e00d98d47aed8e53dde/quantization/utils/binary_packer.py)

## Measured CPU results

Eight hand-written English story openings, 537 next-token predictions. Each line starts at BOS and is scored independently without EOS. The original FP32 model scored 6.9403921 perplexity. As a cross-check, Karpathy's C tokenizer and forward pass scored the same 537 targets at 6.9403928.

| Method | Actual projection bits/weight | Relative Frobenius error before training | PPL before training | PPL after 256 QAT steps |
| --- | ---: | ---: | ---: | ---: |
| Dual-SVID | 0.632 | 0.8420 | 666,465,849 | — |
| Dual-SVID + Joint-ITQ | 0.632 | 0.8226 | 509,708,126 | — |
| Dual-SVID | 0.899 | 0.7926 | 19,239,280 | 102.72 |
| Dual-SVID + Joint-ITQ | 0.899 | 0.7459 | 851,783 | 85.53 |

The two trained archives occupy 136,082 bytes each, including the NPZ container. Their parameter payload is 92,404 bytes for 260,032 original parameters (2.84 bits/weight across the entire model). A fresh x86-64 Linux process scoring the trained ITQ archive peaked at 45.19 MiB RSS. The Python path expands the projections to FP32 on load. This measurement includes the interpreter, libraries, weights and working arrays for these short prompts. Steam Deck device speed and sustained generation were not tested.

The tiny model uses 5 transformer blocks, 35 compressed attention/MLP matrices, a tied embedding and head, and FP16 scales, embeddings and norms. We use exact truncated SVD, rank-one magnitude factors and optional 50-iteration Joint-ITQ before binarization. Rank rounds down to multiples of eight with a floor of eight. The requested 0.5 target exceeds its budget on these small shapes. The mini experiment uses one factor pair per projection, clipped straight-through gradients, Adam and teacher-logit distillation on 128 teacher-sampled prefixes of up to 32 tokens, cycled twice. Both arms see the same samples and ordering. This differs from the papers' residual branch, SmoothSign, long-context training data and training schedule. One seed and a small custom evaluation set limit the comparison.

## Run the checkpoints

Python 3.12 and a Linux CPU are sufficient. All saved models and the original 512-token vocabulary are included.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
python score.py runs/itq-qat.npz tok512.bin
python score.py runs/svid-qat.npz tok512.bin
```

To repeat compression and training from the pretrained model, run the following in the same environment. The loader verifies SHA-256 for the original weights and tokenizer. Results can vary slightly with BLAS and SVD implementations.

```bash
mkdir -p inputs
curl -fL https://huggingface.co/karpathy/tinyllamas/resolve/cbffbc90324256436c13dc6b4592c82c1e716354/stories260K/stories260K.bin -o inputs/stories.bin
python experiment.py inputs/stories.bin tok512.bin --out reproduced
python train.py inputs/stories.bin tok512.bin --out reproduced
```

## Paper-scale experiment

The papers' Llama-2-7B comparison at nominal 1 BPW reports WikiText-2 perplexities of 9.08 and 8.27, respectively, with a total weight footprint of 1.3 GB for either method. For batch one at 2,048 tokens, the standard FP16 KV cache is another 1 GiB (2 x 32 layers x 2,048 x 4,096 x 2 bytes). A runtime's unpacking strategy, temporary buffers and allocator determine actual RAM use. This project contains no measured 7B peak RSS on SteamOS.

On a CUDA training host, obtain authorized access to meta-llama/Llama-2-7b-hf and install the upstream dependencies including a CUDA-compatible PyTorch, Transformers 4.51.x and tensorboard. Run `bash paper-run.sh` from this directory. It pins SamsungLabs commit 0ceadcfa842b2c26e14a6e00d98d47aed8e53dde, trains separate baseline/ITQ checkpoints and scores WikiText-2 at 2,048 tokens plus the five Table 1 zero-shot tasks. See `outputs/*/eval.log`. This runner has only been syntax-checked here. The tiny CPU reader accepts the TinyStories layout only.

## Sources

- [LittleBit, NeurIPS 2025](https://arxiv.org/abs/2506.13771)
- [LittleBit-2, ICML 2026, Table 1](https://arxiv.org/html/2603.00042v2)
- [SamsungLabs implementation and environment instructions](https://github.com/SamsungLabs/LittleBit/tree/0ceadcfa842b2c26e14a6e00d98d47aed8e53dde)
- [Andrej Karpathy's pretrained TinyStories model and tokenizer](https://huggingface.co/karpathy/tinyllamas/tree/cbffbc90324256436c13dc6b4592c82c1e716354/stories260K)
- [llama2.c reference implementation](https://github.com/karpathy/llama2.c)
