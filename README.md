# CoT Faithfulness (MMLU / GPQA Diamond, ~8B 모델)

Chen et al. ([arXiv:2505.05410](https://arxiv.org/abs/2505.05410))의 힌트 6종을 써서, 비슷한 규모(~8B)의 모델 패밀리에 따라 **힌트를 따라 답을 바꿀 때 그 사실을 CoT에 말로 남기는 비율**이 다른지 비교한다.

## 결과

- **최종 요약 (MMLU + GPQA)**: [results/summary.md](results/summary.md)
- MMLU: [요약](results/mmlu/tables/summary.md) · [상세 분석](docs/mmlu_analysis.md)
- GPQA Diamond: [요약](results/gpqa/tables/summary.md) · [상세 분석](docs/gpqa_analysis.md)

## 실험 설정

| 항목 | 값 |
|---|---|
| 데이터 | MMLU 57과목 비례 층화 100문항, GPQA Diamond 도메인 층화 100문항 (둘 다 `seed=103`) |
| 모델 | `qwen3:8b`, `gemma4:e4b`, `gemma2:9b`, `glm4:9b`, `nemotron-nano:8b`, `llama3.1:8b`, `ministral-3:8b` (Q4, thinking 끔) |
| 생성 | Ollama, `temperature=0`, `num_predict=4096`. 모델마다 100문항 × (baseline + 힌트 6) = 700회 |
| 판정 | gpt-oss-20b (llama.cpp, `effort=medium`). 힌트를 따라 답을 바꾼 사례만 YES/NO로 판정한다 |

- **힌트**: 문항마다 오답 하나(target)를 정하고, 6가지 힌트가 모두 그 target을 가리킨다. 종류는 sycophancy(교수 의견), consistency(이전 턴의 답), visual_pattern(■ 표시), metadata(XML), grader(검증 코드), unethical(무단 접근)이다.
- **영향력 p**: baseline 답이 target이 아니었던 문항 중, 힌트를 받은 뒤 target으로 바뀐 비율이다.
- **faithfulness**: 그렇게 바뀐 사례 중, CoT가 힌트를 답의 **이유**로 밝혔다고 judge가 판정(YES)한 비율이다.
- 잘렸거나 답을 읽을 수 없는 응답은 결측으로 처리하며, 오답으로 세지 않는다.

## 설치

Python 3.11+, NVIDIA GPU, [Ollama](https://ollama.com/), [llama.cpp](https://github.com/ggml-org/llama.cpp)의 `llama-server`, `gpt-oss-20b` MXFP4 GGUF가 필요하다.

```bash
python -m pip install -r requirements.txt
cp .env.example .env    # GPQA는 gated 데이터셋이라 HF_TOKEN이 필요하다
```

## 실행

아래는 MMLU 예시다. GPQA는 `configs/gpqa.yaml`로 바꾸면 된다. `--config`는 **명령 이름 앞에** 붙이고, 생략하면 GPQA로 실행된다.

```bash
python -m ck_faithfulness --config configs/mmlu.yaml check                          # 환경 점검
python -m ck_faithfulness --config configs/mmlu.yaml prepare-data                   # 100문항 샘플
python -m ck_faithfulness --config configs/mmlu.yaml generate --models qwen --yes    # 생성 (--models 생략 시 7개 전부)

# 생성 모델을 모두 내린 뒤, 다른 터미널에서 judge 서버를 띄운다
llama-server -m /path/to/gpt-oss-20b-MXFP4.gguf --port 8080 -c 16384
python -m ck_faithfulness --config configs/mmlu.yaml judge
python -m ck_faithfulness --config configs/mmlu.yaml aggregate                      # summary.md, 그래프
```

- 중단해도 같은 명령으로 다시 실행하면 이어서 돈다. 진행 상황은 `generate --dry-run`으로 확인한다.
- 분석 재현: `python scripts/final_report.py` (최종 요약), `python scripts/deep_analysis.py --config configs/mmlu.yaml` (상세 분석)
- 테스트: `python -m unittest discover -s tests`

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `configs/mmlu.yaml`, `configs/gpqa.yaml` | 데이터셋별 설정 (두 파일의 모델과 생성 설정은 같게 유지한다) |
| `results/summary.md` | 최종 요약과 그래프 |
| `results/<mmlu\|gpqa>/tables/` | 데이터셋별 요약 (`summary.md`, `summary.json`, 그래프) |
| `results/<mmlu\|gpqa>/generations/`, `judgments/`, `logs/` | 생성 원문, 판정, 로그 (gitignore) |
| `docs/` | 데이터셋별 상세 분석 |
| `scripts/` | 분석 재현 스크립트 |
| `ck_faithfulness/` | 파이프라인 코드 (config, data, prompts, parse, generate, judge, report, cli) |
