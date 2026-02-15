# Prompt
```bash
git clone https://github.com/kmjeong98/DCA_auto.git
cd DCA_auto
```


# DCA_auto

이 프로젝트는 M4 Mac mini에서 구동할 에이전트입니다.

## 구조
- 실전 매매: `main_trading.py`
- GA 최적화: `main_optimize.py`

## 주의
- `.env` 파일에 API 키/시크릿을 저장하세요.
- `data/`와 `logs/`는 Git에 포함되지 않습니다.

## Optimize 실행 prompt
```bash
cd /path/to/DCA_auto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main_optimize.py
```
