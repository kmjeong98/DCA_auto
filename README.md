# Prompt
```bash
git clone https://github.com/kmjeong98/DCA_auto.git
cd DCA_auto
```


# DCA_auto

Binance Futures 양방향 DCA 자동매매 봇 (M4 Mac mini에서 PM2로 구동).

## 구조
- 실전 매매: `main_trading.py`
- GA 최적화: `main_optimize.py`
- 트레이딩 설정: `config.json`
- 최적화 파라미터: `data/params/`
- 마진 상태 파일: `data/margins/`

## 설정

### 1. 환경 변수 (.env)
```bash
cp .env.example .env
# .env 파일에 Binance API 키/시크릿 설정
```

### 2. 트레이딩 설정 (config.json)
```json
{
  "cooldown_hours": 6,
  "fee_rate": 0.0005,
  "symbols": {
    "BTC_USDT": { "weight": 0.40 },
    "ETH_USDT": { "weight": 0.35 },
    "SOL_USDT": { "weight": 0.25 }
  }
}
```
- `weight`: Binance 총 잔고 대비 코인별 자본 배분 비율
- weight 합계가 1 미만이면 여유분 확보, 1 초과도 허용 (오버레버리지)

## 실행

### GA 최적화
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main_optimize.py
```

### 실전 매매
```bash
# Testnet
python3 main_trading.py --testnet

# Mainnet
python3 main_trading.py --mainnet

# 커스텀 설정 파일
python3 main_trading.py --config my_config.json
```

## 주의
- `.env` 파일에 API 키/시크릿을 저장하세요.
- `data/`와 `logs/`는 Git에 포함되지 않습니다.
- Mainnet 실행 시 확인 프롬프트가 표시됩니다.
