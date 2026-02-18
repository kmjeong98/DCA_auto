# DCA_auto

Binance Futures 양방향 DCA 자동매매 봇.

Long/Short 동시 운영, GA 최적화 파라미터 기반 자동 진입/DCA/TP/SL을 수행합니다.

## 프로젝트 구조

```
DCA_auto/
├── main_trading.py              # 실전 매매 진입점 (PM2 구동)
├── main_optimize.py             # GA 최적화 실행
├── config.json                  # 트레이딩 설정 (코인별 weight, cooldown)
├── ecosystem.config.js          # PM2 실행 설정 (가상환경 interpreter 지정)
├── requirements.txt
├── .env.example
│
├── src/
│   ├── common/
│   │   ├── api_client.py        # Binance Futures REST API (binance-futures-connector)
│   │   ├── trading_config.py    # config.json 로더
│   │   ├── config_loader.py     # 최적화 파라미터 로더 (data/params/)
│   │   ├── data_manager.py      # OHLCV 데이터 수집
│   │   └── logger.py            # 로깅 설정
│   │
│   ├── trading/
│   │   ├── executor.py          # 메인 트레이딩 실행기 (SymbolTrader + TradingExecutor)
│   │   ├── strategy.py          # DCA 전략 계산 (진입, DCA 레벨, TP/SL 가격)
│   │   ├── price_feed.py        # Mark Price WebSocket + User Data Stream
│   │   ├── margin_manager.py    # 코인별 마진 영속화 (정전 복구)
│   │   └── state_manager.py     # 포지션 상태 저장/복구
│   │
│   └── optimization/
│       ├── ga_engine.py         # GA 최적화 엔진
│       └── backtester.py        # 백테스트 엔진
│
├── data/                        # Git 미포함
│   ├── params/                  # GA 최적화 결과 (BTC_USDT.json, ...)
│   └── margins/                 # 코인별 마진 상태 파일
│
└── logs/                        # Git 미포함
```

## 트레이딩 로직

### 양방향 DCA

각 코인에 대해 Long과 Short 포지션을 **동시에** 운영합니다.

1. **Base Order** (MARKET) — 즉시 체결로 빠른 진입
2. **DCA Orders** (LIMIT) — Base 체결가 기준으로 기하급수 간격 배치
3. **Take Profit** (LIMIT, reduceOnly) — 평균 단가 기준, 슬리피지 없는 정확한 체결
4. **Stop Loss** (STOP_MARKET, Mark Price 트리거) — 마크 가격이 SL에 도달하면 즉시 시장가 청산

### DCA 레벨 계산

GA 최적화로 결정된 `price_deviation`, `dev_multiplier`, `vol_multiplier`를 사용합니다.

- 각 레벨의 트리거 가격은 이전 레벨에서 `dev_multiplier`만큼 deviation이 확대
- 각 레벨의 마진은 이전 레벨에서 `vol_multiplier`만큼 증가
- DCA 체결 시 평균 단가 재계산 후 SL/TP를 재배치

### 주문 체결 감지

- **User Data Stream** (WebSocket) — `ORDER_TRADE_UPDATE` 이벤트로 실시간 감지
- **폴링 백업** — 5분마다 미체결 주문 대조로 누락 방지
- Listen Key는 30분마다 자동 갱신

### 포지션 라이프사이클

```
진입 (MARKET) → DCA 배치 (LIMIT) + SL 배치 (STOP_MARKET) + TP 배치 (LIMIT)
  │
  ├── DCA 체결 → avg 재계산 → SL/TP 재배치
  ├── TP 체결  → 상태 리셋 → 마진 업데이트 → 즉시 재진입
  └── SL 체결  → 상태 리셋 → 마진 업데이트 → 쿨다운 대기
```

## 자본 관리

### 자본 배분 (config.json)

봇은 시작 시 Binance 총 잔고(`totalWalletBalance`)를 조회하고, `weight` 비율로 코인별 자본을 배분합니다.

```json
{
  "cooldown_hours": 6,
  "symbols": {
    "BTC_USDT": { "weight": 0.40 },
    "ETH_USDT": { "weight": 0.35 },
    "SOL_USDT": { "weight": 0.25 }
  }
}
```

- weight 합계 < 1.0: 여유분 확보 (추가 코인 대비)
- weight 합계 = 1.0: 전액 배분
- weight 합계 > 1.0: 오버레버리지 (허용)
- **자동 정규화 없음** — 설정 그대로 사용

### Cross 마진 모드

모든 포지션은 Cross 마진으로 운영합니다. 지갑 잔고를 공유하므로 개별 포지션이 단독으로 청산되지 않습니다.

레버리지는 심볼당 하나만 설정 가능하므로, Long/Short 중 큰 값(`max(long_lev, short_lev)`)을 거래소에 설정하고 전략 내부에서는 각 방향의 레버리지를 별도로 사용합니다.

### 마진 영속화 (MarginManager)

각 코인의 할당 자본은 `data/margins/{SYMBOL}_margin.json`에 저장됩니다.

- **정전 복구**: 재시작 시 마진 파일을 읽어 이전 자본으로 복구
- **업데이트 조건**: 해당 코인의 Long/Short **양쪽 모두** 비활성(TP/SL 후)일 때만 업데이트
- **감소 방지**: 잔고가 줄어도 기존 마진을 유지 (DCA 특성상 사이클 내에서 복구 가능)
- **증가 반영**: 잔고 증가 시 새로운 자본 반영

## 설정 방법

### 1. 의존성 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 환경 변수

```bash
cp .env.example .env
```

`.env` 파일에 API 키를 설정합니다:

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
USE_TESTNET=true
```

### 3. GA 최적화 (파라미터 생성)

```bash
python3 main_optimize.py
```

결과는 `data/params/`에 코인별 JSON으로 저장됩니다. 이 파일이 없으면 매매를 시작할 수 없습니다.

### 4. config.json 설정

거래할 코인과 자본 비율을 설정합니다. 심볼 이름은 `data/params/`의 파일명과 일치해야 합니다.

## 실행

```bash
# Testnet
python3 main_trading.py --testnet

# Mainnet
python3 main_trading.py --mainnet

# 커스텀 설정 파일
python3 main_trading.py --config my_config.json
```

### PM2로 상시 구동

프로젝트 루트의 `ecosystem.config.js`를 사용합니다. 가상환경(`.venv`)의 Python을 interpreter로 지정하여 패키지를 올바르게 인식합니다.

```bash
pm2 start ecosystem.config.js
pm2 save
```

Testnet으로 실행하려면 `ecosystem.config.js`의 `args`를 `"--testnet"`으로 변경합니다.

## 기술 스택

- **Binance API**: `binance-futures-connector` (REST: `UMFutures`, WebSocket: `UMFuturesWebsocketClient`)
- **최적화**: Genetic Algorithm (NumPy + Numba JIT)
- **백테스트**: Numba 가속 벡터화 연산
