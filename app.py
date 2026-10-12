from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

app = Flask(__name__)

stocks = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "TSLA": "Tesla",
    "AVGO": "Broadcom",
    "GOOG": "Alphabet",
    "MU": "Micron",
    "AMD": "AMD",
    "NFLX": "Netflix",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "RKLB": "Rocket Lab"
}

FEATURES = [
    "Return1D",
    "Return5D",
    "SMA20Gap",
    "SMA50Gap",
    "Volatility20",
    "RSI",
    "VolumeChange"
]


def prepare_data(raw):
    if raw is None or raw.empty:
        raise ValueError("ไม่พบข้อมูลหุ้น กรุณาลองใหม่อีกครั้ง")

    data = raw.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Close", "Volume"]

    if not all(col in data.columns for col in required):
        raise ValueError("ข้อมูลราคาหุ้นไม่ครบถ้วน")

    data = data[required].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    data = data[
        ~data.index.duplicated(keep="last")
    ].sort_index()

    if len(data) < 150:
        raise ValueError("ข้อมูลย้อนหลังไม่เพียงพอ")

    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()

    data["Return1D"] = data["Close"].pct_change()
    data["Return5D"] = data["Close"].pct_change(5)

    data["SMA20Gap"] = data["Close"] / data["SMA20"] - 1
    data["SMA50Gap"] = data["Close"] / data["SMA50"] - 1

    data["Volatility20"] = data["Return1D"].rolling(20).std()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    data.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    data.loc[(avg_gain == 0) & (avg_loss == 0), "RSI"] = 50

    average_volume = data["Volume"].rolling(20).mean()
    data["VolumeChange"] = data["Volume"] / average_volume - 1

    data["Target"] = (
        data["Close"].shift(-1) > data["Close"]
    ).astype(int)

    data = data.replace([np.inf, -np.inf], np.nan)

    return data


def explain_stock(row):
    up = []
    down = []
    notes = []

    price = float(row["Close"])
    sma20 = float(row["SMA20"])
    sma50 = float(row["SMA50"])
    rsi = float(row["RSI"])
    return5 = float(row["Return5D"]) * 100
    volatility = float(row["Volatility20"]) * 100

    if price > sma20:
        up.append(
            "ราคาสูงกว่า SMA20 สนับสนุนแนวโน้มระยะสั้น"
        )
    else:
        down.append(
            "ราคาต่ำกว่า SMA20 สะท้อนแรงกดดันระยะสั้น"
        )

    if price > sma50:
        up.append(
            "ราคาสูงกว่า SMA50 สนับสนุนแนวโน้มระยะกลาง"
        )
    else:
        down.append(
            "ราคาต่ำกว่า SMA50 สะท้อนแนวโน้มระยะกลางที่อ่อนแอ"
        )

    if rsi >= 70:
        down.append(
            f"RSI อยู่ที่ {rsi:.1f} เข้าสู่โซนซื้อมาก "
            "จึงควรระวังความเสี่ยงจากการพักตัว"
        )
        notes.append(
            "RSI สูงไม่ได้หมายความว่าราคาจะต้องลดลงทันที"
        )
    elif rsi <= 30:
        up.append(
            f"RSI อยู่ที่ {rsi:.1f} เข้าสู่โซนขายมาก "
            "ราคาอาจมีโอกาสฟื้นตัว"
        )
        notes.append(
            "RSI ต่ำไม่ได้ยืนยันว่าราคาจะกลับขึ้น"
        )
    elif rsi >= 50:
        up.append(
            f"RSI อยู่ที่ {rsi:.1f} โมเมนตัมอยู่เหนือระดับกลาง"
        )
    else:
        down.append(
            f"RSI อยู่ที่ {rsi:.1f} โมเมนตัมต่ำกว่าระดับกลาง"
        )

    if return5 > 2:
        up.append(
            f"ราคาปรับขึ้น {return5:.2f}% ใน 5 วันทำการล่าสุด"
        )
    elif return5 < -2:
        down.append(
            f"ราคาปรับลง {abs(return5):.2f}% "
            "ใน 5 วันทำการล่าสุด"
        )
    else:
        notes.append(
            f"ผลตอบแทน 5 วันอยู่ที่ {return5:.2f}% "
            "ยังไม่แสดงทิศทางชัดเจน"
        )

    notes.append(
        f"ความผันผวนรายวันย้อนหลัง 20 วันอยู่ที่ "
        f"{volatility:.2f}%"
    )

    if not up:
        up.append("ยังไม่พบสัญญาณบวกเด่นจากตัวชี้วัดที่ใช้")

    if not down:
        down.append("ยังไม่พบสัญญาณลบเด่น แต่ยังมีความเสี่ยง")

    return up, down, notes


def analyze_stock(ticker):
    if ticker not in stocks:
        raise ValueError("ไม่พบหุ้นที่เลือก")

    raw = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    data = prepare_data(raw)

    usable = data.dropna(subset=FEATURES).copy()

    if len(usable) < 150:
        raise ValueError("ข้อมูลสำหรับวิเคราะห์ไม่เพียงพอ")

    # แถวล่าสุดยังไม่มีผลราคาวันถัดไป จึงไม่ใช้เป็นข้อมูลฝึก
    labeled = usable.iloc[:-1].copy()
    latest = usable.iloc[[-1]].copy()

    split = int(len(labeled) * 0.8)

    X_train = labeled[FEATURES].iloc[:split]
    y_train = labeled["Target"].iloc[:split]

    X_test = labeled[FEATURES].iloc[split:]
    y_test = labeled["Target"].iloc[split:]

    if len(X_test) < 20 or y_train.nunique() < 2:
        raise ValueError("ข้อมูลไม่เพียงพอสำหรับทดสอบโมเดล")

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    test_prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test, test_prediction
    ) * 100

    balanced_accuracy = balanced_accuracy_score(
        y_test, test_prediction
    ) * 100

    probability = model.predict_proba(latest[FEATURES])[0]

    class_probability = {
        int(label): float(prob)
        for label, prob in zip(model.classes_, probability)
    }

    up_probability = class_probability.get(1, 0) * 100
    down_probability = class_probability.get(0, 0) * 100

    row = latest.iloc[0]

    up_reasons, down_reasons, notes = explain_stock(row)

    feature_names = {
        "Return1D": "ผลตอบแทน 1 วัน",
        "Return5D": "ผลตอบแทน 5 วัน",
        "SMA20Gap": "ระยะห่างจาก SMA20",
        "SMA50Gap": "ระยะห่างจาก SMA50",
        "Volatility20": "ความผันผวน 20 วัน",
        "RSI": "RSI",
        "VolumeChange": "ปริมาณซื้อขาย"
    }

    importance = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True
    )

    chart_data = usable.tail(90)

    chart = {
        "labels": [
            date.strftime("%Y-%m-%d")
            for date in chart_data.index
        ],
        "prices": [
            round(float(value), 2)
            for value in chart_data["Close"]
        ],
        "sma20": [
            round(float(value), 2)
            for value in chart_data["SMA20"]
        ],
        "sma50": [
            round(float(value), 2)
            for value in chart_data["SMA50"]
        ]
    }

    return {
        "ticker": ticker,
        "name": stocks[ticker],
        "price": float(row["Close"]),
        "date": latest.index[0].strftime("%d/%m/%Y"),
        "prediction": (
            "UP" if up_probability >= down_probability else "DOWN"
        ),
        "up_probability": up_probability,
        "down_probability": down_probability,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "rsi": float(row["RSI"]),
        "sma20": float(row["SMA20"]),
        "sma50": float(row["SMA50"]),
        "return5d": float(row["Return5D"]) * 100,
        "volatility": float(row["Volatility20"]) * 100,
        "up_reasons": up_reasons,
        "down_reasons": down_reasons,
        "notes": notes,
        "importance": [
            {
                "label": feature_names[name],
                "value": round(float(value) * 100, 1)
            }
            for name, value in importance[:5]
        ],
        "chart": chart
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    selected = "NVDA"

    if request.method == "POST":
        selected = request.form.get("ticker", "NVDA")

        try:
            result = analyze_stock(selected)
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Stock analysis failed")
            error = (
                "ระบบดึงหรือวิเคราะห์ข้อมูลไม่สำเร็จ "
                "กรุณาตรวจสอบอินเทอร์เน็ตแล้วลองใหม่"
            )

    return render_template(
        "index.html",
        stocks=stocks,
        result=result,
        error=error,
        selected=selected
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(__import__("os").environ.get("PORT", 5000)),
        debug=False
    )