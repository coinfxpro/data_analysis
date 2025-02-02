import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Türkçe font desteği için
pdfmetrics.registerFont(TTFont('DejaVuSans', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))

def predict_next_day_values(df, features):
    # Veri hazırlama
    scaler = StandardScaler()
    
    # Açılış tahmini için model
    df['Next_Open'] = df['open'].shift(-1)
    df_open = df.dropna()
    X_open = df_open[features]
    y_open = df_open['Next_Open']
    X_open_scaled = scaler.fit_transform(X_open)
    model_open = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_open.fit(X_open_scaled, y_open)
    
    # Yüksek tahmin için model
    df['Next_High'] = df['high'].shift(-1)
    df_high = df.dropna()
    X_high = df_high[features]
    y_high = df_high['Next_High']
    X_high_scaled = scaler.fit_transform(X_high)
    model_high = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_high.fit(X_high_scaled, y_high)
    
    # Düşük tahmin için model
    df['Next_Low'] = df['low'].shift(-1)
    df_low = df.dropna()
    X_low = df_low[features]
    y_low = df_low['Next_Low']
    X_low_scaled = scaler.fit_transform(X_low)
    model_low = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_low.fit(X_low_scaled, y_low)
    
    # Kapanış tahmini için model
    df['Next_Close'] = df['close'].shift(-1)
    df_close = df.dropna()
    X_close = df_close[features]
    y_close = df_close['Next_Close']
    X_close_scaled = scaler.fit_transform(X_close)
    model_close = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_close.fit(X_close_scaled, y_close)
    
    # Son verilerle tahmin
    last_data = df[features].iloc[-1:].copy()
    last_data_scaled = scaler.transform(last_data)
    
    next_open = model_open.predict(last_data_scaled)[0]
    next_high = model_high.predict(last_data_scaled)[0]
    next_low = model_low.predict(last_data_scaled)[0]
    next_close = model_close.predict(last_data_scaled)[0]
    
    # Volatilite tahmini (son 20 günlük volatilitenin ortalaması ve trendi)
    volatility_trend = df['Volatility'].tail(20).mean()
    if df['Volatility'].tail(5).mean() > df['Volatility'].tail(20).mean():
        volatility_expectation = "Yüksek"
    else:
        volatility_expectation = "Normal"
    
    # Hacim bazlı alternatif senaryolar
    avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 0
    high_volume_scenario = {
        'open': next_open * 1.01,  # %1 yukarı
        'high': next_high * 1.02,  # %2 yukarı
        'low': next_low * 1.005,   # %0.5 yukarı
        'close': next_close * 1.015 # %1.5 yukarı
    }
    
    low_volume_scenario = {
        'open': next_open * 0.995,  # %0.5 aşağı
        'high': next_high * 0.99,   # %1 aşağı
        'low': next_low * 0.98,     # %2 aşağı
        'close': next_close * 0.99   # %1 aşağı
    }
    
    # Hacim analizi
    current_volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    if volume_ratio > 1.5:
        volume_trend = "Yüksek Hacim (Ortalamadan %{:.1f} fazla)".format((volume_ratio - 1) * 100)
    elif volume_ratio < 0.5:
        volume_trend = "Düşük Hacim (Ortalamadan %{:.1f} az)".format((1 - volume_ratio) * 100)
    else:
        volume_trend = "Normal Hacim"
    
    return (next_open, next_high, next_low, next_close, volatility_trend, 
            volatility_expectation, high_volume_scenario, low_volume_scenario, 
            volume_trend, volume_ratio)

def create_analysis_report(hisse_adi, csv_dosyasi):
    # CSV dosyasından verileri okuyalım
    df = pd.read_csv(csv_dosyasi)
    
    # Tarih sütununu datetime formatına çevirelim
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Temel hesaplamalar
    df['Daily_Return'] = df['close'].pct_change() * 100
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    
    # RSI hesaplama
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI'] = calculate_rsi(df['close'])
    
    # MACD hesaplama
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2*df['close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2*df['close'].rolling(window=20).std()
    
    # Feature engineering
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Open_Close_Diff'] = df['close'] - df['open']
    df['Price_Range'] = (df['high'] - df['low']) / df['close'] * 100
    df['Price_Momentum'] = df['close'].pct_change(5)
    df['Volume_Momentum'] = df['Volume'].pct_change(5) if 'Volume' in df.columns else 0
    
    # Trend Analysis
    df['Trend'] = np.where(df['MA20'] > df['MA50'], 1, -1)
    
    # Tahmin özellikleri
    features = ['open', 'high', 'low', 'close', 'Daily_Return', 'Volatility', 
                'MA20', 'MA50', 'RSI', 'High_Low_Diff', 'Open_Close_Diff', 
                'Price_Range', 'Price_Momentum']
    
    # Gelecek gün tahminleri
    (next_open, next_high, next_low, next_close, volatility_trend, 
     volatility_expectation, high_volume_scenario, low_volume_scenario,
     volume_trend, volume_ratio) = predict_next_day_values(df, features)
    
    # PDF Rapor Oluşturma
    doc = SimpleDocTemplate(f"{hisse_adi}_analiz_raporu.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Özel stil tanımlamaları
    styles.add(ParagraphStyle(
        name='TurkishStyle',
        parent=styles['Normal'],
        fontName='DejaVuSans',
        fontSize=12,
        leading=14,
    ))
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='DejaVuSans',
        fontSize=24,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName='DejaVuSans',
        fontSize=16,
        spaceAfter=12
    )
    
    # Başlık
    story.append(Paragraph(f"{hisse_adi} Hisse Senedi Analiz Raporu", title_style))
    story.append(Spacer(1, 12))
    
    # Sonuç ve Tahmin Bölümü
    story.append(Paragraph("1. Sonuc ve Tahminler", heading_style))
    prediction_info = [
        [Paragraph("Metrik", heading_style), Paragraph("Tahmin", heading_style)],
        ["Sonraki Gun Acilis", f"{next_open:.2f} TL"],
        ["Beklenen En Yuksek", f"{next_high:.2f} TL"],
        ["Beklenen En Dusuk", f"{next_low:.2f} TL"],
        ["Beklenen Kapanis", f"{next_close:.2f} TL"],
        ["Volatilite Beklentisi", f"{volatility_expectation} ({volatility_trend:.2f}%)"],
        ["Hacim Durumu", volume_trend],
    ]
    
    t = Table(prediction_info, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Alternatif Senaryolar
    story.append(Paragraph("1.1 Hacim Bazli Alternatif Senaryolar", heading_style))
    
    scenario_info = [
        [Paragraph("Metrik", heading_style), 
         Paragraph("Yuksek Hacim Senaryosu", heading_style),
         Paragraph("Dusuk Hacim Senaryosu", heading_style)],
        ["Acilis", f"{high_volume_scenario['open']:.2f} TL", f"{low_volume_scenario['open']:.2f} TL"],
        ["En Yuksek", f"{high_volume_scenario['high']:.2f} TL", f"{low_volume_scenario['high']:.2f} TL"],
        ["En Dusuk", f"{high_volume_scenario['low']:.2f} TL", f"{low_volume_scenario['low']:.2f} TL"],
        ["Kapanis", f"{high_volume_scenario['close']:.2f} TL", f"{low_volume_scenario['close']:.2f} TL"],
    ]
    
    t = Table(scenario_info, colWidths=[133, 133, 133])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    
    # Hacim Analizi Açıklaması
    volume_analysis = f"""
    Hacim Analizi:
    • Mevcut Hacim Durumu: {volume_trend}
    • Yuksek Hacim Senaryosu: Normal hacmin en az %50 uzerinde islem gerceklesmesi durumu
    • Dusuk Hacim Senaryosu: Normal hacmin %50'sinin altinda islem gerceklesmesi durumu
    
    Not: Hacim artisi genellikle fiyat hareketinin yonunu guclendirir. Yuksek hacimle gelen 
    yukselis daha guclu bir yukselis trendine, dusuk hacimle gelen yukselis ise zayif bir 
    yukselis trendine isaret edebilir.
    """
    story.append(Paragraph(volume_analysis, styles['TurkishStyle']))
    story.append(Spacer(1, 20))
    
    # Temel Analiz
    story.append(Paragraph("2. Temel Analiz", heading_style))
    current_price = df['close'].iloc[-1]
    
    basic_info = [
        [Paragraph("Metrik", heading_style), Paragraph("Deger", heading_style)],
        ["Son Kapanis", f"{current_price:.2f} TL"],
        ["20 Gunluk Ortalama", f"{df['MA20'].iloc[-1]:.2f} TL"],
        ["50 Gunluk Ortalama", f"{df['MA50'].iloc[-1]:.2f} TL"],
        ["RSI", f"{df['RSI'].iloc[-1]:.2f}"],
        ["Volatilite (20 Gunluk)", f"{df['Volatility'].iloc[-1]:.2f}%"]
    ]
    
    t = Table(basic_info, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Grafikleri oluştur ve ekle
    plt.figure(figsize=(15, 10))
    plt.plot(df.index, df['close'], label='Kapanis Fiyati')
    plt.plot(df.index, df['MA20'], label='20 Gunluk MA')
    plt.plot(df.index, df['MA50'], label='50 Gunluk MA')
    plt.plot(df.index, df['MA200'], label='200 Gunluk MA')
    plt.title(f'{hisse_adi} Hisse Fiyat Grafigi')
    plt.xlabel('Tarih')
    plt.ylabel('Fiyat (TL)')
    plt.legend()
    plt.savefig('price_analysis.png')
    plt.close()
    
    story.append(Paragraph("3. Teknik Analiz Grafikleri", heading_style))
    story.append(Image('price_analysis.png', 6*inch, 4*inch))
    story.append(Spacer(1, 12))
    
    # İstatistiksel Analiz
    story.append(Paragraph("4. Istatistiksel Analiz", heading_style))
    daily_returns = df['Daily_Return'].dropna()
    shapiro_stat, shapiro_p = stats.shapiro(daily_returns)
    
    stats_text = f"""
    • Gunluk Getiri Ortalamasi: {daily_returns.mean():.2f}%
    • Gunluk Getiri Std. Sapma: {daily_returns.std():.2f}%
    • Shapiro-Wilk Normallik Testi p-degeri: {shapiro_p:.4f}
    • Carpiklik: {daily_returns.skew():.2f}
    • Basiklik: {daily_returns.kurtosis():.2f}
    """
    story.append(Paragraph(stats_text, styles['TurkishStyle']))
    story.append(Spacer(1, 20))
    
    # Trend ve Momentum Analizi
    story.append(Paragraph("5. Trend ve Momentum Analizi", heading_style))
    current_trend = "Guclu Yukselis" if df['close'].iloc[-1] > df['MA20'].iloc[-1] > df['MA50'].iloc[-1] else \
                   "Yukselis" if df['close'].iloc[-1] > df['MA20'].iloc[-1] else \
                   "Guclu Dusus" if df['close'].iloc[-1] < df['MA20'].iloc[-1] < df['MA50'].iloc[-1] else \
                   "Dusus" if df['close'].iloc[-1] < df['MA20'].iloc[-1] else "Yatay"
    
    rsi_status = "Asiri Alim Bolgesi" if df['RSI'].iloc[-1] > 70 else \
                 "Asiri Satim Bolgesi" if df['RSI'].iloc[-1] < 30 else "Notr Bolge"
    
    trend_text = f"""
    • Mevcut Trend: {current_trend}
    • RSI Durumu: {rsi_status} ({df['RSI'].iloc[-1]:.2f})
    • Momentum: {df['Price_Momentum'].iloc[-1]:.2f}
    """
    story.append(Paragraph(trend_text, styles['TurkishStyle']))
    
    # PDF oluştur
    doc.build(story)
    
    # Geçici dosyaları temizle
    if os.path.exists('price_analysis.png'):
        os.remove('price_analysis.png')
    
    print(f"\n{hisse_adi}_analiz_raporu.pdf dosyasi olusturuldu.")

if __name__ == "__main__":
    hisse_adi = input("Analiz edilecek hisse adini girin (orn: PAPIL): ")
    csv_dosyasi = input("CSV dosyasinin adini girin: ")
    
    if not os.path.exists(csv_dosyasi):
        print("Hata: CSV dosyasi bulunamadi!")
    else:
        create_analysis_report(hisse_adi, csv_dosyasi)
