# Hisse Senedi Analiz Platformu

Bu uygulama, hisse senedi verilerini analiz eden ve detaylı PDF raporu oluşturan bir web uygulamasıdır.

## Özellikler

- Kullanıcı kaydı ve girişi
- CSV dosyası yükleme ve analiz
- Detaylı teknik analiz raporu
- Geçmiş analizleri görüntüleme
- PDF rapor indirme

## Kurulum

1. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

## Kullanım

1. Kayıt olun veya giriş yapın
2. CSV formatındaki hisse senedi verilerinizi yükleyin
3. Hisse adını girin
4. "Analiz Et" butonuna tıklayın
5. Oluşturulan PDF raporunu indirin

## CSV Dosya Formatı

CSV dosyası aşağıdaki sütunları içermelidir:
- time (timestamp)
- open (açılış fiyatı)
- high (en yüksek fiyat)
- low (en düşük fiyat)
- close (kapanış fiyatı)
- Volume (işlem hacmi)

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.
