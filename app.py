import streamlit as st
import pandas as pd
import os
from veri_analizi_V1 import create_analysis_report
import sqlite3
import hashlib
from datetime import datetime

# Veritabanı bağlantısı
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_history
                 (username TEXT, hisse_adi TEXT, analysis_date TEXT, 
                  report_path TEXT, FOREIGN KEY(username) REFERENCES users(username))''')
    conn.commit()
    conn.close()

# Şifre hashleme
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Kullanıcı kaydı
def register_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        hash_pw = hash_password(password)
        c.execute("INSERT INTO users VALUES (?, ?, ?)", 
                 (username, hash_pw, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Kullanıcı girişi
def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

# Analiz geçmişini kaydet
def save_analysis(username, hisse_adi, report_path):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO analysis_history VALUES (?, ?, ?, ?)",
             (username, hisse_adi, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), report_path))
    conn.commit()
    conn.close()

# Analiz geçmişini getir
def get_analysis_history(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT hisse_adi, analysis_date, report_path FROM analysis_history WHERE username=? ORDER BY analysis_date DESC", (username,))
    history = c.fetchall()
    conn.close()
    return history

# Streamlit uygulama arayüzü
def main():
    st.set_page_config(page_title="Hisse Senedi Analiz Platformu", layout="wide")
    
    # Veritabanını başlat
    init_db()
    
    # Session state kontrolü
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    # Sidebar menü
    menu = st.sidebar.selectbox("Menü", ["Giriş", "Kayıt Ol", "Analiz", "Geçmiş Analizler"])
    
    if menu == "Giriş" and not st.session_state.logged_in:
        st.title("Kullanıcı Girişi")
        username = st.text_input("Kullanıcı Adı")
        password = st.text_input("Şifre", type="password")
        
        if st.button("Giriş Yap"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Başarıyla giriş yapıldı!")
                st.experimental_rerun()
            else:
                st.error("Hatalı kullanıcı adı veya şifre!")
                
    elif menu == "Kayıt Ol" and not st.session_state.logged_in:
        st.title("Yeni Kullanıcı Kaydı")
        new_username = st.text_input("Kullanıcı Adı")
        new_password = st.text_input("Şifre", type="password")
        confirm_password = st.text_input("Şifre Tekrar", type="password")
        
        if st.button("Kayıt Ol"):
            if new_password != confirm_password:
                st.error("Şifreler eşleşmiyor!")
            elif register_user(new_username, new_password):
                st.success("Kayıt başarılı! Giriş yapabilirsiniz.")
            else:
                st.error("Bu kullanıcı adı zaten kullanılıyor!")
                
    elif menu == "Analiz" and st.session_state.logged_in:
        st.title("Hisse Senedi Analizi")
        
        uploaded_file = st.file_uploader("CSV Dosyası Yükle", type=['csv'])
        hisse_adi = st.text_input("Hisse Adı (örn: THYAO)")
        
        if uploaded_file and hisse_adi:
            if st.button("Analiz Et"):
                # Geçici dosya oluştur
                temp_csv = f"temp_{st.session_state.username}_{hisse_adi}.csv"
                with open(temp_csv, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Analiz yap
                    create_analysis_report(hisse_adi, temp_csv)
                    
                    # PDF dosyasının yolu
                    pdf_path = f"{hisse_adi}_analiz_raporu.pdf"
                    
                    # Analizi kaydet
                    save_analysis(st.session_state.username, hisse_adi, pdf_path)
                    
                    # PDF'i göster
                    with open(pdf_path, "rb") as pdf_file:
                        PDFbyte = pdf_file.read()
                        st.download_button(label="Raporu İndir", 
                                        data=PDFbyte,
                                        file_name=f"{hisse_adi}_analiz_raporu.pdf",
                                        mime='application/octet-stream')
                    
                    st.success("Analiz tamamlandı! Raporu indirebilirsiniz.")
                    
                except Exception as e:
                    st.error(f"Analiz sırasında bir hata oluştu: {str(e)}")
                
                finally:
                    # Geçici dosyayı temizle
                    if os.path.exists(temp_csv):
                        os.remove(temp_csv)
                
    elif menu == "Geçmiş Analizler" and st.session_state.logged_in:
        st.title("Geçmiş Analizler")
        history = get_analysis_history(st.session_state.username)
        
        if history:
            for hisse, date, report_path in history:
                col1, col2, col3 = st.columns([2,2,1])
                with col1:
                    st.write(f"Hisse: {hisse}")
                with col2:
                    st.write(f"Tarih: {date}")
                with col3:
                    if os.path.exists(report_path):
                        with open(report_path, "rb") as pdf_file:
                            st.download_button(label="Raporu İndir", 
                                            data=pdf_file.read(),
                                            file_name=f"{hisse}_analiz_raporu.pdf",
                                            mime='application/octet-stream')
                    else:
                        st.write("Rapor bulunamadı")
                st.markdown("---")
        else:
            st.info("Henüz analiz geçmişiniz bulunmuyor.")
    
    # Çıkış yap butonu
    if st.session_state.logged_in:
        if st.sidebar.button("Çıkış Yap"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()
