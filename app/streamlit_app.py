import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gemini_model import GeminiChatbot
from models.huggingface_model import HuggingFaceChatbot

st.set_page_config(
    page_title="🛍️ E-Ticaret Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #000000;
    }
    .user-message {
        background-color: #DCF8C6;
        margin-left: 20%;
        color: #000000;
    }
    .bot-message {
        background-color: #F1F1F1;
        margin-right: 20%;
        color: #000000;
    }
    .chat-message strong {
        color: #000000;
    }
    .chat-message small {
        color: #555555;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        color: #000000;
    }
    .metric-card h2 {
        color: #000000;
        font-weight: bold;
    }
    .metric-card h3 {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    """Veri setini yükle"""
    try:
        data = pd.read_csv('data/ecommerce_dataset.csv')
        return data
    except FileNotFoundError:
        st.error("Veri seti bulunamadı! data/ecommerce_dataset.csv dosyasını kontrol edin.")
        return None

def initialize_chatbots():
    """Chatbot'ları başlat"""
    chatbots = {}
    
    # API Key kontrolü burada yapılıyor.
    gemini_key = st.sidebar.text_input("Gemini API Key:", type="password")
    use_huggingface = st.sidebar.checkbox("🤗 Hugging Face Kullan", value=True)
    
    if gemini_key:
        try:
            chatbots['Gemini'] = GeminiChatbot(api_key=gemini_key)
            st.sidebar.success("✅ Gemini modeli hazır!")
        except Exception as e:
            st.sidebar.error(f"❌ Gemini modeli hatası: {e}")
    
    if use_huggingface:
        try:
            with st.spinner("🤗 Hugging Face modeli yükleniyor..."):
                chatbots['Hugging Face'] = HuggingFaceChatbot()
            st.sidebar.success("✅ Hugging Face modeli hazır!")
        except Exception as e:
            st.sidebar.error(f"❌ Hugging Face modeli hatası: {e}")
            st.sidebar.info("Çözüm: pip install transformers torch")
    
    return chatbots

def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">🛍️ E-Ticaret Chatbot</h1>', unsafe_allow_html=True)
    
    # Yan menü
    st.sidebar.title("⚙️ Ayarlar")
    
    chatbots = initialize_chatbots()
    
    # Ana menü
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Veri Analizi", "🔬 Model Karşılaştırma", "📈 Performans"])
    
    with tab1:
        st.header("Chatbot ile Konuş")
        
        if not chatbots:
            st.warning("⚠️ En az bir model seçin!")
            st.info("""
            **🆓 ÜCRETSİZ SEÇENEKLER:**
            - **🤗 Hugging Face:** Tamamen ücretsiz! (Checkbox'ı işaretle)
            - **🤖 Gemini:** https://makersuite.google.com/app/apikey (Aylık 1000 istek ücretsiz)
            """)
        else:
            selected_model = st.selectbox(
                "Model Seçin:",
                options=list(chatbots.keys()),
                help="Konuşmak istediğiniz AI modelini seçin"
            )
            
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>👤 Sen:</strong> {message['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message bot-message">
                            <strong>🤖 {message['model']}:</strong> {message['content']}<br>
                            <small>Intent: {message['intent']} | Güven: {message['confidence']:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            
            if 'input_counter' not in st.session_state:
                st.session_state.input_counter = 0
                
            user_input = st.text_input("💬 Mesajınızı yazın:", key=f"chat_input_{st.session_state.input_counter}", placeholder="Merhaba, size nasıl yardımcı olabilirim?")
            send_button = st.button("📤 Gönder", type="primary")
            
            if (user_input and send_button) and selected_model in chatbots:
                # Kullanıcının gireceği mesaj 
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                with st.spinner(f"{selected_model} düşünüyor..."):
                    result = chatbots[selected_model].chat(user_input)
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': result['response'],
                        'model': selected_model,
                        'intent': result['intent'],
                        'confidence': result['confidence']
                    })
                
                st.session_state.input_counter += 1
                st.rerun()
            
            if st.button("🗑️ Konuşmayı Temizle"):
                st.session_state.chat_history = []
                st.rerun()
    
    with tab2:
        st.header("📊 Veri Seti Analizi")
        
        data = load_dataset()
        if data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Intent Dağılımı")
                intent_counts = data['intent'].value_counts()
                
                fig = px.bar(
                    x=intent_counts.values,
                    y=intent_counts.index,
                    orientation='h',
                    title="Intent Sayıları",
                    color=intent_counts.values,
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🥧 Intent Oranları")
                fig = px.pie(
                    values=intent_counts.values,
                    names=intent_counts.index,
                    title="Intent Dağılım Yüzdeleri"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Veri Seti İstatistikleri")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📝 Toplam Örnek</h3>
                    <h2>{len(data)}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Intent Sayısı</h3>
                    <h2>{data['intent'].nunique()}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_length = data['text'].str.len().mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📏 Ort. Metin Uzunluğu</h3>
                    <h2>{avg_length:.1f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                min_samples = intent_counts.min()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚖️ Min Örnek/Intent</h3>
                    <h2>{min_samples}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("🔍 Örnek Veriler")
            st.dataframe(data.head(10), use_container_width=True)
    
    with tab3:
        st.header("🔬 Model Karşılaştırması")
        
        if len(chatbots) < 2:
            st.warning("⚠️ Model karşılaştırması için en az 2 model API anahtarı gerekli!")
        else:
            data = load_dataset()
            if data is not None:
                st.info("⏳ Model karşılaştırması uzun sürebilir. Lütfen bekleyiniz...")
                
                if st.button("🚀 Modelleri Karşılaştır"):
                    train_data, test_data = train_test_split(
                        data, test_size=0.2, random_state=42, stratify=data['intent']
                    )
                    
                    results = {}
                    
                    for model_name, chatbot in chatbots.items():
                        with st.spinner(f"{model_name} modeli değerlendiriliyor..."):
                            result = chatbot.evaluate_model(test_data.head(20))  # Hız için ilk 20 örnek 
                            results[model_name] = result
                    
                    st.success("✅ Değerlendirme tamamlandı!")
                    
                    metrics_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
                        'Precision': [results[model]['precision'] for model in results.keys()],
                        'Recall': [results[model]['recall'] for model in results.keys()],
                        'F1 Score': [results[model]['f1_score'] for model in results.keys()]
                    })
                    
                    st.subheader("📊 Model Performansları")
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    fig = px.bar(
                        metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                        x='Model',
                        y='Score',
                        color='Metric',
                        barmode='group',
                        title="Model Performans Karşılaştırması"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("📈 Performans Detayları")
        st.info("Bu bölüm model değerlendirme sonrası görüntülenecektir.")
        
        # grafikler
        st.subheader("🎯 Intent Accuracy Breakdown")
        st.info("Model karşılaştırması yapıldıktan sonra intent bazında accuracy gösterilecek.")
        
        st.subheader("🔄 Confusion Matrix")
        st.info("Modellerin confusion matrix'leri burada görüntülenecek.")

if __name__ == "__main__":
    main() 