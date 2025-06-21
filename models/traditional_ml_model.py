"""
Geleneksel ML ile Intent Classification
Bu gerçek model eğitimi örneğidir!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import re
import string

class TraditionalMLChatbot:
    def __init__(self):
        """Geleneksel ML tabanlı chatbot"""
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.label_encoder = None
        
    def preprocess_text(self, text):
        """Metin ön işleme"""
        text = text.lower()
        
        turkish_chars = {'ç':'c', 'ğ':'g', 'ı':'i', 'ö':'o', 'ş':'s', 'ü':'u'}
        for turkish, english in turkish_chars.items():
            text = text.replace(turkish, english)
        
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        text = ' '.join(text.split())
        
        return text
    
    def prepare_data(self, data_path="data/ecommerce_dataset.csv"):
        """Veriyi hazırla"""
        print("📊 Veri hazırlanıyor...")
        
        df = pd.read_csv(data_path)
        
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        X = df['processed_text']
        y = df['intent']
        
        print(f"✅ Toplam örnek: {len(df)}")
        print(f"✅ Intent sayısı: {y.nunique()}")
        
        return X, y
    
    def train_model(self, X, y, model_type='random_forest'):
        """Model eğitimi - GERÇEK EĞİTİM!"""
        print(f"🏋️ {model_type.upper()} modeli eğitiliyor...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            model = SVC(kernel='rbf', random_state=42)
        else:
            raise ValueError("Desteklenmeyen model tipi!")
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),  
                stop_words=None
            )),
            ('classifier', model)
        ])
        
        print("🔥 Eğitim başlıyor...")
        self.pipeline.fit(X_train, y_train)
        
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Eğitim tamamlandı!")
        print(f"📈 Test Accuracy: {accuracy:.3f}")
        
        print("\n📊 DETAYLI PERFORMANS RAPORU:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test
        }
    
    def save_model(self, filepath="trained_chatbot_model.pkl"):
        """Eğitilmiş modeli kaydet"""
        joblib.dump(self.pipeline, filepath)
        print(f"💾 Model kaydedildi: {filepath}")
    
    def load_model(self, filepath="trained_chatbot_model.pkl"):
        """Kaydedilmiş modeli yükle"""
        self.pipeline = joblib.load(filepath)
        print(f"📂 Model yüklendi: {filepath}")
    
    def predict_intent(self, text):
        """Intent tahmin et"""
        if self.pipeline is None:
            raise ValueError("Model eğitilmemiş!")
        
        processed_text = self.preprocess_text(text)
        intent = self.pipeline.predict([processed_text])[0]
        
        try:
            proba = self.pipeline.predict_proba([processed_text])
            confidence = max(proba[0])
        except:
            confidence = 0.8  
        
        return {
            'intent': intent,
            'confidence': confidence,
            'processed_text': processed_text
        }
    
    def chat(self, user_message):
        """Chatbot ana fonksiyonu"""
        result = self.predict_intent(user_message)
        
        responses = {
            'greeting': "Merhaba! Size nasıl yardımcı olabilirim?",
            'product_inquiry': "Hangi ürün hakkında bilgi almak istiyorsunuz?",
            'order_status': "Sipariş numaranızı paylaşabilir misiniz?",
            'cart_operations': "Sepet işlemlerinizde size yardımcı olabilirim.",
            'payment_issues': "Ödeme konusunda nasıl yardımcı olabilirim?",
            'return_refund': "İade işleminiz için size yardımcı olabilirim.",
            'shipping_info': "Kargo bilgileri hakkında ne öğrenmek istiyorsunuz?",
            'goodbye': "İyi günler! Tekrar görüşmek üzere.",
            'complaint': "Sorununuzu anlıyorum. Nasıl yardımcı olabilirim?"
        }
        
        response = responses.get(result['intent'], "Üzgünüm, anlayamadım.")
        
        return {
            'intent': result['intent'],
            'confidence': result['confidence'],
            'response': response
        }

def main():
    """Gerçek model eğitimi demo"""
    print("🚀 GERÇEK MODEL EĞİTİMİ BAŞLIYOR!")
    print("="*50)
    
    chatbot = TraditionalMLChatbot()
    
    X, y = chatbot.prepare_data()
    
    models = ['random_forest', 'logistic_regression', 'svm']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*30}")
        result = chatbot.train_model(X, y, model_type)
        results[model_type] = result['accuracy']
    
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\n🏆 EN İYİ MODEL: {best_model[0]} (Accuracy: {best_model[1]:.3f})")
    
    chatbot.train_model(X, y, best_model[0])
    chatbot.save_model()
    
    print(f"\n💬 CHATBOT TESTİ:")
    test_messages = [
        "Merhaba nasılsınız",
        "Bu ürünün fiyatı nedir",
        "Siparişim nerede",
        "Teşekkürler görüşürüz"
    ]
    
    for msg in test_messages:
        result = chatbot.chat(msg)
        print(f"👤 Kullanıcı: {msg}")
        print(f"🤖 Intent: {result['intent']} (Güven: {result['confidence']:.2f})")
        print(f"🤖 Bot: {result['response']}")
        print("-" * 40)

if __name__ == "__main__":
    main() 