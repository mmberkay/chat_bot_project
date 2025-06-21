"""
Hugging Face Transformers ile Ücretsiz Intent Classification
OpenAI alternatifi - tamamen ücretsiz!
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import warnings
warnings.filterwarnings('ignore')

class HuggingFaceChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """Hugging Face tabanlı chatbot"""
        self.model_name = model_name
        self.classifier = None
        self.chat_pipeline = None
        
        try:
            print("🤖 Hugging Face modeli yükleniyor...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Hugging Face modeli hazır!")
        except Exception as e:
            print(f"⚠️ Alternatif model deneniyor: {e}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  
            )
        
        self.intents = [
            'greeting', 'product_inquiry', 'order_status', 
            'cart_operations', 'payment_issues', 'return_refund',
            'shipping_info', 'goodbye', 'complaint'
        ]
        
        self.intent_descriptions = {
            'greeting': 'selamlama, merhaba, hoşgeldin mesajları',
            'product_inquiry': 'ürün sorguları, fiyat, özellik, stok bilgileri',
            'order_status': 'sipariş durumu, kargo takibi, teslimat sorguları',
            'cart_operations': 'sepet işlemleri, ürün ekleme, çıkarma',
            'payment_issues': 'ödeme sorunları, kredi kartı, ödeme yöntemleri',
            'return_refund': 'iade işlemleri, para iadesi, ürün değişimi',
            'shipping_info': 'kargo bilgileri, teslimat, kargo firması',
            'goodbye': 'vedalaşma, hoşçakal, görüşürüz mesajları',
            'complaint': 'şikayet, memnuniyetsizlik, sorun bildirimi'
        }
        
        self.intent_responses = {
            'greeting': [
                "Merhaba! E-ticaret mağazamıza hoş geldiniz. Size nasıl yardımcı olabilirim?",
                "Selam! Bugün hangi ürünleri arıyorsunuz?",
                "İyi günler! Alışveriş deneyiminizde size yardımcı olmaktan mutluluk duyarım."
            ],
            'product_inquiry': [
                "Hangi ürün hakkında bilgi almak istiyorsunuz? Size detaylı bilgi verebilirim.",
                "Ürün kataloğumuzda aradığınızı bulmanıza yardımcı olabilirim.",
                "Ürün özellikleri, fiyat ve stok durumu hakkında bilgi verebilirim."
            ],
            'order_status': [
                "Sipariş durumunuzu kontrol edebilirim. Sipariş numaranızı paylaşabilir misiniz?",
                "Siparişinizin hangi aşamada olduğunu öğrenmek için sipariş bilgilerinize ihtiyacım var.",
                "Kargo takip numaranız ile güncel durumu kontrol edebiliriz."
            ],
            'cart_operations': [
                "Sepet işlemlerinizde size yardımcı olabilirim. Ne yapmak istiyorsunuz?",
                "Sepetinize ürün eklemek, çıkarmak veya görüntülemek için buradayım.",
                "Sepet içeriğinizi istediğiniz gibi düzenleyebiliriz."
            ],
            'payment_issues': [
                "Ödeme konusunda yaşadığınız sorunu çözmek için yardımcı olabilirim.",
                "Hangi ödeme yöntemiyle ilgili sorun yaşıyorsunuz?",
                "Güvenli ödeme seçeneklerimiz ve çözümlerimiz hakkında bilgi verebilirim."
            ],
            'return_refund': [
                "İade ve iade süreciniz hakkında size yardımcı olabilirim.",
                "İade koşullarımız ve süreçlerimiz hakkında detaylı bilgi verebilirim.",
                "Hangi ürünü iade etmek istiyorsunuz?"
            ],
            'shipping_info': [
                "Kargo ve teslimat bilgileri hakkında size yardımcı olabilirim.",
                "Teslimat seçeneklerimiz ve süreleri hakkında bilgi verebilirim.",
                "Kargo takibi ve teslimat süreci hakkında her şeyi açıklayabilirim."
            ],
            'goodbye': [
                "Alışveriş yapmak için tekrar bekleriz! İyi günler dileriz.",
                "Teşekkür ederiz! Başka ihtiyacınız olursa buradayız.",
                "Görüşmek üzere! Mutlu alışverişler dileriz."
            ],
            'complaint': [
                "Üzgünüz, yaşadığınız sorunu anlıyoruz. Detayları paylaşır mısınız?",
                "Memnuniyetsizliğinizi anlıyoruz. Sorunu çözmek için elimizden geleni yapacağız.",
                "Geri bildiriminiz bizim için çok değerli. Nasıl yardımcı olabiliriz?"
            ]
        }
    
    def classify_intent(self, text):
        """Intent classification using Hugging Face"""
        try:
            candidate_labels = [
                "selamlama ve karşılama: merhaba, selam, iyi günler, hoşgeldin, nasılsın, hey",
                "ürün arama ve sorgulama: ürün arıyorum, fiyat nedir, özellik nedir, stok var mı, katalog, ürün göster, hangi ürünler var",
                "sipariş takibi ve durum sorgulama: siparişim nerede, ne zaman gelir, kargo takip, sipariş durumu, teslimat tarihi",
                "sepet yönetimi ve işlemleri: sepete ekle, sepetten çıkar, sepetimi göster, sepet toplamı, alışveriş sepeti",
                "ödeme problemleri ve sorunları: ödeme yapamıyorum, kredi kartı çalışmıyor, ödeme hatası, taksit, ödeme yöntemleri",
                "iade ve geri ödeme işlemleri: iade etmek istiyorum, para iadesi, ürün değişimi, iade süreci, geri ödeme", 
                "kargo ve teslimat bilgileri: kargo ücreti, ne kadar sürer, teslimat saatleri, ücretsiz kargo, kargo firması",
                "vedalaşma ve ayrılık: hoşçakal, görüşürüz, teşekkürler, elveda, güle güle, iyi günler",
                "şikayet ve memnuniyetsizlik: şikayetim var, memnun değilim, sorun yaşıyorum, kötü hizmet, problem"
            ]
            
            result = self.classifier(text, candidate_labels)
            
            label_to_intent = {
                "selamlama ve karşılama: merhaba, selam, iyi günler, hoşgeldin, nasılsın, hey": "greeting",
                "ürün arama ve sorgulama: ürün arıyorum, fiyat nedir, özellik nedir, stok var mı, katalog, ürün göster, hangi ürünler var": "product_inquiry",
                "sipariş takibi ve durum sorgulama: siparişim nerede, ne zaman gelir, kargo takip, sipariş durumu, teslimat tarihi": "order_status", 
                "sepet yönetimi ve işlemleri: sepete ekle, sepetten çıkar, sepetimi göster, sepet toplamı, alışveriş sepeti": "cart_operations",
                "ödeme problemleri ve sorunları: ödeme yapamıyorum, kredi kartı çalışmıyor, ödeme hatası, taksit, ödeme yöntemleri": "payment_issues",
                "iade ve geri ödeme işlemleri: iade etmek istiyorum, para iadesi, ürün değişimi, iade süreci, geri ödeme": "return_refund",
                "kargo ve teslimat bilgileri: kargo ücreti, ne kadar sürer, teslimat saatleri, ücretsiz kargo, kargo firması": "shipping_info",
                "vedalaşma ve ayrılık: hoşçakal, görüşürüz, teşekkürler, elveda, güle güle, iyi günler": "goodbye",
                "şikayet ve memnuniyetsizlik: şikayetim var, memnun değilim, sorun yaşıyorum, kötü hizmet, problem": "complaint"
            }
            
            best_label = result['labels'][0]
            confidence = result['scores'][0]
            intent = label_to_intent.get(best_label, 'greeting')
            
            return intent, confidence
            
        except Exception as e:
            print(f"Hugging Face classification hatası: {e}")
            return 'greeting', 0.5
    
    def generate_response(self, intent, user_message):
        """Intent'e göre response üretme"""
        if intent in self.intent_responses:
            responses = self.intent_responses[intent]
            return np.random.choice(responses)
        else:
            return "Üzgünüm, bu konuda size yardımcı olamayabilirim. Başka bir konuda yardımcı olabilir miyim?"
    
    def chat(self, user_message):
        """Ana chat fonksiyonu"""
        intent, confidence = self.classify_intent(user_message)
        response = self.generate_response(intent, user_message)
        
        if confidence < 0.5:
            response = "Anlayamadım, daha net açıklar mısınız? " + response
        
        return {
            'intent': intent,
            'response': response,
            'confidence': confidence
        }
    
    def evaluate_model(self, test_data):
        """Model performansını değerlendirme"""
        true_intents = []
        predicted_intents = []
        
        print("🤖 Hugging Face modeli değerlendiriliyor...")
        
        for idx, row in test_data.iterrows():
            true_intent = row['intent']
            text = row['text']
            
            predicted_intent, _ = self.classify_intent(text)
            
            true_intents.append(true_intent)
            predicted_intents.append(predicted_intent)
            
            if idx % 10 == 0:
                print(f"İşlenen: {idx}/{len(test_data)}")
        
        accuracy = accuracy_score(true_intents, predicted_intents)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_intents, predicted_intents, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predicted_intents,
            'true_labels': true_intents
        }


if __name__ == "__main__":
    try:
        print("🚀 Hugging Face Chatbot Test Başlıyor...")
        chatbot = HuggingFaceChatbot()
        
        test_messages = [
            "Merhaba, nasılsınız?",
            "Bu ürünün fiyatı nedir?",
            "Siparişim nerede?",
            "Sepetime ürün eklemek istiyorum",
            "Teşekkürler, görüşürüz"
        ]
        
        for message in test_messages:
            result = chatbot.chat(message)
            print(f"👤 Kullanıcı: {message}")
            print(f"🤖 Intent: {result['intent']} (Güven: {result['confidence']:.2f})")
            print(f"🤖 Bot: {result['response']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Hata: {e}")
        print("Transformers kütüphanesi yüklenemiyor olabilir.")
        print("Çözüm: pip install transformers torch") 