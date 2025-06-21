import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from dotenv import load_dotenv
import time

load_dotenv()

class GeminiChatbot:
    def __init__(self, api_key=None):
        """Gemini Chatbot sınıfı"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key gerekli!")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.intent_responses = {
            'greeting': [
                "Merhaba! E-ticaret platformumuza hoş geldiniz. Size nasıl yardımcı olabilirim?",
                "Selam! Alışveriş yapmak için hangi ürünleri arıyorsunuz?",
                "İyi günler! Müşteri hizmetlerimizde size yardımcı olmaktan memnuniyet duyarım."
            ],
            'product_inquiry': [
                "Hangi ürün hakkında detaylı bilgi almak istiyorsunuz?",
                "Ürün katalogumuzdan size en uygun seçenekleri bulabilirim.",
                "Ürün özellikleri, fiyatlar ve stok durumu hakkında bilgi alabilirim."
            ],
            'order_status': [
                "Siparişinizin durumunu kontrol etmek için sipariş numaranıza ihtiyacım var.",
                "Sipariş takibi için gerekli bilgileri paylaşabilir misiniz?",
                "Kargo durumunuzu ve teslimat bilgilerinizi sorgulayabilirim."
            ],
            'cart_operations': [
                "Sepet işlemlerinizde size yardımcı olabilirim. Ne yapmak istiyorsunuz?",
                "Sepetinizi yönetmek için buradayım. Ürün eklemek veya çıkarmak istiyor musunuz?",
                "Sepet içeriğinizi istediğiniz şekilde düzenleyebiliriz."
            ],
            'payment_issues': [
                "Ödeme sürecinde yaşadığınız sorun nedir? Size yardımcı olabilirim.",
                "Hangi ödeme yöntemiyle ilgili problem yaşıyorsunuz?",
                "Güvenli ödeme alternatifleri ve çözüm önerileri sunabilirim."
            ],
            'return_refund': [
                "İade işleminizde size yardımcı olabilirim. Hangi ürünü iade etmek istiyorsunuz?",
                "İade koşulları ve süreçleri hakkında size bilgi verebilirim.",
                "İade başvurunuzu hızlıca işleme alabiliriz."
            ],
            'shipping_info': [
                "Kargo ve teslimat hakkında size bilgi verebilirim.",
                "Teslimat seçenekleri ve süreleri konusunda yardımcı olabilirim.",
                "Kargo takip ve teslimat detayları hakkında her türlü sorunuzu yanıtlayabilirim."
            ],
            'goodbye': [
                "Alışveriş yapmak için tekrar görüşmek üzere! İyi günler dileriz.",
                "Teşekkür ederiz! Her zaman buradayız, iyi alışverişler!",
                "Görüşürüz! Başka ihtiyacınız olduğunda bizi unutmayın."
            ],
            'complaint': [
                "Yaşadığınız sorunu anlıyoruz. Lütfen detayları paylaşın, çözüm bulalım.",
                "Memnuniyetsizliğinizi gidermek için elimizden geleni yapacağız.",
                "Geri bildiriminiz çok önemli. Sorunu nasıl çözebileceğimizi anlatın."
            ]
        }
    
    def classify_intent(self, text):
        """Intent classification using Gemini with Few-Shot Learning"""
        prompt = f"""
Sen uzman bir e-ticaret müşteri hizmetleri chatbot'usun. Aşağıdaki örneklere bakarak, kullanıcı mesajlarını doğru kategorilere ayır:

📚 ÖRNEKLER:
"Merhaba" → greeting
"İyi günler" → greeting  
"Selam" → greeting

"Bu ürünün fiyatı nedir?" → product_inquiry
"Ürün arıyorum" → product_inquiry
"Stokta var mı?" → product_inquiry

"Siparişim nerede?" → order_status
"Kargo takip numarası" → order_status
"Ne zaman gelecek?" → order_status

"Sepete ekle" → cart_operations
"Sepetimi göster" → cart_operations
"Sepet toplamı" → cart_operations

"Ödeme yapamıyorum" → payment_issues
"Kredi kartım çalışmıyor" → payment_issues
"Taksit seçenekleri" → payment_issues

"İade etmek istiyorum" → return_refund  
"Para iadesi" → return_refund
"Ürün değişimi" → return_refund

"Kargo ne kadar sürer?" → shipping_info
"Teslimat saatleri" → shipping_info
"Ücretsiz kargo" → shipping_info

"Hoşçakal" → goodbye
"Görüşürüz" → goodbye
"Teşekkürler" → goodbye

"Şikayetim var" → complaint
"Memnun değilim" → complaint
"Sorun yaşıyorum" → complaint

🎯 ŞİMDİ BU METNİ SINIFLANDIR:
Kullanıcı Mesajı: "{text}"

📋 CEVAP FORMATI (TAM OLARAK ŞU ŞEKİLDE):
Kategori: [kategori_adı]
Güven: [0.0-1.0 arası sayı]
Açıklama: [kısa neden]

ÖRNEK CEVAP:
Kategori: product_inquiry
Güven: 0.95
Açıklama: Kullanıcı ürün araması yapıyor
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            intent, confidence = self._parse_gemini_response(response_text)
            
            time.sleep(0.1)
            
            return intent, confidence
            
        except Exception as e:
            print(f"Gemini API Hatası: {e}")
            return "greeting", 0.5 

    def _parse_gemini_response(self, response_text):
        """Gemini'nin cevabını parse etme"""
        try:
            lines = response_text.split('\n')
            intent = "greeting"
            confidence = 0.5
            
            for line in lines:
                if "Kategori:" in line:
                    intent = line.split("Kategori:")[1].strip().lower()
                elif "Güven:" in line:
                    try:
                        confidence_str = line.split("Güven:")[1].strip()
                        confidence = float(confidence_str)
                    except:
                        confidence = 0.7  
            
            valid_intents = [
                'greeting', 'product_inquiry', 'order_status', 
                'cart_operations', 'payment_issues', 'return_refund',
                'shipping_info', 'goodbye', 'complaint'
            ]
            
            if intent not in valid_intents:
                intent = "greeting"
                confidence = 0.3  
            
            return intent, confidence
        
        except Exception as e:
            print(f"Response parsing hatası: {e}")
            return "greeting", 0.5
    
    def generate_response(self, intent, user_message):
        """Intent'e göre response üretme"""
        if intent in self.intent_responses:
            responses = self.intent_responses[intent]
            return np.random.choice(responses)
        else:
            return "Bu konuda size yardımcı olmakta güçlük çekiyorum. Başka nasıl yardımcı olabilirim?"
    
    def chat(self, user_message):
        """Ana chat fonksiyonu"""
        intent, confidence = self.classify_intent(user_message)
        response = self.generate_response(intent, user_message)
        
        if confidence < 0.6:
            response = "Tam olarak anlayamadım. Lütfen daha açık bir şekilde söyler misiniz? " + response
        
        return {
            'intent': intent,
            'response': response,
            'confidence': confidence
        }
    
    def evaluate_model(self, test_data):
        """Model performansını değerlendirme"""
        true_intents = []
        predicted_intents = []
        
        print("Gemini modeli değerlendiriliyor...")
        
        for idx, row in test_data.iterrows():
            true_intent = row['intent']
            text = row['text']
            
            predicted_intent, _ = self.classify_intent(text)
            
            true_intents.append(true_intent)
            predicted_intents.append(predicted_intent)
            
            if idx % 10 == 0:
                print(f"İşlenen: {idx}/{len(test_data)}")
            
            time.sleep(0.2)
        
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
        chatbot = GeminiChatbot()
        
        test_messages = [
            "Merhaba, nasılsınız?",
            "Bu ürünün fiyatı nedir?", 
            "Siparişim nerede?",
            "Sepetime ürün eklemek istiyorum"
        ]
        
        for message in test_messages:
            result = chatbot.chat(message)
            print(f"Kullanıcı: {message}")
            print(f"Intent: {result['intent']}")
            print(f"Bot: {result['response']}")
            print("-" * 50)
            
    except ValueError as e:
        print(f"Hata: {e}")
        print("GEMINI_API_KEY environment variable'ını ayarlayın.") 