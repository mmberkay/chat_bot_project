#!/usr/bin/env python3
"""
E-Ticaret Chatbot Model Değerlendirme Scripti
Bu script Gemini ve Hugging Face modellerini karşılaştırır.
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.gemini_model import GeminiChatbot
from models.huggingface_model import HuggingFaceChatbot

def load_data(file_path="data/ecommerce_dataset.csv"):
    """Veri setini yükle"""
    try:
        data = pd.read_csv(file_path)
        print(f"✅ Veri seti yüklendi: {len(data)} örnek")
        return data
    except FileNotFoundError:
        print(f"❌ Veri seti bulunamadı: {file_path}")
        return None

def evaluate_models(data, test_size=0.2, sample_size=50):
    """Modelleri değerlendir"""
    print("🔄 Veri seti train/test olarak bölünüyor...")
    
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=42, 
        stratify=data['intent']
    )
    
    
    test_sample = test_data.head(sample_size) if len(test_data) > sample_size else test_data
    print(f"📊 Test seti: {len(test_sample)} örnek")
    
    results = {}
    models = {}
    
    try:
        print("🤖 Gemini modeli başlatılıyor...")
        models['Gemini'] = GeminiChatbot()
        print("✅ Gemini modeli hazır")
    except Exception as e:
        print(f"❌ Gemini model hatası: {e}")
    
    try:
        print("🤗 Hugging Face modeli başlatılıyor...")
        models['Hugging Face'] = HuggingFaceChatbot()
        print("✅ Hugging Face modeli hazır")
    except Exception as e:
        print(f"❌ Hugging Face model hatası: {e}")
    
    if not models:
        print("❌ Hiçbir model başlatılamadı. API anahtarlarını kontrol edin.")
        return None
    
    for model_name, model in models.items():
        print(f"\n🔬 {model_name} modeli değerlendiriliyor...")
        result = model.evaluate_model(test_sample)
        results[model_name] = result
        
        print(f"📈 {model_name} Sonuçları:")
        print(f"   Accuracy: {result['accuracy']:.3f}")
        print(f"   Precision: {result['precision']:.3f}")
        print(f"   Recall: {result['recall']:.3f}")
        print(f"   F1 Score: {result['f1_score']:.3f}")
    
    return results, test_sample

def create_comparison_report(results, test_data):
    """Karşılaştırma raporu oluştur"""
    print("\n" + "="*60)
    print("📊 MODEL KARŞILAŞTIRMA RAPORU")
    print("="*60)
    
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'Precision': [results[model]['precision'] for model in results.keys()],
        'Recall': [results[model]['recall'] for model in results.keys()],
        'F1 Score': [results[model]['f1_score'] for model in results.keys()]
    })
    
    print("\n🏆 PERFORMANS SONUÇLARI:")
    print(metrics_df.to_string(index=False, float_format='%.3f'))
    
    best_model = metrics_df.loc[metrics_df['F1 Score'].idxmax(), 'Model']
    best_f1 = metrics_df.loc[metrics_df['F1 Score'].idxmax(), 'F1 Score']
    
    print(f"\n🥇 EN İYİ MODEL: {best_model} (F1 Score: {best_f1:.3f})")
    
    print(f"\n🎯 INTENT BAZINDA ANALIZ:")
    unique_intents = test_data['intent'].unique()
    print(f"   Toplam Intent: {len(unique_intents)}")
    print(f"   Intent'ler: {', '.join(unique_intents)}")
    
    return metrics_df

def plot_results(results):
    """Sonuçları görselleştir"""
    try:
        import matplotlib.pyplot as plt
        
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 Grafik kaydedildi: model_comparison.png")
        
    except ImportError:
        print("⚠️ Matplotlib yüklü değil, grafik oluşturulamadı")

def main():
    """Ana fonksiyon"""
    print("🚀 E-Ticaret Chatbot Model Değerlendirmesi Başlıyor...")
    print("="*60)
    
    data = load_data()
    if data is None:
        return
    
    print(f"📋 Veri Seti Bilgileri:")
    print(f"   Toplam Örnek: {len(data)}")
    print(f"   Intent Sayısı: {data['intent'].nunique()}")
    print(f"   Intent Dağılımı:")
    for intent, count in data['intent'].value_counts().items():
        print(f"      {intent}: {count}")
    
    results = evaluate_models(data, sample_size=30) 
    
    if results:
        results_dict, test_sample = results
        
        metrics_df = create_comparison_report(results_dict, test_sample)
        
        plot_results(results_dict)
        
        metrics_df.to_csv('model_comparison_results.csv', index=False)
        print("💾 Sonuçlar kaydedildi: model_comparison_results.csv")
        
        print("\n✅ Değerlendirme tamamlandı!")
    else:
        print("❌ Model değerlendirmesi yapılamadı")

if __name__ == "__main__":
    main() 