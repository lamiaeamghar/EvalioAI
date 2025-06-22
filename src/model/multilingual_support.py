# multilingual_support.py
from transformers import pipeline
from typing import Dict, Any
import warnings

class MultilingualRealEstateAssistant:
    def __init__(self):
        self.languages = {
            'en': 'english',
            'fr': 'french',
            'ar': 'arabic'
        }
        self.translator = None
        self.nlp = None
        self.use_simple_fallback = False  

        try:
            # Lightweight translation model
            self.translator = pipeline('translation', 
                                       model='Helsinki-NLP/opus-mt-mul-en',
                                       device='cpu')
            
            # Small multilingual NLP model
            self.nlp = pipeline('zero-shot-classification',
                                model='joeddav/xlm-roberta-large-xnli',
                                device='cpu')
        except Exception as e:
            warnings.warn(f"Could not load models: {str(e)}")
            self.use_simple_fallback = True


    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        if self.use_simple_fallback:
            # Basic keyword detection
            if any(char in text for char in 'اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي'):
                return 'ar'
            elif any(word in text.lower() for word in ['le', 'la', 'bonjour']):
                return 'fr'
            return 'en'
        
        # Use model if available
        result = self.nlp(text, candidate_labels=list(self.languages.values()))
        # Find label with highest score
        best_label = result['labels'][0]
        # Map back from language name to language code
        for code, lang_name in self.languages.items():
            if lang_name == best_label:
                return code
        return 'en'  # fallback

    def translate_to_english(self, text: str) -> str:
        """Convert input to English for processing"""
        if self.use_simple_fallback:
            return text  # Skip translation in fallback mode
            
        lang = self.detect_language(text)
        if lang != 'en':
            translated = self.translator(text, src_lang=lang, tgt_lang='en')
            return translated[0]['translation_text']
        return text

    def generate_response(self, text: str, context: Dict[str, Any]) -> str:
        """Generate multilingual response based on context"""
        lang = self.detect_language(text)
        
        # Your real estate logic would use the context
        response_content = {
            'price': context.get('predicted_price', 0),
            'features': context.get('important_features', [])
        }
        
        # Simple template responses (expand with your actual logic)
        responses = {
            'en': f"The predicted price is {response_content['price']:,} MAD. Key features: {', '.join(response_content['features'][:3])}",
            'fr': f"Le prix estimé est {response_content['price']:,} MAD. Caractéristiques: {', '.join(response_content['features'][:3])}",
            'ar': f"السعر المتوقع هو {response_content['price']:,} درهم. الميزات: {', '.join(response_content['features'][:3])}"
        }
        
        return responses.get(lang, responses['en'])

# Singleton instance for easy import
assistant = MultilingualRealEstateAssistant()
