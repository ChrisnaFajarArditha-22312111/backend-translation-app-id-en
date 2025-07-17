import os
import sys
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import logging
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable untuk menyimpan model pipeline
translator = None

# Nama model Hugging Face milikmu
HUGGINGFACE_MODEL_ID = "miawmiaw23/models-transformers-translation"

def load_translation_model():
    """
    Memuat model terjemahan dari Hugging Face Hub.
    """
    global translator
    try:
        logger.info(f"Memuat model dari Hugging Face Hub: {HUGGINGFACE_MODEL_ID}")
        
        tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(HUGGINGFACE_MODEL_ID)
        
        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer
        )

        logger.info("Model terjemahan berhasil dimuat.")
        return True

    except Exception as e:
        logger.error(f"Error saat memuat model: {str(e)}")
        return False

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """
    Endpoint untuk menerjemahkan teks dari bahasa Indonesia ke Inggris.
    """
    global translator
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Request harus berformat JSON"}), 400

        data = request.get_json()
        if 'text' not in data:
            return jsonify({"success": False, "error": "Field 'text' diperlukan"}), 400

        text_to_translate = data['text'].strip()
        if not text_to_translate:
            return jsonify({"success": False, "error": "Teks tidak boleh kosong"}), 400

        if translator is None:
            logger.info("Model belum dimuat, mencoba memuat sekarang...")
            if not load_translation_model():
                return jsonify({"success": False, "error": "Model tidak tersedia"}), 500

        logger.info(f"Menerjemahkan teks: {text_to_translate[:50]}...")
        result = translator(text_to_translate)
        translated_text = result[0]['translation_text']
        logger.info(f"Hasil terjemahan: {translated_text[:50]}...")

        return jsonify({
            "success": True,
            "original_text": text_to_translate,
            "translated_text": translated_text,
            "source_language": "id",
            "target_language": "en"
        })

    except Exception as e:
        logger.error(f"Terjadi kesalahan saat menerjemahkan: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    global translator
    model_status = "loaded" if translator is not None else "not_loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "service": "Translation API"
    })

@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    global translator
    try:
        translator = None
        success = load_translation_model()

        if success:
            return jsonify({"success": True, "message": "Model berhasil dimuat ulang"})
        else:
            return jsonify({"success": False, "error": "Gagal memuat ulang model"}), 500

    except Exception as e:
        logger.error(f"Error saat reload model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if app.static_folder is None:
        return "Static folder not configured", 404

    full_path = os.path.join(app.static_folder, path)
    if path != "" and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    elif os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    else:
        return "index.html not found", 404

if __name__ == '__main__':
    print("Memuat model terjemahan...")
    load_translation_model()
    print("Aplikasi siap digunakan!")
    app.run(host='0.0.0.0', port=5000, debug=True)
