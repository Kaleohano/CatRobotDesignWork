from flask import Flask, request, jsonify
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import io

app = Flask(__name__)

# 使用对应的模型
model_name = "semihdervis/cat-emotion-classifier"
model = ViTForImageClassification.from_pretrained(model_name)
image_processor = ViTImageProcessor.from_pretrained(model_name)

@app.route('/')
def home():
    return "Welcome to the Cat Emotion Classifier API. Use the '/classify' endpoint to classify cat emotions.", 200

@app.route('/classify', methods=['POST'])
def classify_emotion():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        
        # 打印图像信息以便调试
        print(f"Original Image Size: {image.size}, Mode: {image.mode}")

        # 确保图像是 RGB 格式
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image to (224, 224)
        image = image.resize((224, 224))  # Resize to expected size

        # 处理图像输入并打印输入维度
        inputs = image_processor(images=image, return_tensors="pt")
        print(f"Processed Input Shape: {inputs['pixel_values'].shape}")

        # 使用模型进行分类
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()
        
        # 获取情感标签
        label = model.config.id2label[predicted_class_id]
        
        return jsonify({"emotion": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # 返回具体错误信息

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 返回一个204无内容状态

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # 启动 Flask 应用于5001端口