from openai import OpenAI
import base64
client = OpenAI(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Путь к локальному изображению
image_path = "/home/robot/every_embodied_navigation/dog.webp"
# image_path = "/home/robot/every_embodied_navigation/dog.png"

# Кодируем в Base64
with open(image_path, "rb") as f:
    base64_data = base64.b64encode(f.read()).decode("utf-8")

# Формируем запрос в формате, который поддерживает Tongyi Qianwen (通义千问)
base64_url = f"data:image/webp;base64,{base64_data}"
# base64_url = f"data:image/png;base64,{base64_data}"

# Отправляем запрос
completion = client.chat.completions.create(
    model="qwen3-vl-8b-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": base64_url}
                },
                {"type": "text", "text": "Какая сцена изображена на картинке?"}
            ]
        }
    ],
)

print(completion.choices[0].message.content)