import base64
import requests

with open("/home/rin/.gemini/antigravity/brain/d29cfc15-d5c4-4412-ac6b-cd2ece6338d3/system_architecture.md", "r") as f:
    content = f.read()

# Extract mermaid block
start = content.find("```mermaid") + 10
end = content.find("```", start)
mermaid_code = content[start:end].strip()

graphbytes = mermaid_code.encode("utf-8")
base64_bytes = base64.b64encode(graphbytes)
base64_string = base64_bytes.decode("utf-8")

url = "https://mermaid.ink/img/" + base64_string
response = requests.get(url)

if response.status_code == 200:
    with open("architecture.png", "wb") as f:
        f.write(response.content)
    print("Xong!")
else:
    print(f"Lỗi: {response.status_code}")
