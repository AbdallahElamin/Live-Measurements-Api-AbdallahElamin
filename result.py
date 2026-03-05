import requests

url = "http://localhost:8001/upload_images"
files = {
    'front': open('front.jpeg', 'rb'),
    'left_side': open('left_side.jpeg', 'rb')
}
data = {'height_cm': '170'}

response = requests.post(url, files=files, data=data)
print(f"Status Code: {response.status_code}")
try:
    print(response.json())
except Exception as e:
    print(f"Error parsing JSON: {e}")
    print(f"Response text: {response.text}")

