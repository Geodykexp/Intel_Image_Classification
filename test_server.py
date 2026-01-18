import requests

url = "http://127.0.0.1:8080/predict"
path = "C:\\Users\\Ikenna George\\Backup\\Documents\\DATA SCIENCE & PROGRAMMING\\DATA SCIENCE PROJECTS\\Intel_Image_Classification\\seg_test\\forest\\20056.jpg"

try:
    with open(path, 'rb') as file:
        files = {'file': file}
        response = requests.post(url, files=files)
        response.raise_for_status()
        data = response.content
        print(data)
except requests.exceptions.RequestException as e:
    print(f"Error during request: {e}")
except FileNotFoundError:
    print(f"Image file not found: {path}")
  
    