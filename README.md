# Intel Image Classification – End-to-End Project

This project involves the use of convulated neural network to classify images from the Intel Image Classification dataset. The project includes a Jupyter notebook for exploratory data analysis, a training script for model development, and a FastAPI inference service for real-time predictions. The project is designed to be production-ready and includes containerization with Docker and serverless inference with AWS Lambda. The repository also includes a deployment guide for AWS, dataset samples.

## Repository Structure
Production-grade image classification system built on the Intel Image Classification dataset. This repository includes:
- Deep learning model development using Tensorflow and Keras in a Jupyter Notebook and a training script
- FastAPI inference service for real-time predictions
- Local test client for quick verification
- Containerization with Docker
- Serverless inference with AWS Lambda
- Deployment guidance for AWS (ECR, ECS/Fargate, and Lambda)


## Repository Structure

- intel-image-classification.ipynb – model prototyping, and training notebook
- train_model.py – Scripted training pipeline (headless, repeatable training)
- xception_v2_10_0.917.keras – Trained model
- main.py – FastAPI app exposing a REST endpoint for inference
- test_server.py – Local test client that sends an image to the FastAPI server
- Dockerfile – Container definition for the FastAPI service
- Lambda_function.py – AWS Lambda handler for serverless inference
- pyproject.toml / uv.lock – Python dependency management (PEP 621 + uv/pip-tools style lock)
- AWS_DEPLOYMENT_GUIDE.md – Additional AWS deployment tips
- seg_train/, seg_test/, seg_pred/ – Dataset samples and/or prediction images


## Environment and Dependencies

This project targets Python 3.12 (adjust if your environment differs).

Core libraries typically used (check pyproject.toml for authoritative versions):
- TensorFlow / Keras (Xception backbone)
- FastAPI, Uvicorn
- Pillow, NumPy
- boto3 (for AWS), mangum (if using API Gateway proxy for Lambda)
- pydantic

Install dependencies using uv or pip:

- With uv
  - pip install uv
  - uv pip install -r pyproject.toml

- With pip
  - python -m venv .venv
  - .venv\\Scripts\\activate (Windows)
  - pip install -e . or pip install -r requirements.txt (if you export one)


## 1) Model Development (Notebook and Script)

### Notebook: intel-image-classification.ipynb

The notebook demonstrates the full workflow:
- Data ingestion and preprocessing from seg_train/ and seg_test/
- Model architecture (Xception-based classifier)
- Training schedule, callbacks (e.g., EarlyStopping, ModelCheckpoint)
- Evaluation metrics (accuracy, confusion matrix)
- Model export to .keras or .h5

To run:
- Launch Jupyter: jupyter lab or jupyter notebook
- Open intel-image-classification.ipynb and run cells sequentially

Output artifacts: xception_v*.keras or xception_v*_*.h5 saved to project root.

### Scripted Training: train_model.py

Provides a repeatable, CLI-driven training pipeline mirroring the notebook logic. Typical flow:
- Load and augment dataset from seg_train/ and seg_test/
- Build Xception-based model
- Train and save best weights to xception_v*.keras

Example usage:
- python train_model.py --epochs 25 --batch-size 32 --img-size 150, 150 --model-out xception_v2_24_0.915.keras

Common flags (inspect train_model.py for exact names):
- --epochs E
- --batch-size N
- --img-size W H
- --learning-rate LR
- --model-out PATH
- --data-dir PATH


## 2) FastAPI Inference Service (main.py)

The FastAPI app loads the trained model at startup and exposes a REST endpoint for prediction.

Typical design:
- Startup event: load model artifact (e.g., xception_v2_24_0.915.keras)
- Endpoint: POST /predict
  - Accepts multipart/form-data with an image file (field name: file)
  - Preprocesses image (resize, scale)
  - Runs model.predict
  - Maps output to class label and probability

Local run:
- uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Open docs:
- http://127.0.0.1:8000/docs

Environment variables (suggested):
- MODEL_PATH: path to model file (e.g., xception_v2_24_0.915.keras)
- CLASSES: optional JSON or comma-separated class names


## 3) Local Test Client (test_server.py)

Utility script to test the FastAPI endpoint with a local image.

Typical usage:
- python test_server.py --image seg_pred/10336.jpg --url http://127.0.0.1:8000/predict

Expected output:
- JSON with predicted class and probability


## 4) Containerization (Dockerfile)

A minimal container to run the FastAPI app with Uvicorn.

Typical Dockerfile outline:
- Base: python:3.10-slim (or NVIDIA CUDA base if GPU is required)
- Install system packages (e.g., libgl1 for OpenCV/Pillow)
- Copy pyproject.toml/uv.lock and install deps
- Copy source (main.py, model artifact)
- Expose port 8000 and start uvicorn main:app

Build image:
- docker build -t intel-image-classifier:latest .

Run locally:
- docker run --rm -p 9696:8080 -e MODEL_PATH=/app/xception_v2_10_0.917.keras intel-image-classifier:latest

Test:
- curl -X POST http://0.0.0.0:8080/predict -F "file=@seg_pred/10336.jpg"


## 5) AWS Lambda (Lambda_function.py)

Serverless inference handler to classify a single image per invocation. Two common integration patterns:
- Direct payload: base64-encoded image in the event
- S3 trigger: event with bucket/key; function fetches the object, runs inference, and returns prediction

Typical handler signature:
- def lambda_handler(event, context):
  - Load model (use global singleton for cold-start reuse)
  - Decode input image (base64 or S3)
  - Preprocess and predict
  - Return class and probability

Recommended packaging:
- Use a Lambda Layer for heavy dependencies (TensorFlow, Pillow, NumPy), or
- Container image for Lambda: build with Docker and push to ECR, set as Lambda image

Environment variables:
- MODEL_PATH: "/var/taskxception_v2_10_0.917.keras" (in package) or "/opt/model/xception_v2_10_0.917.keras" (if Layer)
- CLASSES: class names
- S3_BUCKET: if loading model or images from S3


## 6) AWS Deployment Options

### A) AWS Lambda (Container Image) Using my already stored repo

1. Authenticate Docker to ECR
- aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

3. Tag and push
- docker tag intel-image-classifier:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/my_lambda_repo:latest
- docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/my_lambda_repo:latest

4. Create Lambda from image
- Set handler to default for container (CMD in Dockerfile should run a web adapter like awslambdaric if using handler, or expose the function for API Gateway if using Lambda Web Adapter)
- Configure memory (512–2048MB) and timeout (10–30s)
- Add environment variables (MODEL_PATH, CLASSES)

5. Invoke via API Gateway or Lambda URL
- For API Gateway: integrate with Lambda proxy or use AWS Lambda Web Adapter

Notes:
- For pure handler usage (Lambda_function.py), build a separate image that uses the AWS Lambda Python base image and sets CMD to Lambda handler runtime (awslambdaric) with handler "Lambda_function.lambda_handler".


### B) AWS ECS on Fargate (FastAPI Service)

1. Push the Docker image as above
2. Create a Task Definition
- Container: image=<ECR URI>, portMappings=[{"containerPort": 8000}], env=[MODEL_PATH]
- CPU/Memory per your needs
3. Create a Service on Fargate
- Attach to a public subnet with a security group allowing TCP/80 or TCP/8000
- Place behind an Application Load Balancer (ALB)
4. Test the public ALB DNS name
- curl -X POST http://<ALB_DNS>/predict -F "file=@local.jpg"


### C) EC2 or Lightsail (Simple VM)

- Provision an instance with Docker installed
- Pull and run the image; attach an Elastic IP or use the public IP


## 7) Data and Classes

The Intel Image Classification dataset contains 6 scene classes:
- buildings, forest, glacier, mountain, sea, street

Preprocessing used in this project (typical):
- Resize to 224x224
- Scale to [0,1]
- Optional: center-crop or aspect-ratio preserving resize

Ensure the model and services share the same preprocessing pipeline to avoid train/serve skew.


## 8) Inference API Contract

- Endpoint: POST /predict
- Request: multipart/form-data
  - Key: file, Value: binary image
- Response (application/json):
  - {
    "class": "mountain",
    "probability": 0.917,
    "top_k": [
      {"class": "mountain", "prob": 0.917},
      {"class": "glacier", "prob": 0.061},
      {"class": "forest", "prob": 0.012}
    ]
  }

Error handling:
- 400 if file is missing or invalid image
- 500 for internal server errors; logs include stacktrace


## 9) Local Development Workflow

- Train or pick a model artifact: xception_v2_24_0.915.keras
- Start API locally: uvicorn main:app --reload
- Test with test_server.py against seg_pred/ images
- Build and run Docker image
- Push to ECR and deploy on ECS or Lambda


## 10) Operational Notes

Performance:
- Warm model at startup to avoid first-request latency
- Consider ONNX or TF SavedModel + TF-Serving for higher throughput

Observability:
- Add request/response logging (without storing raw images)
- Expose /health for readiness/liveness probes

Security:
- Validate content-type and image size
- Rate-limit public endpoints behind an ALB/WAF or API Gateway

Scalability:
- ECS/Fargate: set auto-scaling policies on CPU/Memory
- Lambda: provisioned concurrency for steady low-latency

Model management:
- Version artifacts (e.g., xception_v2_10_0.917.keras)
- Store models in S3 and configure the service to pull at boot if needed


## 11) Quick Commands

- Start API locally:
  - uvicorn main:app --host 0.0.0.0 --port 8000 --reload

- Test locally:
  - python test_server.py --image seg_pred/10336.jpg --url http://0.0.0.0:8080/predict

- Docker build + run:
  - docker build -t intel-image-classifier:latest .
  - docker run --rm -p 8000:8000 -e MODEL_PATH=/app/xception_v2_24_0.915.keras intel-image-classifier:latest

- cURL prediction:
  - curl -X POST http://127.0.0.1:8000/predict -F "file=@seg_pred/10336.jpg"


## 12) Troubleshooting

- Model file not found
  - Ensure MODEL_PATH points to an existing file inside the container/host
- TensorFlow import error in Lambda
  - Use a Lambda container image or a Layer compiled for Amazon Linux 2
- 413 Request Entity Too Large behind ALB/API Gateway
  - Increase body size limit (ALB target group/NGINX/API Gateway binary media settings)
- Mismatched classes
  - Ensure the CLASSES env var matches training order


## License

This project is for educational and demonstration purposes. Verify dataset and dependency licenses before commercial use.
