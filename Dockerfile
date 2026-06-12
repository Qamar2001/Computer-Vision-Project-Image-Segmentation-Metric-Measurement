# Use official PyTorch base image with GPU/CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Tesla-V100;Turing;Ampere" 

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpng-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Detectron2 pre-built binary matching CUDA 12.1 and PyTorch 2.1.x
RUN pip install --no-cache-dir detectron2 -f https://dl.fbaipubliccloud.com/detectron2/wheels/cu121/torch2.1/index.html

# Copy requirements file first to leverage docker caching
COPY requirements.txt .

# Install python dependencies (filtering out torch/torchvision as they are already in the base image)
RUN grep -v -E "torch|torchvision" requirements.txt > req_filtered.txt && \
    pip install --no-cache-dir -r req_filtered.txt && \
    rm req_filtered.txt

# Copy the rest of the application code
COPY . .

# Set entry point
CMD ["python", "models/train.py"]
