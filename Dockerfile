FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights and the NLTK corpora into the image. Without this, every new
# worker downloads ~1.5GB from the Hugging Face hub before it can answer anything.
ENV HF_HOME=/app/.hf_cache
ENV NLTK_DATA=/app/nltk_data
ARG FLORENCE_MODEL_ID=microsoft/Florence-2-large
ENV FLORENCE_MODEL_ID=${FLORENCE_MODEL_ID}
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoProcessor; \
from transformers.dynamic_module_utils import get_imports; \
from unittest.mock import patch; \
import os; \
mid = os.environ['FLORENCE_MODEL_ID']; \
fix = lambda f: [i for i in get_imports(f) if i != 'flash_attn'] if str(f).endswith('modeling_florence2.py') else get_imports(f); \
patch('transformers.dynamic_module_utils.get_imports', fix).start(); \
AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True); \
AutoProcessor.from_pretrained(mid, trust_remote_code=True)" \
    && python -c "\
import nltk; \
[nltk.download(p, download_dir='/app/nltk_data', quiet=True) for p in \
 ['wordnet','omw-1.4','words','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"

# Everything is in the image now: a worker must not depend on the hub to boot.
ENV HF_HUB_OFFLINE=1

COPY app ./app

EXPOSE 8000

# PORT lets a RunPod load-balancing endpoint choose the port (keep PORT_HEALTH equal to it).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
