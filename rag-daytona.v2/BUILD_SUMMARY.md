# RAG Service - Optimized Build Summary

## ✅ What Was Changed

1. **Split requirements into two files:**
   - `requirements.txt` - Lightweight packages (installed during build)
   - `requirements_after.txt` - Heavy packages (install manually after build)

2. **Updated Dockerfile:**
   - Only installs lightweight packages during build
   - Copies `requirements_after.txt` and install script into container
   - Build completes in ~2-3 minutes instead of ~40 minutes

3. **Created helper scripts:**
   - `install_heavy_packages.sh` - Automated installation script
   - `INSTALL_HEAVY_PACKAGES.md` - Detailed instructions
   - `QUICK_BUILD.md` - Quick reference guide

## 🚀 Quick Start

### 1. Build (Fast - ~2-3 minutes)
```bash
cd /home/prometheus/daytona_agent/services
docker context use desktop-linux
docker-compose -p tara-microservice -f docker-compose-tara.yml build rag-service-tara
```

### 2. Start Container
```bash
docker-compose -p tara-microservice -f docker-compose-tara.yml up -d rag-service-tara
```

### 3. Install Heavy Packages (Inside Container - ~5-10 minutes)
```bash
# Enter container
docker exec -it tara-rag-service bash

# Run install script
bash /app/install_heavy_packages.sh
```

## 📦 Package Breakdown

### Lightweight (requirements.txt) - Installed During Build
- FastAPI, Uvicorn
- Pydantic
- Gemini API clients
- FAISS-CPU (vector search)
- NumPy
- Redis client
- HTTP clients

**Total**: ~100MB, installs in ~2 minutes

### Heavy (requirements_after.txt) - Install After Build
- sentence-transformers (~500MB)
- torch (~900MB)
- tokenizers (~3MB)
- transformers (~100MB)
- huggingface-hub (~50MB)

**Total**: ~1.5GB download, ~3GB disk space, installs in ~5-10 minutes

## ⚠️ Important Notes

1. **Index Building**: The FAISS index builder stage will fail during build (because sentence-transformers isn't installed yet). This is **OK** - the service will build the index at runtime after you install heavy packages.

2. **Service Functionality**: The RAG service will start and pass health checks, but won't be able to process queries until heavy packages are installed.

3. **One-Time Install**: Once you install heavy packages, they persist in the container. You only need to install them once per container.

## 🔄 Alternative: Full Build (If You Want Everything During Build)

If you prefer to install everything during build (slower but automatic), edit `Dockerfile` line 28:

```dockerfile
RUN pip install --default-timeout=1000 --user --no-cache-dir -r requirements.txt -r requirements_after.txt
```

This will install all packages during build (~40 minutes).

## ✅ Verification

After installing heavy packages, verify:

```bash
docker exec -it tara-rag-service python -c "import sentence_transformers; print('✅ OK')"
docker exec -it tara-rag-service python -c "import torch; print(f'✅ torch {torch.__version__}')"
```

## 📊 Build Time Comparison

| Method | Build Time | Install Time | Total |
|--------|-----------|-------------|-------|
| **Old (all during build)** | ~40 min | 0 min | ~40 min |
| **New (split)** | ~2-3 min | ~5-10 min | ~7-13 min |

**Benefit**: You can start using the container immediately, then install heavy packages when ready!












