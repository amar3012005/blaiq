# RAG Service - Development Docker Build - Implementation Checklist

## 📋 Verification Checklist

Use this checklist to verify that the development-optimized Docker build is working correctly.

### ✅ Files Created/Modified

- [x] **Dockerfile** - Modified to use lightweight build strategy
  - Path: `daytona_agent/services/rag/Dockerfile`
  - Changes: 3-stage build with lightweight deps only, setup script embedded
  
- [x] **Docker Compose** - Added dev/prod profiles
  - Path: `docker-compose-leibniz.yml`
  - Changes: `rag` service (dev mode), `rag-prod` service (prod mode), volume mounts
  
- [x] **README** - Added development workflow section
  - Path: `daytona_agent/services/rag/README.md`
  - Changes: "Docker Deployment" section updated with dev vs prod modes
  
- [x] **Quick Start Guide** - Created TL;DR reference
  - Path: `daytona_agent/services/rag/QUICK_START.md`
  - Content: 5-command workflow, common commands, troubleshooting
  
- [x] **Development Guide** - Created comprehensive documentation
  - Path: `daytona_agent/services/rag/DOCKER_DEV_GUIDE.md`
  - Content: Build strategy, 3 operational modes, manual installation, troubleshooting
  
- [x] **PowerShell Helper** - Created automation script
  - Path: `daytona_agent/services/rag/rag-dev.ps1`
  - Content: 15 commands for build/start/setup/run/test/debug
  
- [x] **Bash Helper** - Created automation script
  - Path: `daytona_agent/services/rag/rag-dev.sh`
  - Content: Same 15 commands for Unix/Linux/Mac
  
- [x] **Summary Document** - Created implementation summary
  - Path: `daytona_agent/services/rag/RAG_DEV_DOCKER_SUMMARY.md`
  - Content: Objectives, changes, performance comparison, learnings
  
- [x] **Verification Script** - Created automated test
  - Path: `daytona_agent/services/rag/verify_docker_build.py`
  - Content: Test dev mode, prod mode, health checks

---

## 🧪 Manual Testing Steps

### Test 1: Development Mode (Fast Rebuild)

```powershell
# Step 1: Build image (should take ~2 min)
cd c:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete
.\daytona_agent\services\rag\rag-dev.ps1 build

# Expected: Build completes in ~2 minutes
# ✅ Pass if build time < 3 minutes
# ❌ Fail if build time > 5 minutes

# Step 2: Start container
.\daytona_agent\services\rag\rag-dev.ps1 start

# Expected: Container starts and stays running
# ✅ Pass if `docker ps | grep rag-daytona` shows running container
# ❌ Fail if container exits immediately

# Step 3: Run setup (one-time, ~5 min)
.\daytona_agent\services\rag\rag-dev.ps1 setup

# Expected: Installs torch + builds FAISS index
# ✅ Pass if completes without errors
# ❌ Fail if any step fails

# Step 4: Start service with hot-reload
.\daytona_agent\services\rag\rag-dev.ps1 run

# Expected: Service starts, accessible at http://localhost:8000/health
# ✅ Pass if health endpoint responds
# ❌ Fail if service crashes or endpoint unreachable

# Step 5: Test hot-reload
# (In another terminal) Edit daytona_agent/services/rag/app.py
# Add a comment or whitespace change
# Expected: Service auto-reloads (see logs: "Reloading...")
# ✅ Pass if reload happens automatically
# ❌ Fail if manual restart needed
```

### Test 2: Production Mode (Auto-Setup)

```powershell
# Step 1: Start production service
cd c:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete
.\daytona_agent\services\rag\rag-dev.ps1 prod

# Expected: First start takes ~8 min, subsequent starts ~2s
# ✅ Pass if service becomes healthy within 10 minutes
# ❌ Fail if service doesn't start or crashes

# Step 2: Check health
.\daytona_agent\services\rag\rag-dev.ps1 health

# Expected: Returns 200 OK with status JSON
# ✅ Pass if health check succeeds
# ❌ Fail if 500 error or timeout

# Step 3: Test restart (should be fast)
docker restart rag-daytona-prod

# Expected: Restart completes in <10 seconds
# ✅ Pass if service healthy within 30s
# ❌ Fail if takes >1 minute
```

### Test 3: Helper Scripts

```powershell
# Test each command in rag-dev.ps1
.\daytona_agent\services\rag\rag-dev.ps1 logs        # ✅ Shows logs
.\daytona_agent\services\rag\rag-dev.ps1 shell       # ✅ Opens bash
.\daytona_agent\services\rag\rag-dev.ps1 test        # ✅ Runs pytest
.\daytona_agent\services\rag\rag-dev.ps1 rebuild     # ✅ Rebuilds index
.\daytona_agent\services\rag\rag-dev.ps1 metrics     # ✅ Shows JSON metrics
.\daytona_agent\services\rag\rag-dev.ps1 query "What is Daytona University?"  # ✅ Returns answer
.\daytona_agent\services\rag\rag-dev.ps1 stop        # ✅ Stops container
```

### Test 4: Performance Benchmarks

```powershell
# Measure build time
$start = Get-Date
docker-compose -f docker-compose-leibniz.yml build rag
$buildTime = (Get-Date) - $start
Write-Host "Build time: $($buildTime.TotalSeconds)s"

# Expected: <180 seconds (3 minutes)
# ✅ Pass if <3 min
# ❌ Fail if >5 min

# Measure rebuild time (after code change)
# Edit any .py file
$start = Get-Date
docker-compose -f docker-compose-leibniz.yml build rag
$rebuildTime = (Get-Date) - $start
Write-Host "Rebuild time: $($rebuildTime.TotalSeconds)s"

# Expected: <60 seconds (cached layers)
# ✅ Pass if <2 min
# ❌ Fail if >3 min
```

---

## 🤖 Automated Testing

Run the comprehensive verification script:

```powershell
cd c:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete\daytona_agent\services\rag
python verify_docker_build.py --mode all
```

**Expected Output**:
```
ℹ️  Testing Development Mode Workflow
✅ Build completed in 120.5s
✅ Container running
✅ Setup script exists
✅ Lightweight dependencies installed
✅ Torch not installed (correct - install interactively)
✅ Setup completed in 287.3s
✅ Torch installed: 2.0.1
✅ FAISS index exists
✅ Service started successfully
✅ Health check passed: {'status': 'healthy', ...}
✅ Development Mode Verification: PASSED ✅

ℹ️  Testing Production Mode Workflow
✅ Production service started in 456.2s
✅ Setup marker exists
✅ Health check passed: {'status': 'healthy', ...}
✅ Restart successful (fast)
✅ Production Mode Verification: PASSED ✅

🎉 All verification tests passed!
```

---

## 📊 Success Criteria

### Build Performance
- [ ] Initial build completes in <3 minutes (target: ~2 min)
- [ ] Rebuild after code change completes in <2 minutes (should use cached layers)
- [ ] Image size (base) is <600MB (excludes torch)
- [ ] Container size after setup is <3GB (includes torch)

### Functionality
- [ ] Development mode: Container stays running with `tail -f /dev/null`
- [ ] Setup script: Installs torch + builds index successfully
- [ ] Service startup: Health endpoint responds within 30s
- [ ] Hot-reload: Code changes trigger automatic reload
- [ ] Production mode: Auto-runs setup on first start
- [ ] Production restart: Fast restart (<10s) after initial setup

### Developer Experience
- [ ] Helper scripts work on both PowerShell and Bash
- [ ] All 15 commands execute without errors
- [ ] Documentation is clear and complete
- [ ] Quick Start guide enables 5-command workflow
- [ ] Troubleshooting guide covers common issues

### Production Readiness
- [ ] Production mode completes first start in <10 minutes
- [ ] Health checks pass consistently
- [ ] Metrics endpoint provides performance data
- [ ] Redis caching works (cache hit rate >0% after queries)
- [ ] FAISS index loaded successfully (no errors in logs)

---

## 🐛 Troubleshooting Verification

If any test fails, check:

### Build Issues
```powershell
# Check Docker daemon running
docker info

# Check Dockerfile syntax
docker build -f daytona_agent/services/rag/Dockerfile --no-cache .

# Check base image accessible
docker pull python:3.11-slim
```

### Container Issues
```powershell
# Check container logs
docker logs rag-daytona

# Check container resources
docker stats rag-daytona

# Exec into container
docker exec -it rag-daytona /bin/bash
```

### Setup Issues
```powershell
# Check setup script
docker exec rag-daytona cat /app/setup_heavy_deps.sh

# Run setup manually
docker exec -it rag-daytona /bin/bash
/app/setup_heavy_deps.sh

# Check pip install
docker exec rag-daytona pip list | grep torch
```

### Service Issues
```powershell
# Check service logs
docker exec rag-daytona cat /var/log/uvicorn.log 2>/dev/null || docker logs rag-daytona

# Check port binding
netstat -an | Select-String "8000"

# Test endpoint manually
curl http://localhost:8000/health
```

---

## ✅ Final Sign-Off

After completing all tests above:

- [ ] All manual tests passed
- [ ] Automated verification script passed
- [ ] Performance benchmarks met targets
- [ ] Documentation reviewed and accurate
- [ ] Helper scripts functional
- [ ] Both dev and prod modes working

**Status**: _____________________ (Ready for Production / Needs Fixes)

**Tested By**: ___________________

**Date**: ___________________

**Notes**: 
```
(Add any observations, issues found, or recommendations)
```

---

## 📚 Reference Documentation

- **Quick Start**: `QUICK_START.md` - 5-command workflow
- **Development Guide**: `DOCKER_DEV_GUIDE.md` - Comprehensive documentation
- **Summary**: `RAG_DEV_DOCKER_SUMMARY.md` - Implementation details
- **README**: `README.md` - Full service documentation

---

**Next Steps After Verification**:

1. ✅ Commit changes to repository
2. ✅ Update team documentation
3. ✅ Share Quick Start guide with developers
4. ✅ Add to CI/CD pipeline (use dev profile for faster builds)
5. ✅ Monitor developer feedback and iterate
