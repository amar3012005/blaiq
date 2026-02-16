import sys
import os
import asyncio
import logging
import requests
import random

# Add current dir to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_embeddings import OptimizedEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "tara_case_memory"
MODEL_PATH = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"

# Comprehensive Daytona support cases database
CASES = [
    # SDK Installation & Setup (10 cases)
    {"id": 1, "issue": "How do I install the Daytona Python SDK?", "solution": "Run pip install daytona-sdk. Import with: from daytona import Daytona, CreateSandboxParams", "issue_type": "sdk_setup", "customer_segment": "developer"},
    {"id": 2, "issue": "Daytona SDK installation fails with pip", "solution": "Try: pip install --upgrade pip && pip install daytona-sdk. If SSL error, use: pip install --trusted-host pypi.org daytona-sdk", "issue_type": "sdk_setup", "customer_segment": "developer"},
    {"id": 3, "issue": "How to install Daytona SDK in a virtual environment?", "solution": "python -m venv daytona_env && source daytona_env/bin/activate && pip install daytona-sdk", "issue_type": "sdk_setup", "customer_segment": "developer"},
    {"id": 4, "issue": "TypeScript/JavaScript SDK installation", "solution": "npm install @daytona/sdk or yarn add @daytona/sdk. Import: import { Daytona } from '@daytona/sdk'", "issue_type": "sdk_setup", "customer_segment": "developer"},
    {"id": 5, "issue": "SDK version compatibility issues", "solution": "Check Python >= 3.8 required. Use pip show daytona-sdk to verify version. Update with pip install --upgrade daytona-sdk", "issue_type": "sdk_setup", "customer_segment": "developer"},
    {"id": 6, "issue": "ImportError when importing Daytona", "solution": "Ensure package installed in correct environment. Run: python -c 'import daytona; print(daytona.__version__)'", "issue_type": "sdk_setup", "customer_segment": "developer"},
    {"id": 7, "issue": "How to configure Daytona SDK with environment variables?", "solution": "Set DAYTONA_API_KEY env var. SDK auto-detects: export DAYTONA_API_KEY=your_key. Or pass directly: Daytona(api_key=key)", "issue_type": "sdk_setup", "customer_segment": "developer"},
    {"id": 8, "issue": "Daytona SDK proxy configuration", "solution": "Set HTTP_PROXY and HTTPS_PROXY environment variables. SDK respects standard proxy settings.", "issue_type": "sdk_setup", "customer_segment": "enterprise"},
    {"id": 9, "issue": "SDK not working behind corporate firewall", "solution": "Configure proxy settings, whitelist *.daytona.io domains. Contact IT for firewall rules.", "issue_type": "sdk_setup", "customer_segment": "enterprise"},
    {"id": 10, "issue": "How to use Daytona SDK in Jupyter notebooks?", "solution": "pip install daytona-sdk in Jupyter kernel. Use async: await daytona.create_async(params) or sync API normally.", "issue_type": "sdk_setup", "customer_segment": "developer"},
    
    # Sandbox Creation & Management (15 cases)
    {"id": 11, "issue": "How to create a Python sandbox?", "solution": "daytona = Daytona(); sandbox = daytona.create(CreateSandboxParams(language='python'))", "issue_type": "sandbox_creation", "customer_segment": "developer"},
    {"id": 12, "issue": "Sandbox creation timeout error", "solution": "Increase timeout: daytona.create(params, timeout=120). Check network connectivity. Retry with exponential backoff.", "issue_type": "sandbox_creation", "customer_segment": "developer"},
    {"id": 13, "issue": "How to create sandbox with custom Docker image?", "solution": "params = CreateSandboxParams(image='python:3.11-slim'); sandbox = daytona.create(params)", "issue_type": "sandbox_creation", "customer_segment": "devops"},
    {"id": 14, "issue": "Sandbox creation fails with resource limit error", "solution": "Check account quotas in dashboard. Upgrade plan for more concurrent sandboxes. Delete unused sandboxes.", "issue_type": "sandbox_creation", "customer_segment": "startup"},
    {"id": 15, "issue": "How to create multiple sandboxes in parallel?", "solution": "Use asyncio: tasks = [daytona.create_async(params) for _ in range(5)]; sandboxes = await asyncio.gather(*tasks)", "issue_type": "sandbox_creation", "customer_segment": "developer"},
    {"id": 16, "issue": "Sandbox stuck in 'creating' state", "solution": "Wait 2 minutes, then call daytona.get_sandbox(id) to check status. If still stuck, delete and recreate.", "issue_type": "sandbox_creation", "customer_segment": "developer"},
    {"id": 17, "issue": "How to specify sandbox resources (CPU/memory)?", "solution": "params = CreateSandboxParams(language='python', cpu=2, memory_gb=4); sandbox = daytona.create(params)", "issue_type": "sandbox_creation", "customer_segment": "enterprise"},
    {"id": 18, "issue": "Sandbox creation with GPU support", "solution": "params = CreateSandboxParams(language='python', gpu=True, gpu_type='nvidia-t4'). GPU sandboxes available on Enterprise plan.", "issue_type": "sandbox_creation", "customer_segment": "enterprise"},
    {"id": 19, "issue": "How to list all active sandboxes?", "solution": "sandboxes = daytona.list_sandboxes(); for s in sandboxes: print(s.id, s.status)", "issue_type": "sandbox_creation", "customer_segment": "developer"},
    {"id": 20, "issue": "Sandbox auto-shutdown configuration", "solution": "params = CreateSandboxParams(idle_timeout_minutes=30). Sandbox auto-stops after 30 min of inactivity.", "issue_type": "sandbox_creation", "customer_segment": "startup"},
    {"id": 21, "issue": "How to keep sandbox running indefinitely?", "solution": "Set idle_timeout_minutes=0 to disable auto-shutdown. Note: billing continues. Use keep-alive pings for long tasks.", "issue_type": "sandbox_creation", "customer_segment": "enterprise"},
    {"id": 22, "issue": "Sandbox region selection", "solution": "params = CreateSandboxParams(region='eu-central-1'). Available: us-east-1, us-west-2, eu-central-1, ap-south-1", "issue_type": "sandbox_creation", "customer_segment": "enterprise"},
    {"id": 23, "issue": "How to clone a sandbox?", "solution": "new_sandbox = daytona.clone_sandbox(original_sandbox.id). Clones filesystem and environment.", "issue_type": "sandbox_creation", "customer_segment": "developer"},
    {"id": 24, "issue": "Sandbox with pre-installed packages", "solution": "Create with requirements: params = CreateSandboxParams(requirements=['numpy', 'pandas', 'scikit-learn'])", "issue_type": "sandbox_creation", "customer_segment": "developer"},
    {"id": 25, "issue": "How to snapshot and restore sandbox state?", "solution": "snapshot_id = daytona.snapshot(sandbox.id); later: sandbox = daytona.restore(snapshot_id)", "issue_type": "sandbox_creation", "customer_segment": "enterprise"},
    
    # Code Execution (12 cases)
    {"id": 26, "issue": "How to run Python code in sandbox?", "solution": "result = sandbox.process.code_run('print(1+1)'); print(result.output)", "issue_type": "code_execution", "customer_segment": "developer"},
    {"id": 27, "issue": "Code execution returns empty output", "solution": "Ensure code has print() statements. Check result.stderr for errors. Use result.exit_code to verify success.", "issue_type": "code_execution", "customer_segment": "developer"},
    {"id": 28, "issue": "Execution timeout - code takes too long", "solution": "Increase timeout: sandbox.process.code_run(code, timeout=300) for 5 minutes. Consider chunking long operations.", "issue_type": "code_execution", "customer_segment": "developer"},
    {"id": 29, "issue": "How to run shell commands?", "solution": "result = sandbox.process.exec('ls -la /home/daytona'); print(result.stdout)", "issue_type": "code_execution", "customer_segment": "developer"},
    {"id": 30, "issue": "Exit code is non-zero but code looks correct", "solution": "Check result.stderr for error details. Common: missing imports, file not found, permission denied.", "issue_type": "code_execution", "customer_segment": "developer"},
    {"id": 31, "issue": "How to pass environment variables to code?", "solution": "sandbox.process.code_run(code, env={'API_KEY': 'secret', 'DEBUG': 'true'})", "issue_type": "code_execution", "customer_segment": "developer"},
    {"id": 32, "issue": "Code execution with pip install", "solution": "First install: sandbox.process.exec('pip install requests'); then: sandbox.process.code_run('import requests; print(requests.get(url).text)')", "issue_type": "code_execution", "customer_segment": "developer"},
    {"id": 33, "issue": "How to run background processes?", "solution": "result = sandbox.process.exec('nohup python server.py &', background=True). Use sandbox.process.ps() to check.", "issue_type": "code_execution", "customer_segment": "devops"},
    {"id": 34, "issue": "Streaming output from long-running code", "solution": "Use streaming callback: sandbox.process.code_run(code, stream=lambda chunk: print(chunk, end=''))", "issue_type": "code_execution", "customer_segment": "developer"},
    {"id": 35, "issue": "Code crashes with out of memory error", "solution": "Upgrade sandbox resources or optimize code. Use generators instead of lists. Process data in chunks.", "issue_type": "code_execution", "customer_segment": "developer"},
    {"id": 36, "issue": "How to run multiple scripts sequentially?", "solution": "results = [sandbox.process.code_run(script) for script in scripts]. Check each result.exit_code.", "issue_type": "code_execution", "customer_segment": "developer"},
    {"id": 37, "issue": "Interactive REPL session in sandbox", "solution": "Use sandbox.process.interactive() for REPL. Send commands with session.send('x = 5\\n').", "issue_type": "code_execution", "customer_segment": "developer"},
    
    # File Operations (10 cases)
    {"id": 38, "issue": "How to upload files to sandbox?", "solution": "sandbox.fs.upload_file('/home/daytona/data.csv', file_bytes). Or upload_file_from_path(local_path, remote_path)", "issue_type": "file_operations", "customer_segment": "developer"},
    {"id": 39, "issue": "Download files from sandbox", "solution": "content = sandbox.fs.download_file('/home/daytona/output.json'); with open('local.json', 'wb') as f: f.write(content)", "issue_type": "file_operations", "customer_segment": "developer"},
    {"id": 40, "issue": "List files in sandbox directory", "solution": "files = sandbox.fs.list_dir('/home/daytona'); for f in files: print(f.name, f.size)", "issue_type": "file_operations", "customer_segment": "developer"},
    {"id": 41, "issue": "Upload large files (>100MB)", "solution": "Use chunked upload: sandbox.fs.upload_file_chunked(path, chunk_size=10*1024*1024). Or use presigned URLs.", "issue_type": "file_operations", "customer_segment": "enterprise"},
    {"id": 42, "issue": "File upload fails with permission denied", "solution": "Upload to /home/daytona directory. Or run: sandbox.process.exec('chmod 777 /target/dir') first.", "issue_type": "file_operations", "customer_segment": "developer"},
    {"id": 43, "issue": "How to create directories in sandbox?", "solution": "sandbox.fs.mkdir('/home/daytona/my_project', parents=True). Creates all parent directories.", "issue_type": "file_operations", "customer_segment": "developer"},
    {"id": 44, "issue": "Delete files from sandbox", "solution": "sandbox.fs.delete('/home/daytona/temp_file.txt'). For directories: sandbox.fs.delete_dir('/path', recursive=True)", "issue_type": "file_operations", "customer_segment": "developer"},
    {"id": 45, "issue": "Sync local folder to sandbox", "solution": "sandbox.fs.sync_folder('./local_project', '/home/daytona/project'). Syncs entire directory tree.", "issue_type": "file_operations", "customer_segment": "developer"},
    {"id": 46, "issue": "File encoding issues (UTF-8)", "solution": "Ensure files are UTF-8 encoded. Use: content.encode('utf-8') before upload. Specify encoding in code_run.", "issue_type": "file_operations", "customer_segment": "developer"},
    {"id": 47, "issue": "How to check if file exists?", "solution": "exists = sandbox.fs.exists('/home/daytona/config.json'); if exists: process_file()", "issue_type": "file_operations", "customer_segment": "developer"},
    
    # Authentication & API Keys (8 cases)
    {"id": 48, "issue": "How to get Daytona API key?", "solution": "Log in to daytona.io/dashboard → Settings → API Keys → Create New Key. Save securely, shown only once.", "issue_type": "authentication", "customer_segment": "onboarding"},
    {"id": 49, "issue": "API key not working - unauthorized error", "solution": "Check key hasn't expired. Verify no extra spaces. Regenerate if compromised. Check account status.", "issue_type": "authentication", "customer_segment": "developer"},
    {"id": 50, "issue": "How to rotate API keys safely?", "solution": "Create new key first, update all deployments, test, then revoke old key from dashboard.", "issue_type": "authentication", "customer_segment": "enterprise"},
    {"id": 51, "issue": "API key security best practices", "solution": "Never commit to git. Use environment variables. Rotate quarterly. Use different keys for dev/prod.", "issue_type": "authentication", "customer_segment": "enterprise"},
    {"id": 52, "issue": "Multiple API keys for team members", "solution": "Enterprise plan allows multiple keys. Create per-user keys in Team Settings for audit trails.", "issue_type": "authentication", "customer_segment": "enterprise"},
    {"id": 53, "issue": "API rate limiting - too many requests", "solution": "Default: 100 requests/min. Implement exponential backoff. Upgrade plan for higher limits.", "issue_type": "authentication", "customer_segment": "startup"},
    {"id": 54, "issue": "OAuth/SSO integration", "solution": "Enterprise feature. Configure SAML/OIDC in Admin Settings. Contact sales for setup assistance.", "issue_type": "authentication", "customer_segment": "enterprise"},
    {"id": 55, "issue": "API key permissions and scopes", "solution": "Keys can be scoped: read-only, sandbox-only, admin. Set in dashboard when creating key.", "issue_type": "authentication", "customer_segment": "enterprise"},
    
    # Pricing & Billing (8 cases)
    {"id": 56, "issue": "What is Daytona pricing?", "solution": "Pay-as-you-go: CPU $0.0504/hr, Memory $0.0162/GB/hr, Storage $0.0001/GB/hr. Free tier: $200 compute credits.", "issue_type": "billing", "customer_segment": "startup"},
    {"id": 57, "issue": "How to track usage and costs?", "solution": "Dashboard → Billing → Usage shows real-time costs. Set up billing alerts for thresholds.", "issue_type": "billing", "customer_segment": "startup"},
    {"id": 58, "issue": "Unexpected charges on invoice", "solution": "Check for forgotten sandboxes: daytona.list_sandboxes(). Review usage in billing dashboard.", "issue_type": "billing", "customer_segment": "startup"},
    {"id": 59, "issue": "How to set spending limits?", "solution": "Billing → Spending Limits → Set monthly cap. Sandboxes pause when limit reached.", "issue_type": "billing", "customer_segment": "startup"},
    {"id": 60, "issue": "Enterprise pricing and contracts", "solution": "Contact sales for volume discounts, annual contracts, SLAs, and dedicated support.", "issue_type": "billing", "customer_segment": "enterprise"},
    {"id": 61, "issue": "Free trial extension request", "solution": "Contact support with use case. Extensions granted for active evaluation, up to 30 days.", "issue_type": "billing", "customer_segment": "startup"},
    {"id": 62, "issue": "How to download invoices?", "solution": "Dashboard → Billing → Invoices → Download PDF. Invoices available for current and past months.", "issue_type": "billing", "customer_segment": "enterprise"},
    {"id": 63, "issue": "Changing payment method", "solution": "Billing → Payment Methods → Add card / Update. Old cards can be removed after new one added.", "issue_type": "billing", "customer_segment": "startup"},
    
    # Performance & Optimization (8 cases)  
    {"id": 64, "issue": "Sandbox creation is slow", "solution": "Use warm pools: daytona.create(params, warm_pool=True). Pre-warmed sandboxes start in <1s.", "issue_type": "performance", "customer_segment": "developer"},
    {"id": 65, "issue": "Code execution latency is high", "solution": "Choose nearest region. Use keep-alive connections. Batch multiple operations. Cache sandbox instance.", "issue_type": "performance", "customer_segment": "developer"},
    {"id": 66, "issue": "How to optimize for cold starts?", "solution": "Use minimal base image. Pre-install dependencies in custom image. Use sandbox pools.", "issue_type": "performance", "customer_segment": "devops"},
    {"id": 67, "issue": "API response times are slow", "solution": "Check network latency to Daytona servers. Use regional endpoints. Enable gzip compression.", "issue_type": "performance", "customer_segment": "developer"},
    {"id": 68, "issue": "Parallel execution not scaling", "solution": "Check account concurrency limits. Use connection pooling. Implement proper async patterns.", "issue_type": "performance", "customer_segment": "enterprise"},
    {"id": 69, "issue": "Memory usage growing over time", "solution": "Sandbox memory leaks: restart periodically. Clean up temp files. Use gc.collect() in Python.", "issue_type": "performance", "customer_segment": "developer"},
    {"id": 70, "issue": "How to benchmark Daytona performance?", "solution": "Use daytona.benchmark() method. Measures: creation time, execution latency, file I/O speeds.", "issue_type": "performance", "customer_segment": "devops"},
    {"id": 71, "issue": "Network latency between sandbox and external APIs", "solution": "Sandbox in same region as your APIs. Use async/parallel requests. Implement caching.", "issue_type": "performance", "customer_segment": "developer"},
    
    # Docker & Containers (7 cases)
    {"id": 72, "issue": "Can I use Docker inside Daytona?", "solution": "Yes! Docker-in-Docker supported. sandbox.process.exec('docker run hello-world'). Requires Docker-enabled sandbox.", "issue_type": "docker", "customer_segment": "devops"},
    {"id": 73, "issue": "Docker build inside sandbox", "solution": "Add Dockerfile, then: sandbox.process.exec('docker build -t myapp .'). Full docker CLI available.", "issue_type": "docker", "customer_segment": "devops"},
    {"id": 74, "issue": "Docker compose in sandbox", "solution": "sandbox.process.exec('docker compose up -d'). Ensure docker-compose.yml uploaded first.", "issue_type": "docker", "customer_segment": "devops"},
    {"id": 75, "issue": "Custom base image for sandbox", "solution": "Use any public Docker image: CreateSandboxParams(image='pytorch/pytorch:2.0-cuda11.8-cudnn8-runtime')", "issue_type": "docker", "customer_segment": "devops"},
    {"id": 76, "issue": "Private Docker registry support", "solution": "Configure registry credentials: daytona.add_registry('registry.mycompany.com', user, password)", "issue_type": "docker", "customer_segment": "enterprise"},
    {"id": 77, "issue": "Docker volume persistence", "solution": "Named volumes persist across sandbox restarts. Use: CreateSandboxParams(volumes={'data': '/app/data'})", "issue_type": "docker", "customer_segment": "devops"},
    {"id": 78, "issue": "Container networking in sandbox", "solution": "Containers share network. Use localhost to communicate. Port mapping: sandbox.expose_port(8080)", "issue_type": "docker", "customer_segment": "devops"},
    
    # Security & Compliance (7 cases)
    {"id": 79, "issue": "Is Daytona SOC 2 compliant?", "solution": "Yes, Daytona is SOC 2 Type II certified. Request compliance docs from enterprise sales.", "issue_type": "security", "customer_segment": "enterprise"},
    {"id": 80, "issue": "HIPAA compliance for healthcare data", "solution": "HIPAA-enabled sandboxes available on Enterprise plan. BAA required. Contact sales.", "issue_type": "security", "customer_segment": "enterprise"},
    {"id": 81, "issue": "Data isolation between sandboxes", "solution": "Full isolation: separate VMs, network namespaces, encrypted storage. No cross-sandbox access.", "issue_type": "security", "customer_segment": "enterprise"},
    {"id": 82, "issue": "How is data encrypted?", "solution": "AES-256 at rest, TLS 1.3 in transit. Customer-managed keys available on Enterprise.", "issue_type": "security", "customer_segment": "enterprise"},
    {"id": 83, "issue": "Audit logs for compliance", "solution": "Full audit trail in Enterprise dashboard. Export logs to SIEM. API for log retrieval.", "issue_type": "security", "customer_segment": "enterprise"},
    {"id": 84, "issue": "VPC/Private deployment option", "solution": "Daytona Private Cloud: runs in your AWS/GCP/Azure. Full data sovereignty. Contact sales.", "issue_type": "security", "customer_segment": "enterprise"},
    {"id": 85, "issue": "IP whitelisting for API access", "solution": "Enterprise feature: Settings → Security → IP Allowlist. Add CIDR ranges.", "issue_type": "security", "customer_segment": "enterprise"},
    
    # Integrations (8 cases)
    {"id": 86, "issue": "How to integrate Daytona with GitHub Actions?", "solution": "Use daytona-action in workflow. Store API key in secrets. Example: uses: daytona/setup-action@v1", "issue_type": "integration", "customer_segment": "devops"},
    {"id": 87, "issue": "GitLab CI integration", "solution": "Add DAYTONA_API_KEY to CI variables. Use SDK in pipeline scripts. See docs for .gitlab-ci.yml example.", "issue_type": "integration", "customer_segment": "devops"},
    {"id": 88, "issue": "Slack notifications for sandbox events", "solution": "Configure webhooks: Settings → Integrations → Slack. Get notified on create/destroy/error.", "issue_type": "integration", "customer_segment": "startup"},
    {"id": 89, "issue": "Terraform provider for Daytona", "solution": "terraform-provider-daytona available. Manage sandboxes as infrastructure code.", "issue_type": "integration", "customer_segment": "devops"},
    {"id": 90, "issue": "VS Code extension for Daytona", "solution": "Install 'Daytona' extension from marketplace. Connect sandboxes, edit files, run code in IDE.", "issue_type": "integration", "customer_segment": "developer"},
    {"id": 91, "issue": "Jupyter integration with Daytona", "solution": "Daytona Jupyter kernel allows running notebook cells in cloud sandboxes. pip install daytona-jupyter", "issue_type": "integration", "customer_segment": "developer"},
    {"id": 92, "issue": "Webhook notifications for events", "solution": "Settings → Webhooks → Add URL. Events: sandbox.created, sandbox.destroyed, execution.completed", "issue_type": "integration", "customer_segment": "devops"},
    {"id": 93, "issue": "REST API for programmatic access", "solution": "Full REST API available. OpenAPI spec at api.daytona.io/docs. SDK is REST wrapper.", "issue_type": "integration", "customer_segment": "developer"},
    
    # Troubleshooting (7 cases)
    {"id": 94, "issue": "Sandbox not responding", "solution": "Check status: daytona.get_sandbox(id).status. If 'unhealthy', recreate. Check resource limits.", "issue_type": "troubleshooting", "customer_segment": "developer"},
    {"id": 95, "issue": "Connection reset errors", "solution": "Network issue: retry with backoff. Check firewall. Verify API endpoint reachable.", "issue_type": "troubleshooting", "customer_segment": "developer"},
    {"id": 96, "issue": "Sandbox filesystem is full", "solution": "Clean temp files: sandbox.process.exec('rm -rf /tmp/*'). Check with df -h. Increase storage if needed.", "issue_type": "troubleshooting", "customer_segment": "developer"},
    {"id": 97, "issue": "Package installation fails in sandbox", "solution": "Check internet access. Use mirrors: pip install -i https://pypi.org/simple package. Verify package exists.", "issue_type": "troubleshooting", "customer_segment": "developer"},
    {"id": 98, "issue": "How to get sandbox logs?", "solution": "sandbox.logs() returns recent logs. For streaming: sandbox.logs(follow=True, since='1h')", "issue_type": "troubleshooting", "customer_segment": "developer"},
    {"id": 99, "issue": "Debug mode for verbose logging", "solution": "Enable debug: daytona = Daytona(debug=True). All API calls logged with timing.", "issue_type": "troubleshooting", "customer_segment": "developer"},
    {"id": 100, "issue": "How to report bugs to Daytona?", "solution": "Email support@daytona.io with sandbox ID, timestamps, error messages. Or use in-app chat.", "issue_type": "troubleshooting", "customer_segment": "developer"},
]

async def populate():
    # 1. Initialize Embeddings
    logger.info(f"🚀 Initializing OptimizedEmbeddings with model: {MODEL_PATH}")
    embeddings = OptimizedEmbeddings(model_path=MODEL_PATH)
    
    # 2. Delete and recreate collection for clean state
    logger.info(f"🗑️ Deleting existing collection: {COLLECTION_NAME}")
    try:
        requests.delete(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
    except:
        pass
    
    logger.info(f"📊 Creating fresh collection with 384 dimensions...")
    create_resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
        json={
            "vectors": {
                "size": 384,
                "distance": "Cosine"
            }
        }
    )
    if create_resp.status_code == 200:
        logger.info("✅ Collection created successfully!")
    else:
        logger.error(f"❌ Failed to create collection: {create_resp.text}")
        return

    # 3. Process and Upsert points in batches
    batch_size = 10
    total_points = len(CASES)
    
    for batch_start in range(0, total_points, batch_size):
        batch_end = min(batch_start + batch_size, total_points)
        batch = CASES[batch_start:batch_end]
        
        points = []
        for case in batch:
            logger.info(f"📝 [{case['id']}/{total_points}] Embedding: {case['issue'][:50]}...")
            # Combine issue + solution for richer embedding
            text = f"{case['issue']} {case['solution']}"
            vector = embeddings.embed_query(text)
            
            points.append({
                "id": case['id'],
                "vector": vector,
                "payload": {
                    "issue": case['issue'],
                    "solution": case['solution'],
                    "issue_type": case['issue_type'],
                    "customer_segment": case['customer_segment'],
                    "product": "daytona",
                    "timestamp": "2024-01-18T12:00:00Z"
                }
            })
        
        # Upsert batch
        upsert_resp = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json={"points": points}
        )
        
        if upsert_resp.status_code == 200:
            logger.info(f"✅ Batch {batch_start//batch_size + 1} upserted ({len(points)} points)")
        else:
            logger.error(f"❌ Batch upsert failed: {upsert_resp.text}")

    # 4. Final verification
    count_resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/count",
        json={}
    )
    if count_resp.status_code == 200:
        count = count_resp.json().get("result", {}).get("count", 0)
        logger.info(f"\n🎉 SUCCESS! Total points in collection: {count}")

if __name__ == "__main__":
    asyncio.run(populate())
