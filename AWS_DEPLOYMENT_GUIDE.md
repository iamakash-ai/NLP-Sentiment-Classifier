# AWS Cloud Deployment Guide for Beginners

This guide will walk you through deploying the NLP ML project on AWS step by step.

## Prerequisites Checklist

- [ ] AWS Account (create at https://aws.amazon.com)
- [ ] AWS CLI installed locally
- [ ] Git installed
- [ ] GitHub account (for code repository)

---

## STEP 1: Set Up AWS Account & Configure Credentials

### 1.1 Create AWS Account
1. Go to https://aws.amazon.com
2. Click "Create an AWS Account"
3. Fill in your email, password, and account name
4. Add billing information
5. Verify phone number
6. Choose support plan (Free tier is fine)

### 1.2 Create IAM User for Deployment
1. Log into AWS Console
2. Go to **IAM > Users**
3. Click **Create user**
   - Username: `nlp-deployer`
   - Enable console access (optional)
4. Click **Next**
5. Attach permissions: Click **Attach policies directly**
   - Search and select: `AdministratorAccess` (for simplicity, use more restrictive permissions in production)
6. Click **Create user**
7. Copy the **Access Key ID** and **Secret Access Key**

### 1.3 Install AWS CLI
Open terminal/PowerShell and run:
```bash
# For Windows, use Python pip
pip install awscli

# Verify installation
aws --version
```

### 1.4 Configure AWS Credentials
```bash
aws configure
```

When prompted, enter:
```
AWS Access Key ID: [paste your Access Key ID]
AWS Secret Access Key: [paste your Secret Access Key]
Default region name: us-east-1
Default output format: json
```

Verify configuration:
```bash
aws sts get-caller-identity
```

---

## STEP 2: Create GitHub Repository

### 2.1 Create Repository
1. Go to https://github.com/new
2. Repository name: `nlp-ml-project`
3. Description: "NLP ML Project with AWS CI/CD"
4. Choose Public or Private
5. Click **Create repository**

### 2.2 Push Code to GitHub
In your local project folder:
```bash
cd nlp_project

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial NLP ML project setup"

# Add remote (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/nlp-ml-project.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## STEP 3: Create S3 Buckets for Artifacts & Data

### 3.1 Create Artifact Bucket
```bash
# Replace ACCOUNT_ID with your AWS Account ID (12 digits)
aws s3api create-bucket \
    --bucket nlp-artifacts-ACCOUNT_ID \
    --region us-east-1
```

### 3.2 Create Data Bucket
```bash
aws s3api create-bucket \
    --bucket nlp-data-ACCOUNT_ID \
    --region us-east-1
```

### 3.3 Upload Sample Data
```bash
# Upload training data to S3
aws s3 cp data/raw/training_data.csv \
    s3://nlp-data-ACCOUNT_ID/data/training_data.csv
```

---

## STEP 4: Deploy Infrastructure with CloudFormation

### 4.1 Create CloudFormation Stack
```bash
python ci_cd/aws_deploy.py \
    --action create \
    --stack-name nlp-ml-stack \
    --region us-east-1 \
    --environment dev \
    --project-name nlp-project
```

Monitor the deployment:
```bash
# Check stack status
aws cloudformation describe-stacks \
    --stack-name nlp-ml-stack \
    --query 'Stacks[0].StackStatus'
```

Wait for status to show `CREATE_COMPLETE` (usually 2-5 minutes)

---

## STEP 5: Set Up CodePipeline for CI/CD

### 5.1 Create CodePipeline
1. Go to **AWS Console > CodePipeline**
2. Click **Create pipeline**
3. Pipeline settings:
   - Pipeline name: `nlp-training-pipeline`
   - Service role: Create new role (AWS creates automatically)
4. Click **Next**

### 5.2 Add Source Stage
- Source provider: **GitHub**
- Click **Connect to GitHub**
- Authorize AWS CodePipeline
- Repository: Select `nlp-ml-project`
- Branch: `main`
- Click **Next**

### 5.3 Add Build Stage
- Build provider: **AWS CodeBuild**
- Click **Create project**
  - Project name: `nlp-model-training`
  - Environment: Managed image
  - OS: Amazon Linux 2
  - Runtime: Standard
  - Image: Latest available
  - Buildspec name: `ci_cd/buildspec.yml`
  - Environment variables:
    - `AWS_S3_BUCKET`: `nlp-artifacts-ACCOUNT_ID`
- Click **Create build project**
- Back in pipeline, select the build project
- Click **Next**

### 5.4 Skip Deploy Stage (For Now)
- Click **Skip deploy stage**
- Click **Create pipeline**

---

## STEP 6: Run Your First Training Job

### 6.1 Generate Sample Data
```bash
python create_sample_data.py
```

### 6.2 Push Changes to Trigger Pipeline
```bash
git add .
git commit -m "Add sample training data"
git push origin main
```

### 6.3 Monitor Build
1. Go to **AWS Console > CodePipeline > nlp-training-pipeline**
2. Watch the pipeline stages
3. Click on **Build** stage to see logs
4. Wait for status to turn **Succeeded**

---

## STEP 7: Deploy Streamlit App to AWS

### 7.1 Option A: Using EC2 (Recommended for Beginners)

**Step 1: Launch EC2 Instance**
1. Go to **AWS Console > EC2 > Instances**
2. Click **Launch Instance**
3. Choose AMI: **Ubuntu 22.04 LTS** (Free tier eligible)
4. Instance type: **t3.micro** or **t2.micro** (Free tier)
5. Key pair: Create new
   - Name: `nlp-app-key`
   - Type: `.pem`
   - Download the key file (save it securely)
6. Network settings: Allow SSH, HTTP, HTTPS traffic
7. Storage: 20 GB (default is fine)
8. Click **Launch instance**

**Step 2: Connect to EC2**
```bash
# Navigate to your key file location
cd path/to/nlp-app-key.pem

# Change permissions (Linux/Mac)
chmod 400 nlp-app-key.pem

# Connect to instance (replace INSTANCE_IP with your instance's public IP)
ssh -i nlp-app-key.pem ubuntu@INSTANCE_IP
```

**Step 3: Set Up Environment**
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3-pip python3-venv git

# Clone your repository
git clone https://github.com/USERNAME/nlp-ml-project.git
cd nlp-ml-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create models directory (download from S3 or train)
mkdir -p models
aws s3 cp s3://nlp-artifacts-ACCOUNT_ID/models/ models/ --recursive
```

**Step 4: Run Streamlit App**
```bash
# Install Streamlit specifically
pip install streamlit

# Run app (binds to all interfaces)
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0
```

**Step 5: Access App**
1. Get your EC2 instance's **Public IP**
2. Open browser: `http://YOUR_INSTANCE_IP:8501`

---

### 7.2 Option B: Using AWS AppRunner (Easiest - No Server Management)

**Step 1: Create Docker Image**

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

**Step 2: Push to ECR (Elastic Container Registry)**
```bash
# Create ECR repository
aws ecr create-repository --repository-name nlp-app

# Get login token
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push image
docker build -t nlp-app .
docker tag nlp-app:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nlp-app:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nlp-app:latest
```

**Step 3: Deploy with AppRunner**
1. Go to **AWS Console > App Runner**
2. Click **Create service**
3. Source: **ECR**
   - Repository URI: `ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nlp-app:latest`
   - Deployment: Automatic
4. Service name: `nlp-inference-app`
5. Port: `8501`
6. Click **Create and deploy**
7. Wait for service status to show **Running**

Your app URL will be displayed (e.g., `https://xxxxx.us-east-1.apprunner.amazonaws.com`)

---

### 7.3 Option C: Using Streamlit Cloud (Simplest)

1. Push your repo to GitHub
2. Go to https://streamlit.io/cloud
3. Click **New app**
4. Connect your GitHub account
5. Select repository: `nlp-ml-project`
6. Main file path: `streamlit_app.py`
7. Click **Deploy**

Your app will be live at: `https://nlp-ml-project.streamlit.app`

---

## STEP 8: Set Up Auto-Retraining

### 8.1 Configure Data Change Detection
1. Go to **AWS Console > S3 > nlp-data-ACCOUNT_ID**
2. Upload new training data whenever you want to retrain:
```bash
aws s3 cp new_data.csv \
    s3://nlp-data-ACCOUNT_ID/data/new_training_data.csv
```

### 8.2 Automatic Pipeline Trigger (Optional)
Create EventBridge rule to trigger pipeline on data upload:
1. Go to **AWS Console > EventBridge**
2. Create rule > Event source: S3
3. Set trigger: Object uploaded to `nlp-data-ACCOUNT_ID`
4. Target: CodePipeline `nlp-training-pipeline`

---

## STEP 9: Monitor Your Application

### 9.1 View Training Logs
```bash
# Get latest build logs
aws codebuild batch-get-builds \
    --ids $(aws codebuild list-builds-for-project \
    --project-name nlp-model-training \
    --query 'ids[0]' --output text)
```

### 9.2 Check Model Artifacts
```bash
# List models in S3
aws s3 ls s3://nlp-artifacts-ACCOUNT_ID/models/
```

### 9.3 View CloudWatch Logs
1. Go to **AWS Console > CloudWatch > Log Groups**
2. Look for logs starting with `/aws/codebuild/`

---

## Cost Estimation (Free Tier)

| Service | Free Tier Limit | Estimated Monthly Cost |
|---------|-----------------|------------------------|
| S3 | 5 GB storage | Free |
| CodeBuild | 100 build minutes/month | Free |
| CodePipeline | 1 active pipeline | Free |
| EC2 | 1 t2/t3.micro instance 750 hrs | Free |
| CloudFormation | Free | Free |
| **Total** | | **~$0-5** (if staying within free tier) |

---

## Troubleshooting

### Issue: Build fails with "Module not found"
**Solution:**
```bash
# SSH into EC2 and reinstall
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### Issue: Models not found during inference
**Solution:**
```bash
# Download models from S3
aws s3 cp s3://nlp-artifacts-ACCOUNT_ID/models/ models/ --recursive
```

### Issue: "Access Denied" when accessing S3
**Solution:**
```bash
# Check IAM permissions
aws s3 ls
# Update IAM user policy to include S3 access
```

### Issue: CodePipeline fails at build stage
**Solution:**
1. Go to **CodeBuild** and click on the failed build
2. View logs to see the error
3. Fix code locally and push to GitHub
4. Pipeline will automatically retry

---

## Quick Reference Commands

```bash
# Create sample data
python create_sample_data.py

# Train locally
python train.py

# Run Streamlit locally
streamlit run streamlit_app.py

# Deploy to AWS
python ci_cd/aws_deploy.py --action create --stack-name nlp-ml-stack

# Upload data to S3
aws s3 cp data/raw/training_data.csv s3://nlp-data-ACCOUNT_ID/

# Trigger pipeline
aws codepipeline start-pipeline-execution --name nlp-training-pipeline

# Check stack status
aws cloudformation describe-stacks --stack-name nlp-ml-stack

# View EC2 instances
aws ec2 describe-instances --query 'Reservations[].Instances[].PublicIpAddress'
```

---

## Next Steps

1. ✅ Set up AWS account & credentials
2. ✅ Create GitHub repository
3. ✅ Deploy infrastructure
4. ✅ Set up CI/CD pipeline
5. ✅ Deploy Streamlit app
6. ✅ Monitor application
7. 🔄 Continuously update data and retrain
8. 📊 Track model performance

---

## Support Resources

- **AWS Documentation**: https://docs.aws.amazon.com
- **AWS CLI Reference**: https://docs.aws.amazon.com/cli/latest/userguide/
- **Streamlit Docs**: https://docs.streamlit.io
- **scikit-learn Docs**: https://scikit-learn.org/stable/documentation.html

---

**Last Updated**: November 2024
**Status**: Ready for Deployment
