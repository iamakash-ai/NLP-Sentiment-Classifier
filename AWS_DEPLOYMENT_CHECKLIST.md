# ✅ AWS Deployment Checklist & Troubleshooting

Use this checklist to track your AWS deployment progress.

---

## PRE-DEPLOYMENT CHECKLIST

### Local Testing (Must Complete First!)
- [ ] `python create_sample_data.py` - Creates sample data
- [ ] `python train.py` - Model trains without errors
- [ ] Models created: `models/classifier.pkl` and `models/vectorizer.pkl`
- [ ] `streamlit run streamlit_app.py` - Web app runs locally
- [ ] `python validate_data.py` - All validation checks pass
- [ ] Test prediction with: `python example_usage.py`

### AWS Account Setup
- [ ] AWS account created at aws.amazon.com
- [ ] IAM user created named `nlp-deployer`
- [ ] Access Key ID and Secret Access Key generated and saved
- [ ] AWS CLI installed: `pip install awscli`
- [ ] AWS CLI configured: `aws configure`
- [ ] Verify credentials: `aws sts get-caller-identity`

### Code Repository
- [ ] GitHub account created
- [ ] Repository created named `nlp-ml-project`
- [ ] Code pushed to GitHub:
  ```bash
  git init
  git add .
  git commit -m "Initial commit"
  git remote add origin https://github.com/USERNAME/nlp-ml-project.git
  git push -u origin main
  ```

---

## STEP 1: AWS S3 Setup

### Create S3 Buckets
- [ ] Note your AWS Account ID (12 digits): `________________`

**For Artifacts:**
```bash
aws s3api create-bucket \
    --bucket nlp-artifacts-YOUR_ACCOUNT_ID \
    --region us-east-1
```
- [ ] Artifact bucket created: `nlp-artifacts-YOUR_ACCOUNT_ID`

**For Data:**
```bash
aws s3api create-bucket \
    --bucket nlp-data-YOUR_ACCOUNT_ID \
    --region us-east-1
```
- [ ] Data bucket created: `nlp-data-YOUR_ACCOUNT_ID`

### Upload Initial Data
```bash
aws s3 cp data/raw/training_data.csv \
    s3://nlp-data-YOUR_ACCOUNT_ID/data/training_data.csv
```
- [ ] Training data uploaded to S3

### Verify Buckets
```bash
aws s3 ls
```
- [ ] Both buckets appear in list

---

## STEP 2: CloudFormation Deployment

### Deploy Infrastructure
```bash
python ci_cd/aws_deploy.py \
    --action create \
    --stack-name nlp-ml-stack \
    --region us-east-1 \
    --environment dev \
    --project-name nlp-project
```

- [ ] Deployment command executed
- [ ] CloudFormation stack shows status: `CREATE_IN_PROGRESS`

### Monitor Deployment
```bash
aws cloudformation describe-stacks \
    --stack-name nlp-ml-stack \
    --query 'Stacks[0].StackStatus'
```

- [ ] Wait for status: `CREATE_COMPLETE` (2-5 minutes)
- [ ] Get stack outputs:
  ```bash
  aws cloudformation describe-stacks \
      --stack-name nlp-ml-stack \
      --query 'Stacks[0].Outputs'
  ```
- [ ] Note these values:
  - Artifact Bucket: `__________________________`
  - Data Bucket: `__________________________`
  - CodeBuild Project: `__________________________`

---

## STEP 3: CodePipeline Setup

### Create CodePipeline in AWS Console

1. [ ] Go to AWS Console > CodePipeline
2. [ ] Click "Create pipeline"
3. [ ] Pipeline name: `nlp-training-pipeline`

### Source Stage
- [ ] Source provider: GitHub
- [ ] Connect to GitHub (authorize AWS)
- [ ] Repository: `nlp-ml-project`
- [ ] Branch: `main`

### Build Stage
- [ ] Build provider: AWS CodeBuild
- [ ] Create project:
  - [ ] Project name: `nlp-model-training`
  - [ ] Environment: Managed image > Amazon Linux 2
  - [ ] Runtime: Standard
  - [ ] Buildspec: `ci_cd/buildspec.yml`
  - [ ] Environment variables:
    - [ ] `AWS_S3_BUCKET`: `nlp-artifacts-YOUR_ACCOUNT_ID`
  - [ ] Click "Create build project"
- [ ] Select newly created project

### Deploy Stage
- [ ] Skip deploy for now (we'll deploy Streamlit separately)
- [ ] Click "Create pipeline"

### Verify Pipeline
- [ ] Pipeline status shows "Succeeded" (wait 5-10 minutes)
- [ ] Check logs at: AWS Console > CodeBuild > Build History

---

## STEP 4: Trigger First Training Run

### Push Code to Trigger Pipeline
```bash
git add .
git commit -m "Trigger initial training"
git push origin main
```

- [ ] Changes pushed to GitHub
- [ ] Pipeline automatically triggered

### Monitor Training
- [ ] Go to AWS Console > CodePipeline > nlp-training-pipeline
- [ ] Watch Source stage: `Succeeded`
- [ ] Watch Build stage: `In Progress` → `Succeeded`
- [ ] Training logs appear in CodeBuild

### Verify Artifacts
```bash
aws s3 ls s3://nlp-artifacts-YOUR_ACCOUNT_ID/
```

- [ ] Models uploaded to S3
- [ ] Check: `models/classifier.pkl`
- [ ] Check: `models/vectorizer.pkl`

---

## STEP 5: Deploy Streamlit App

### Option A: EC2 (Most Flexible)

#### Launch EC2 Instance
1. [ ] Go to AWS Console > EC2 > Instances > Launch Instance
2. [ ] AMI: Ubuntu 22.04 LTS (Free tier)
3. [ ] Instance type: t2.micro or t3.micro (Free tier)
4. [ ] Key pair: Create new > Name: `nlp-app-key` > Download `.pem` file
5. [ ] Security group: Allow SSH (22), HTTP (80), HTTPS (443)
6. [ ] Storage: 20 GB (default)
7. [ ] Launch instance
8. [ ] Record instance IP: `__________________________`

#### Connect to Instance
```bash
chmod 400 nlp-app-key.pem
ssh -i nlp-app-key.pem ubuntu@YOUR_INSTANCE_IP
```
- [ ] Successfully connected to EC2

#### Set Up Environment
```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv git

git clone https://github.com/YOUR_USERNAME/nlp-ml-project.git
cd nlp-ml-project

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```
- [ ] Environment set up

#### Download Models from S3
```bash
mkdir -p models
aws s3 cp s3://nlp-artifacts-YOUR_ACCOUNT_ID/models/ models/ --recursive
```
- [ ] Models downloaded

#### Run Streamlit
```bash
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0
```
- [ ] Streamlit running

#### Access App
- [ ] Browser: `http://YOUR_INSTANCE_IP:8501`
- [ ] Web app displays correctly

### Option B: Streamlit Cloud (Easiest)

1. [ ] Go to https://streamlit.io/cloud
2. [ ] Click "New app"
3. [ ] Connect GitHub account
4. [ ] Select repository: `nlp-ml-project`
5. [ ] Main file: `streamlit_app.py`
6. [ ] Click "Deploy"
7. [ ] App URL: `https://your-app.streamlit.app`
8. [ ] Wait for deployment to complete
9. [ ] Test web app in browser

### Option C: AWS AppRunner (Easiest AWS Option)

#### Build Docker Image
```bash
docker build -t nlp-app .
```
- [ ] Docker image built successfully

#### Push to ECR
```bash
aws ecr create-repository --repository-name nlp-app

aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker tag nlp-app:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nlp-app:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nlp-app:latest
```
- [ ] Image pushed to ECR

#### Deploy with AppRunner
1. [ ] AWS Console > App Runner > Create service
2. [ ] Source: ECR
3. [ ] Repository: `nlp-app`
4. [ ] Tag: `latest`
5. [ ] Service name: `nlp-inference-app`
6. [ ] Port: `8501`
7. [ ] Deploy
8. [ ] Wait for "Running" status
9. [ ] App URL provided (e.g., `https://xxxxx.us-east-1.apprunner.amazonaws.com`)

---

## STEP 6: Test Everything

### Test Model Training
```bash
python ci_cd/aws_deploy.py --action create --stack-name nlp-ml-test
```
- [ ] CloudFormation stack created successfully

### Test Single Prediction
From local machine:
```bash
from src.prediction import single_predict
result = single_predict("This is great!")
print(result)
```
- [ ] Returns prediction with confidence

### Test Web App
1. [ ] Navigate to app URL
2. [ ] Enter test text in "Single Prediction"
3. [ ] Click "Predict"
4. [ ] See prediction result
5. [ ] Try "Batch Prediction" with multiple texts
6. [ ] Try "File Upload" with CSV

### Test Auto-Retraining
1. [ ] Upload new data to S3:
   ```bash
   aws s3 cp new_data.csv s3://nlp-data-YOUR_ACCOUNT_ID/
   ```
2. [ ] Push code change to GitHub
3. [ ] Pipeline automatically triggers
4. [ ] Training completes
5. [ ] New models uploaded to S3

---

## STEP 7: Production Hardening

### Security
- [ ] Enable S3 versioning on data bucket
- [ ] Enable S3 encryption on both buckets
- [ ] Restrict IAM user permissions (least privilege)
- [ ] Enable CloudTrail for audit logging

### Monitoring
- [ ] Set up CloudWatch alarms for build failures
- [ ] Monitor EC2 CPU usage
- [ ] Check logs regularly

### Cost Optimization
- [ ] Review AWS Free Tier limits
- [ ] Set up billing alerts
- [ ] Delete test stacks when not needed

---

## TROUBLESHOOTING GUIDE

### Error: "AWS Access Denied"
**Problem:** Credentials not configured
**Solution:**
```bash
aws configure
# Enter Access Key ID
# Enter Secret Access Key
# Enter region: us-east-1
```
- [ ] Credentials reconfigured
- [ ] Test: `aws sts get-caller-identity`

### Error: "CodeBuild: Build failed"
**Problem:** Build script error
**Solution:**
1. [ ] Go to AWS Console > CodeBuild > Build History
2. [ ] Click failed build
3. [ ] View build logs
4. [ ] Common issues:
   - Missing NLTK data: Add to buildspec.yml
   - Memory issue: Increase CodeBuild instance type
   - S3 permission: Add S3 policy to IAM role

### Error: "Streamlit app shows 'Models not found'"
**Problem:** Models not downloaded
**Solution:**
```bash
# On EC2/container:
aws s3 cp s3://nlp-artifacts-YOUR_ACCOUNT_ID/models/ models/ --recursive
```
- [ ] Models manually downloaded

### Error: "Port 8501 already in use"
**Problem:** Another process using port
**Solution:**
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
# Access at: http://localhost:8502
```
- [ ] App running on different port

### Error: "Cannot connect to EC2"
**Problem:** SSH key or security group issue
**Solution:**
```bash
# Check key permissions
chmod 400 nlp-app-key.pem

# Verify security group allows SSH (port 22)
# AWS Console > EC2 > Security Groups > Check inbound rules
```
- [ ] SSH connection successful

### Error: "Pipeline fails at 'Source' stage"
**Problem:** GitHub authorization
**Solution:**
1. [ ] Go to AWS Console > CodePipeline
2. [ ] Click "Edit"
3. [ ] Source stage > Edit > Disconnect > Reconnect to GitHub
4. [ ] Authorize again
- [ ] Pipeline reconnected

---

## COMPLETION CHECKLIST

### All Deployments Done?
- [ ] Step 1: S3 buckets created
- [ ] Step 2: CloudFormation stack deployed
- [ ] Step 3: CodePipeline created
- [ ] Step 4: First training run succeeded
- [ ] Step 5: Streamlit app deployed
- [ ] Step 6: All tests passed
- [ ] Step 7: Production hardening complete

### Final Verification
```bash
# Check S3 buckets
aws s3 ls

# Check CloudFormation stacks
aws cloudformation describe-stacks --query 'Stacks[0].StackStatus'

# Check CodePipeline status
aws codepipeline get-pipeline-state --name nlp-training-pipeline

# Check EC2 instances (if using EC2)
aws ec2 describe-instances --query 'Reservations[].Instances[].PublicIpAddress'
```

- [ ] All resources exist and are operational
- [ ] App is accessible and working
- [ ] Models are training automatically

---

## 🎉 SUCCESS!

Your NLP ML project is now deployed on AWS with:
- ✅ Automated CI/CD pipeline
- ✅ Auto-retraining on data changes
- ✅ Web interface for predictions
- ✅ Production-ready infrastructure

### Estimated Monthly Cost (with Free Tier):
- S3: $0 (5 GB included)
- CodeBuild: $0 (100 minutes included)
- CodePipeline: $0 (1 pipeline free)
- EC2/AppRunner: $0-10 (depends on usage)
- **Total: ~$0-15/month**

---

## 📞 Support

1. See **AWS_DEPLOYMENT_GUIDE.md** for detailed instructions
2. See **README.md** for full documentation
3. See **QUICKSTART_BEGINNER.md** for local setup
4. Check **example_usage.py** for code examples
5. Run **validate_data.py** to check setup

---

**Status:** Ready for Production
**Last Updated:** November 2024
