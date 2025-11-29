# 🌐 AWS DEPLOYMENT - ONLY 5 SIMPLE STEPS

**Time needed:** 30-45 minutes  
**Cost:** Free (within AWS Free Tier)  
**Difficulty:** Beginner-friendly

---

## ⚠️ PREREQUISITE: Test Locally First!

Before AWS, make sure it works locally:

```bash
# 1. Create sample data
python create_sample_data.py

# 2. Train model (wait 5 min)
python train.py

# 3. Run web app locally
streamlit run streamlit_app.py

# If all works, continue to AWS...
```

---

## 🔑 STEP 1: Set Up AWS Account (10 minutes)

### 1a. Create AWS Account
```
1. Go to https://aws.amazon.com
2. Click "Create an AWS Account"
3. Fill in email, password, and account name
4. Add billing information
5. Verify phone number
6. Verify email
7. Choose support plan (Free tier is fine)
```

✅ **You now have an AWS account!**

### 1b. Create IAM User
```
AWS Console → IAM → Users → Create user
- Username: nlp-deployer
- Check: Enable console access
→ Next → Attach policies directly
- Search: AdministratorAccess
- Select it
→ Create user
→ Copy Access Key ID and Secret Access Key
```

✅ **Save these! You'll need them:**
- Access Key ID: `_________________________`
- Secret Access Key: `_________________________`
- AWS Account ID (12 digits): `_________________________`

### 1c. Install AWS CLI
```bash
pip install awscli
aws --version
```

### 1d. Configure AWS
```bash
aws configure
```

When prompted:
```
AWS Access Key ID: [paste Access Key ID]
AWS Secret Access Key: [paste Secret Access Key]
Default region name: us-east-1
Default output format: json
```

**Test it:**
```bash
aws sts get-caller-identity
```

You should see your account info.

✅ **AWS is configured!**

---

## 💾 STEP 2: Create S3 Buckets (5 minutes)

### 2a. Create Artifact Bucket
```bash
# Replace 123456 with your AWS Account ID (12 digits)
aws s3api create-bucket \
    --bucket nlp-artifacts-123456 \
    --region us-east-1
```

### 2b. Create Data Bucket
```bash
aws s3api create-bucket \
    --bucket nlp-data-123456 \
    --region us-east-1
```

### 2c. Upload Training Data
```bash
aws s3 cp data/raw/training_data.csv \
    s3://nlp-data-123456/data/training_data.csv
```

### 2d. Verify Buckets
```bash
aws s3 ls
```

You should see both buckets listed.

✅ **S3 buckets created!**

---

## 🏗️ STEP 3: Deploy Infrastructure (10 minutes)

### 3a. Deploy with CloudFormation
```bash
python ci_cd/aws_deploy.py \
    --action create \
    --stack-name nlp-ml-stack \
    --region us-east-1 \
    --environment dev
```

### 3b. Wait for Deployment
```bash
# Check status (wait for CREATE_COMPLETE)
aws cloudformation describe-stacks \
    --stack-name nlp-ml-stack \
    --query 'Stacks[0].StackStatus'
```

⏳ **Wait 2-5 minutes...**

When you see `CREATE_COMPLETE`, continue.

### 3c. Get Outputs
```bash
aws cloudformation describe-stacks \
    --stack-name nlp-ml-stack \
    --query 'Stacks[0].Outputs'
```

✅ **Infrastructure deployed!**

---

## 🔄 STEP 4: Set Up CI/CD Pipeline (10 minutes)

### 4a. Push to GitHub
```bash
# If not already done
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/nlp-ml-project.git
git branch -M main
git push -u origin main
```

### 4b. Create Pipeline in AWS Console
1. Go to **AWS Console → CodePipeline**
2. Click **Create pipeline**
3. **Pipeline settings:**
   - Name: `nlp-training-pipeline`
   - Service role: Create new
4. Click **Next**

### 4c. Add Source Stage
- Provider: GitHub
- Click **Connect to GitHub**
- Authorize AWS
- Repository: `nlp-ml-project`
- Branch: `main`
- Click **Next**

### 4d. Add Build Stage
- Provider: **AWS CodeBuild**
- Click **Create project**
  - Name: `nlp-model-training`
  - OS: Amazon Linux 2
  - Runtime: Standard
  - Buildspec: `ci_cd/buildspec.yml`
  - Environment variables:
    - Name: `AWS_S3_BUCKET`
    - Value: `nlp-artifacts-123456`
- Click **Create build project**
- Click **Next**

### 4e. Skip Deploy Stage
- Click **Skip deploy stage**
- Click **Create pipeline**

⏳ **Wait 5-10 minutes for first run...**

### 4f. Verify Pipeline
1. Go to AWS Console → CodePipeline
2. Watch pipeline run
3. Wait for all stages to show **Succeeded** (green)

✅ **Pipeline created and running!**

---

## 🌐 STEP 5: Deploy Web App (10 minutes)

### Option A: EASIEST - Streamlit Cloud

**Pros:** No AWS infrastructure needed, free hosting  
**Cons:** Requires Streamlit Cloud account

```bash
# 1. Ensure code is on GitHub
git push origin main

# 2. Go to https://streamlit.io/cloud

# 3. Click "New app"

# 4. Connect GitHub account and authorize

# 5. Select:
   - Repository: nlp-ml-project
   - Branch: main
   - Main file path: streamlit_app.py

# 6. Click "Deploy"

# Your app URL: https://[your-app-name].streamlit.app
```

✅ **App deployed! Share the URL!**

---

### Option B: FLEXIBLE - AWS EC2

**Pros:** Full control, more customizable  
**Cons:** Manage server yourself

#### Step 1: Launch EC2 Instance
```
AWS Console → EC2 → Instances → Launch Instance
- AMI: Ubuntu 22.04 LTS (Free tier)
- Instance type: t2.micro (Free tier)
- Key pair: Create new → nlp-app-key.pem (Download!)
- Security: Allow SSH (22), HTTP (80), HTTPS (443)
- Storage: 20 GB
- Launch
```

Save your **Public IP**: `__________________________`

#### Step 2: Connect to Instance
```bash
chmod 400 nlp-app-key.pem
ssh -i nlp-app-key.pem ubuntu@YOUR_PUBLIC_IP
```

#### Step 3: Install & Setup
```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv git

git clone https://github.com/YOUR_USERNAME/nlp-ml-project.git
cd nlp-ml-project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download models from S3
mkdir -p models
aws s3 cp s3://nlp-artifacts-123456/models/ models/ --recursive
```

#### Step 4: Run Streamlit
```bash
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0
```

#### Step 5: Access App
Open browser: `http://YOUR_PUBLIC_IP:8501`

✅ **App running on AWS!**

---

### Option C: SIMPLEST AWS - AppRunner

**Pros:** No server management, auto-scaling  
**Cons:** Slightly higher cost than EC2

#### Step 1: Create Docker Image
Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

#### Step 2: Push to ECR
```bash
# Create repository
aws ecr create-repository --repository-name nlp-app

# Login
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    123456.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t nlp-app .
docker tag nlp-app:latest \
    123456.dkr.ecr.us-east-1.amazonaws.com/nlp-app:latest
docker push \
    123456.dkr.ecr.us-east-1.amazonaws.com/nlp-app:latest
```

#### Step 3: Deploy with AppRunner
```
AWS Console → App Runner → Create service
- Source: ECR
- Image URI: 123456.dkr.ecr.us-east-1.amazonaws.com/nlp-app:latest
- Service name: nlp-inference-app
- Port: 8501
- Deploy
```

Wait for "Running" status.

Your app URL will be displayed!

✅ **App deployed on AppRunner!**

---

## ✅ VERIFICATION CHECKLIST

After all 5 steps:

- [ ] AWS account created
- [ ] IAM user created with credentials
- [ ] AWS CLI configured
- [ ] S3 buckets created (artifacts & data)
- [ ] Training data uploaded to S3
- [ ] CloudFormation stack shows `CREATE_COMPLETE`
- [ ] CodePipeline created
- [ ] First build succeeded
- [ ] Models uploaded to S3
- [ ] Streamlit app deployed and accessible
- [ ] Can make predictions on web app

---

## 🎯 HOW IT ALL WORKS

```
You Push Code to GitHub
        ↓
CodePipeline Detects Change
        ↓
CodeBuild Runs buildspec.yml
        ↓
train.py Runs Automatically
        ↓
Model Trains in AWS
        ↓
Models Uploaded to S3
        ↓
Streamlit App Uses Models
        ↓
You Get Predictions on Web!
```

---

## 🔄 TO UPDATE MODEL

**Whenever you want to retrain:**

```bash
# 1. Update data
aws s3 cp new_data.csv s3://nlp-data-123456/

# 2. Push changes to GitHub (if any)
git add .
git commit -m "Update data"
git push origin main

# 3. Pipeline automatically runs!

# 4. New model automatically deployed
```

✅ **Everything updates automatically!**

---

## 📊 EXPECTED COSTS (Free Tier)

| Service | Free Tier | Monthly Cost |
|---------|-----------|-------------|
| S3 | 5 GB storage | $0 |
| CodeBuild | 100 build min | $0 |
| CodePipeline | 1 active pipeline | $0 |
| EC2 | 750 hours/month | $0 |
| AppRunner | 1 GB/month free | $0-5 |
| **Total** | | **$0-5/month** |

✅ **Basically free for small projects!**

---

## 🚨 TROUBLESHOOTING

### Error: "Access Denied"
**Fix:**
```bash
aws configure
# Re-enter credentials
```

### Error: "Bucket already exists"
**Fix:** Use different bucket name with timestamp

### Error: "CodeBuild failed"
**Fix:** 
1. AWS Console → CodeBuild → Logs
2. Read error message
3. Update buildspec.yml
4. Push to GitHub (auto-retries)

### Error: "Can't connect to EC2"
**Fix:**
```bash
chmod 400 nlp-app-key.pem
# Check security group allows SSH
```

---

## 📞 QUICK REFERENCE

**Deploy infrastructure:**
```bash
python ci_cd/aws_deploy.py --action create --stack-name nlp-ml-stack
```

**Upload data:**
```bash
aws s3 cp data.csv s3://nlp-data-123456/
```

**Check pipeline:**
```bash
aws codepipeline get-pipeline-state --name nlp-training-pipeline
```

**View logs:**
```bash
aws logs tail /aws/codebuild/nlp-model-training --follow
```

**Update stack:**
```bash
python ci_cd/aws_deploy.py --action update --stack-name nlp-ml-stack
```

**Delete stack:**
```bash
python ci_cd/aws_deploy.py --action delete --stack-name nlp-ml-stack
```

---

## 🎉 CONGRATULATIONS!

Your NLP ML project is now on AWS with:

✅ Automated training pipeline  
✅ Auto-retraining on data changes  
✅ Web interface for predictions  
✅ Production-ready infrastructure  
✅ Cost-effective (free tier)  

---

## 📚 NEXT STEPS

1. **Test predictions** on your web app
2. **Monitor pipeline** after each update
3. **Share app URL** with others
4. **Update data** to retrain models
5. **Scale up** as needed

---

## 🆘 NEED HELP?

1. See **AWS_DEPLOYMENT_CHECKLIST.md** for detailed steps
2. See **AWS_DEPLOYMENT_GUIDE.md** for detailed guide
3. See **FILE_GUIDE.md** to understand all files
4. Run **validate_data.py** to check setup

---

**Status:** ✅ Ready to Deploy  
**Created:** November 2024  
**Version:** 1.0.0
