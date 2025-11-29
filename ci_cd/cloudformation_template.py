"""
AWS CloudFormation template for NLP project infrastructure
JSON template for IaC deployment
"""
template = {
    "AWSTemplateFormatVersion": "2010-09-09",
    "Description": "CloudFormation template for NLP ML project infrastructure",
    "Parameters": {
        "ProjectName": {
            "Type": "String",
            "Default": "nlp-project",
            "Description": "Name of the project"
        },
        "Environment": {
            "Type": "String",
            "Default": "dev",
            "AllowedValues": ["dev", "staging", "prod"],
            "Description": "Environment name"
        },
        "GitHubRepo": {
            "Type": "String",
            "Description": "GitHub repository URL"
        },
        "GitHubBranch": {
            "Type": "String",
            "Default": "main",
            "Description": "GitHub branch to track"
        }
    },
    "Resources": {
        # S3 Bucket for artifacts and data
        "ArtifactBucket": {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "BucketName": {"Fn::Sub": "${ProjectName}-artifacts-${AWS::AccountId}"},
                "VersioningConfiguration": {
                    "Status": "Enabled"
                },
                "PublicAccessBlockConfiguration": {
                    "BlockPublicAcls": True,
                    "BlockPublicPolicy": True,
                    "IgnorePublicAcls": True,
                    "RestrictPublicBuckets": True
                },
                "LifecycleConfiguration": {
                    "LifecycleRules": [
                        {
                            "Id": "DeleteOldArtifacts",
                            "Status": "Enabled",
                            "ExpirationInDays": 30,
                            "Prefix": "build-artifacts/"
                        }
                    ]
                }
            }
        },
        
        # S3 Bucket for data
        "DataBucket": {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "BucketName": {"Fn::Sub": "${ProjectName}-data-${AWS::AccountId}"},
                "VersioningConfiguration": {
                    "Status": "Enabled"
                }
            }
        },
        
        # IAM Role for CodeBuild
        "CodeBuildRole": {
            "Type": "AWS::IAM::Role",
            "Properties": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "codebuild.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                },
                "ManagedPolicyArns": [
                    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
                ],
                "Policies": [
                    {
                        "PolicyName": "CodeBuildPolicy",
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": [
                                        "logs:CreateLogGroup",
                                        "logs:CreateLogStream",
                                        "logs:PutLogEvents"
                                    ],
                                    "Resource": "*"
                                },
                                {
                                    "Effect": "Allow",
                                    "Action": [
                                        "s3:GetObject",
                                        "s3:PutObject",
                                        "s3:ListBucket"
                                    ],
                                    "Resource": [
                                        {"Fn::GetAtt": ["ArtifactBucket", "Arn"]},
                                        {"Fn::Sub": "${ArtifactBucket.Arn}/*"},
                                        {"Fn::GetAtt": ["DataBucket", "Arn"]},
                                        {"Fn::Sub": "${DataBucket.Arn}/*"}
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        },
        
        # IAM Role for CodePipeline
        "CodePipelineRole": {
            "Type": "AWS::IAM::Role",
            "Properties": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "codepipeline.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                },
                "Policies": [
                    {
                        "PolicyName": "CodePipelinePolicy",
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": [
                                        "s3:GetObject",
                                        "s3:PutObject",
                                        "s3:ListBucket"
                                    ],
                                    "Resource": [
                                        {"Fn::GetAtt": ["ArtifactBucket", "Arn"]},
                                        {"Fn::Sub": "${ArtifactBucket.Arn}/*"}
                                    ]
                                },
                                {
                                    "Effect": "Allow",
                                    "Action": [
                                        "codebuild:BatchGetBuilds",
                                        "codebuild:BatchGetBuildBatches",
                                        "codebuild:StartBuild"
                                    ],
                                    "Resource": "*"
                                }
                            ]
                        }
                    }
                ]
            }
        },
        
        # CloudWatch Log Group for CodeBuild
        "BuildLogGroup": {
            "Type": "AWS::Logs::LogGroup",
            "Properties": {
                "LogGroupName": {"Fn::Sub": "/aws/codebuild/${ProjectName}-build"},
                "RetentionInDays": 30
            }
        },
        
        # CodeBuild Project for training
        "TrainingBuildProject": {
            "Type": "AWS::CodeBuild::Project",
            "Properties": {
                "Name": {"Fn::Sub": "${ProjectName}-training-build"},
                "ServiceRole": {"Fn::GetAtt": ["CodeBuildRole", "Arn"]},
                "Artifacts": {
                    "Type": "CODEPIPELINE"
                },
                "Environment": {
                    "Type": "LINUX_CONTAINER",
                    "ComputeType": "BUILD_GENERAL1_MEDIUM",
                    "Image": "aws/codebuild/standard:7.0",
                    "EnvironmentVariables": [
                        {
                            "Name": "AWS_S3_BUCKET",
                            "Value": {"Ref": "ArtifactBucket"}
                        },
                        {
                            "Name": "AWS_ACCOUNT_ID",
                            "Value": {"Ref": "AWS::AccountId"}
                        }
                    ]
                },
                "Source": {
                    "Type": "CODEPIPELINE",
                    "BuildSpec": "ci_cd/buildspec.yml"
                },
                "LogsConfig": {
                    "CloudWatchLogs": {
                        "Status": "ENABLED",
                        "GroupName": {"Ref": "BuildLogGroup"}
                    }
                }
            }
        }
    },
    "Outputs": {
        "ArtifactBucketName": {
            "Description": "S3 bucket for artifacts",
            "Value": {"Ref": "ArtifactBucket"}
        },
        "DataBucketName": {
            "Description": "S3 bucket for data",
            "Value": {"Ref": "DataBucket"}
        },
        "CodeBuildProjectName": {
            "Description": "CodeBuild project name",
            "Value": {"Ref": "TrainingBuildProject"}
        }
    }
}

# Export as JSON
import json

if __name__ == "__main__":
    print(json.dumps(template, indent=2))
