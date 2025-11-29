#!/usr/bin/env python3
"""
AWS deployment script for NLP ML project
Handles CloudFormation stack creation and deployment
"""
import json
import boto3
import argparse
import logging
import time
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AWSDeploy:
    """Handle AWS CloudFormation deployments"""
    
    def __init__(self, region='us-east-1', profile=None):
        """
        Initialize AWS deployment handler
        
        Args:
            region: AWS region
            profile: AWS profile name
        """
        session_kwargs = {'region_name': region}
        if profile:
            session_kwargs['profile_name'] = profile
        
        self.session = boto3.Session(**session_kwargs)
        self.cf_client = self.session.client('cloudformation')
        self.s3_client = self.session.client('s3')
        self.codepipeline_client = self.session.client('codepipeline')
        self.codebuild_client = self.session.client('codebuild')
        self.region = region
    
    def create_stack(self, stack_name, template_body, parameters=None, tags=None):
        """
        Create CloudFormation stack
        
        Args:
            stack_name: Name of the stack
            template_body: CloudFormation template JSON
            parameters: Stack parameters
            tags: Stack tags
        
        Returns:
            str: Stack ID
        """
        logger.info(f"Creating CloudFormation stack: {stack_name}")
        
        params = {
            'StackName': stack_name,
            'TemplateBody': json.dumps(template_body) if isinstance(template_body, dict) else template_body,
            'Capabilities': ['CAPABILITY_NAMED_IAM'],
        }
        
        if parameters:
            params['Parameters'] = [
                {'ParameterKey': k, 'ParameterValue': str(v)}
                for k, v in parameters.items()
            ]
        
        if tags:
            params['Tags'] = [
                {'Key': k, 'Value': v}
                for k, v in tags.items()
            ]
        
        try:
            response = self.cf_client.create_stack(**params)
            stack_id = response['StackId']
            logger.info(f"Stack creation initiated: {stack_id}")
            
            # Wait for stack creation
            self.wait_for_stack(stack_name, 'CREATE_COMPLETE')
            
            logger.info(f"Stack {stack_name} created successfully")
            return stack_id
        
        except self.cf_client.exceptions.AlreadyExistsException:
            logger.warning(f"Stack {stack_name} already exists")
            return None
        except Exception as e:
            logger.error(f"Error creating stack: {e}")
            raise
    
    def update_stack(self, stack_name, template_body, parameters=None):
        """Update existing CloudFormation stack"""
        logger.info(f"Updating CloudFormation stack: {stack_name}")
        
        params = {
            'StackName': stack_name,
            'TemplateBody': json.dumps(template_body) if isinstance(template_body, dict) else template_body,
            'Capabilities': ['CAPABILITY_NAMED_IAM'],
        }
        
        if parameters:
            params['Parameters'] = [
                {'ParameterKey': k, 'ParameterValue': str(v)}
                for k, v in parameters.items()
            ]
        
        try:
            response = self.cf_client.update_stack(**params)
            stack_id = response['StackId']
            logger.info(f"Stack update initiated: {stack_id}")
            
            # Wait for stack update
            self.wait_for_stack(stack_name, 'UPDATE_COMPLETE')
            
            logger.info(f"Stack {stack_name} updated successfully")
            return stack_id
        except Exception as e:
            logger.error(f"Error updating stack: {e}")
            raise
    
    def delete_stack(self, stack_name):
        """Delete CloudFormation stack"""
        logger.info(f"Deleting CloudFormation stack: {stack_name}")
        
        try:
            self.cf_client.delete_stack(StackName=stack_name)
            self.wait_for_stack(stack_name, 'DELETE_COMPLETE')
            logger.info(f"Stack {stack_name} deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting stack: {e}")
            raise
    
    def wait_for_stack(self, stack_name, target_status, timeout=3600):
        """Wait for stack to reach target status"""
        waiter = self.cf_client.get_waiter('stack_create_complete')
        
        if 'UPDATE' in target_status:
            waiter = self.cf_client.get_waiter('stack_update_complete')
        elif 'DELETE' in target_status:
            waiter = self.cf_client.get_waiter('stack_delete_complete')
        
        try:
            waiter.wait(StackName=stack_name, WaiterConfig={'Delay': 30, 'MaxAttempts': timeout//30})
        except Exception as e:
            logger.error(f"Wait failed: {e}")
            raise
    
    def get_stack_outputs(self, stack_name):
        """Get CloudFormation stack outputs"""
        try:
            response = self.cf_client.describe_stacks(StackName=stack_name)
            stack = response['Stacks'][0]
            
            outputs = {}
            for output in stack.get('Outputs', []):
                outputs[output['OutputKey']] = output['OutputValue']
            
            return outputs
        except Exception as e:
            logger.error(f"Error getting stack outputs: {e}")
            raise
    
    def upload_data_to_s3(self, bucket_name, local_file, s3_key):
        """Upload file to S3"""
        logger.info(f"Uploading {local_file} to s3://{bucket_name}/{s3_key}")
        
        try:
            self.s3_client.upload_file(local_file, bucket_name, s3_key)
            logger.info("Upload completed")
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
    
    def trigger_pipeline(self, pipeline_name):
        """Trigger CodePipeline execution"""
        logger.info(f"Triggering pipeline: {pipeline_name}")
        
        try:
            response = self.codepipeline_client.start_pipeline_execution(
                name=pipeline_name
            )
            execution_id = response['pipelineExecutionId']
            logger.info(f"Pipeline execution started: {execution_id}")
            return execution_id
        except Exception as e:
            logger.error(f"Error triggering pipeline: {e}")
            raise


def load_cloudformation_template():
    """Load CloudFormation template"""
    try:
        from cloudformation_template import template
        return template
    except ImportError:
        logger.error("Could not import cloudformation_template module")
        return None


def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(description='Deploy NLP ML project to AWS')
    parser.add_argument('--action', choices=['create', 'update', 'delete'], default='create',
                       help='CloudFormation action')
    parser.add_argument('--stack-name', default='nlp-ml-project',
                       help='CloudFormation stack name')
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region')
    parser.add_argument('--profile', help='AWS profile name')
    parser.add_argument('--project-name', default='nlp-project',
                       help='Project name')
    parser.add_argument('--environment', default='dev',
                       help='Environment (dev, staging, prod)')
    
    args = parser.parse_args()
    
    # Initialize deployment
    deployer = AWSDeploy(region=args.region, profile=args.profile)
    
    # Load template
    template = load_cloudformation_template()
    if not template:
        logger.error("Failed to load CloudFormation template")
        sys.exit(1)
    
    # Prepare parameters
    parameters = {
        'ProjectName': args.project_name,
        'Environment': args.environment,
    }
    
    # Prepare tags
    tags = {
        'Project': args.project_name,
        'Environment': args.environment,
        'ManagedBy': 'CloudFormation'
    }
    
    try:
        if args.action == 'create':
            deployer.create_stack(args.stack_name, template, parameters, tags)
            outputs = deployer.get_stack_outputs(args.stack_name)
            logger.info(f"Stack outputs: {outputs}")
        
        elif args.action == 'update':
            deployer.update_stack(args.stack_name, template, parameters)
            outputs = deployer.get_stack_outputs(args.stack_name)
            logger.info(f"Stack outputs: {outputs}")
        
        elif args.action == 'delete':
            deployer.delete_stack(args.stack_name)
        
        logger.info(f"Deployment completed successfully")
    
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
