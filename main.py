"""
Copyright 2016 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with
the License. A copy of the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

import boto3
from botocore.config import Config
from botocore.signers import RequestSigner
from dateutil.tz import tzutc

DEFAULT_PUSH_DATE = datetime.now(tzutc())
DEFAULT_IMAGES_TO_KEEP = 100
DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 60
DEFAULT_MAX_ATTEMPTS = 10

@dataclass
class Inputs:
    dryrun: bool | None = None
    region: str | None = None
    repo_name_regex: re.Pattern | None = None
    images_to_keep: int | None = None
    older_than: datetime | None = None
    protect_tags_regex: re.Pattern | None = None
    protect_latest: bool | None = None
    check_ecs: bool | None = None
    check_eks: bool | None = None
    connect_timeout: int | None = None
    read_timeout: int | None = None
    max_attempts: int | None = None
    logger: logging.Logger | None = None

    def __post_init__(self):
        if self.logger is None:
            self.logger = get_default_logger()

        if not isinstance(self.dryrun, bool):
            if isinstance(self.dryrun, str):
                self.dryrun = (self.dryrun or "false").lower() != 'false'
            else:
                raise Exception(f"Unexpected {type(self.dryrun)=}")

        self.region = self.region or "None"
        if not isinstance(self.region, str):
            raise Exception(f"Unexpected {type(self.region)=}")

        if not isinstance(self.repo_name_regex, re.Pattern):
            if not self.repo_name_regex:
                self.repo_name_regex = None
            elif isinstance(self.repo_name_regex, str):
                self.repo_name_regex = re.compile(self.repo_name_regex)
            else:
                raise Exception(f"Unexpected {type(self.repo_name_regex)=}")

        if not isinstance(self.images_to_keep, int):
            if isinstance(self.images_to_keep, str):
                self.images_to_keep = int(self.images_to_keep or DEFAULT_IMAGES_TO_KEEP)
            else:
                raise Exception(f"Unexpected {type(self.images_to_keep)=}")

        if not isinstance(self.older_than, datetime):
            if not self.older_than:
                self.older_than = None
            elif isinstance(self.older_than, (str, float, int)):
                self.older_than = datetime.now(tzutc()) - timedelta(days=float(self.older_than))
            else:
                raise Exception(f"Unexpected {type(self.older_than)=}")

        if not isinstance(self.protect_latest, bool):
            if isinstance(self.protect_latest, str):
                self.protect_latest = (self.protect_latest or "false").lower() != 'false'
            else:
                raise Exception(f"Unexpected {type(self.protect_latest)=}")

        if not isinstance(self.protect_tags_regex, re.Pattern):
            if not self.protect_tags_regex:
                self.protect_tags_regex = None
            elif isinstance(self.protect_tags_regex, str):
                self.protect_tags_regex = re.compile(self.protect_tags_regex)
            else:
                raise Exception(f"Unexpected {type(self.protect_tags_regex)=}")

        if not isinstance(self.check_ecs, bool):
            if isinstance(self.check_ecs, str):
                self.check_ecs = (self.check_ecs or "false").lower() != 'false'
            else:
                raise Exception(f"Unexpected {type(self.check_ecs)=}")

        if not isinstance(self.check_eks, bool):
            if isinstance(self.check_eks, str):
                self.check_eks = (self.dryrun or "false").lower() != 'false'
            else:
                raise Exception(f"Unexpected {type(self.check_eks)=}")

        self.connect_timeout = int(self.connect_timeout or DEFAULT_CONNECT_TIMEOUT)
        self.read_timeout = int(self.read_timeout or DEFAULT_READ_TIMEOUT)
        self.max_attempts = int(self.max_attempts or DEFAULT_MAX_ATTEMPTS)

    @classmethod
    def from_env(cls):
        return cls(
            dryrun=os.environ.get('DRYRUN'),
            region=os.environ.get('REGION'),
            repo_name_regex=os.environ.get('REPO_NAME_REGEX'),
            images_to_keep=os.environ.get('IMAGES_TO_KEEP'),
            older_than=os.environ.get('OLDER_THAN'),
            protect_tags_regex=os.environ.get('PROTECT_TAGS_REGEX'),
            protect_latest=os.environ.get('PROTECT_LATEST'),
            check_ecs=os.environ.get('CHECK_ECS'),
            check_eks=os.environ.get('CHECK_EKS'),
            connect_timeout=os.environ.get('CONNECT_TIMEOUT'),
            read_timeout=os.environ.get('READ_TIMEOUT'),
            max_attempts=os.environ.get('MAX_ATTEMPTS'),
        )

    @classmethod
    def from_parser(cls):
        parser = argparse.ArgumentParser(
            description='Deletes stale ECR images',
            epilog='Deletion logic: In each ECR repository that contains more than `images_to_keep` non running images '
                   '(running images are images that currently deployed on a container), '
                   'Iterate through a list of images, sorted from oldest to newest '
                   '(image date: if an image was pulled - last pull date, otherwise push date), '
                   'Mark images for deletion, until less then `images_to_keep` are left, '
                   'or until iterated through the entire list. '
                   'An image is marked for deletion if it is older than `older_than`; '
                   'AND is not tagged `latest`, AND does not have has any tags that matches `protect_tags_regex`. '
                   'Note that if not enough images are marked for deletion, '
                   'more than `images_to_keep` may be left untouched.')

        parser.add_argument('-no-dryrun',
                            help="Don't just prints the repository to be deleted, actually delete them",
                            dest="dryrun", action='store_false')
        parser.add_argument('-region', help='ECR/ECS region', default=os.environ.get("AWS_REGION"),
                            action='store')
        parser.add_argument('-repo_name_regex', help='Regex of repo names to search', default=None,
                            action='store')
        parser.add_argument('-images_to_keep', help='Number of image tags to keep',
                            default=DEFAULT_IMAGES_TO_KEEP, action='store')
        parser.add_argument('-older_than',
                            help='Only delete images older than a specified amount of days',
                            default=None, type=float, action='store')
        parser.add_argument('-protect_tags_regex', help='Regex of tag names to protect (not delete)',
                            default=None, action='store')
        parser.add_argument('-unprotect-latest', help='Allow deletion images with `latest` tag',
                            dest="protect_latest", action='store_false')
        parser.add_argument('-no-ecs', help="Don't search ECS for running images",
                            dest="check_ecs", action='store_false')
        parser.add_argument('-no-eks', help="Don't search EKS for running images",
                            dest="check_eks", action='store_false')

        parser.add_argument('-connect_timeout', type=int, default=DEFAULT_CONNECT_TIMEOUT,
                            help="ECS connection timeout (in seconds)", action='store')
        parser.add_argument('-read_timeout', type=int, default=DEFAULT_READ_TIMEOUT,
                            help="ECS read timeout (in seconds)", action='store')
        parser.add_argument('-max_attempts', type=int, default=DEFAULT_MAX_ATTEMPTS,
                            help="ECS maximum number of attempts", action='store')

        args = parser.parse_args()
        return cls(
            dryrun=args.dryrun,
            region=args.region,
            repo_name_regex=args.repo_name_regex,
            images_to_keep=args.images_to_keep,
            older_than=args.older_than,
            protect_tags_regex=args.protect_tags_regex,
            protect_latest=args.protect_latest,
            check_ecs=args.check_ecs,
            check_eks=args.check_eks,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            max_attempts=args.max_attempts,
        )


def handler(event, context):
    inputs = Inputs.from_env()
    logger = inputs.logger
    try:
        body = main(inputs)
        return {"statusCode": 200, "body": json.dumps(body)}
    except Exception as e:
        logger.error(str(e))
        return {"statusCode": 500, "body": str(e)}


def main(inputs: Inputs):
    logger = inputs.logger
    logger.info(f"Starting ECR cleanup {str(inputs)}")
    regions = get_regions(logger) if inputs.region is None or inputs.region == "None" else [inputs.region]
    deleted_digests = []
    for region in regions:
        inputs.region = region
        try:
            deleted_digests += discover_delete_images(inputs=inputs)
        except Exception as e:
            e.add_note(f"Error while deleting images in region {region}")
            logger.error(str(e))
            raise
    logger.info(f"ECR cleanup across all region is completed! {len(deleted_digests)} images "
                f"were {f'pretending to be' if inputs.dryrun else ''} deleted.")
    return deleted_digests


def get_regions(logger: logging.Logger):
    """
    Source: Retrieves available AWS regions dynamically using the describe_regions API from the EC2 client.
    How It Works:
    Calls ec2_client.describe_regions() to fetch a list of regions.
    Extracts the RegionName from the response and uses it to call discover_delete_images for each region.
    """
    ec2_client = boto3.client('ec2')
    available_regions = ec2_client.describe_regions()['Regions']
    regions = [region['RegionName'] for region in available_regions]
    logger.info(f"Found {len(regions)} regions in ECR: {regions}.")
    return regions


def get_image_date(image_detail: dict, logger: logging.Logger) -> datetime:
    result = image_detail.get('lastRecordedPullTime') or image_detail.get('imagePushedAt')
    if not result:
        logger.error(f"Image {image_detail} does not have a 'imagePushedAt' or 'lastRecordedPullTime' key, "
                       f"just to be safe, assuming it's fresh.")
        return DEFAULT_PUSH_DATE
    return result


def sort_image_details_by_date(tagged_images, title: str, logger: logging.Logger):
    tagged_images.sort(key=lambda k: get_image_date(k, logger=logger), reverse=True)
    oldest_image = tagged_images[-1]
    newest_image = tagged_images[0]
    oldest_image_date = get_image_date(oldest_image, logger=logger)
    newest_image_date = get_image_date(newest_image, logger=logger)
    logger.info(f"{title}: images dates are dated between {oldest_image_date} and {newest_image_date}.")
    logger.info(f"{title}: Oldest image: {oldest_image}")
    logger.info(f"{title}: Newest image: {newest_image}")


def discover_delete_images(inputs: Inputs):
    logger = inputs.logger
    region_name = inputs.region
    logger.info(f"Discovering images in region {region_name}...")
    ecr_client = boto3.client('ecr', region_name=region_name)

    repositories = []
    describe_repo_paginator = ecr_client.get_paginator('describe_repositories')
    for describe_repo_response in describe_repo_paginator.paginate():
        for repo in describe_repo_response['repositories']:
            repositories.append(repo)

    logger.info(f"Found {len(repositories)} repos in region {region_name}...")
    logger.info([repo['repositoryUri'] for repo in repositories])
    if not inputs.repo_name_regex:
        logger.info("Processing all repos in region...")
    else:
        repositories = [repo for repo in repositories if inputs.repo_name_regex.match(repo['repositoryName'])]
        logger.info(f"Processing {len(repositories)} repos in region {region_name} "
                    f"(After matching repo names by regex {inputs.repo_name_regex})...")
        logger.info([repo['repositoryUri'] for repo in repositories])

    if not repositories:
        return []

    # ecs_client = boto3.client('ecs', region_name=region_name)
    ecs_client = boto3.client('ecs', region_name=region_name,
                              config=Config(
                                  connect_timeout=inputs.connect_timeout,
                                  read_timeout=inputs.read_timeout,
                                  retries={'max_attempts': inputs.max_attempts})
                              )

    running_containers = get_running_containers(ecs_client, inputs=inputs)

    # example of `image`
    # {
    #     "registryId": "123456789012",
    #     "repositoryName": "my-repo",
    #     "imageDigest": "sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
    #     "imageTags": ["latest", "v1.0.0"],
    #     "imagePushedAt": "2025-01-01T12:00:00Z",
    #     "imageSizeInBytes": 12345678,
    #     "lastRecordedPullTime": "2025-01-10T15:30:00Z",
    #     "artifactMediaType": "application/vnd.docker.container.image.v1+json"
    # }
    # Explanation of Fields
    # registryId: The AWS account ID associated with the image.
    # repositoryName: The name of the ECR repository where the image is stored.
    # imageDigest: A unique identifier for the image, derived from the image's contents (SHA-256 hash).
    # imageTags: A list of tags associated with the image, e.g., "latest", "v1.0.0".
    # If the image is untagged, this field is absent.
    # imagePushedAt: The timestamp of when the image was pushed to the repository.
    # imageSizeInBytes: The size of the image in bytes.
    # lastRecordedPullTime: The timestamp of the last time the image was pulled from the repository.
    # This field may be null if the image has never been pulled.
    # artifactMediaType: The media type of the image artifact.

    if not running_containers:
        logger.info(f"No images are running in region {region_name}.")
    else:
        logger.info(f"{len(running_containers)} images are running in region {region_name}:")
        for image_detail in running_containers:
            logger.info(image_detail)

    deleted_digests = []
    for repository in repositories:
        logger.info("------------------------")
        logger.info(f"Starting with repository: {repository['repositoryUri']}")
        delete_images_details = []
        tagged_images = []
        untagged_digests = set()

        describe_image_paginator = ecr_client.get_paginator('describe_images')
        for describe_image_response in describe_image_paginator.paginate(
                registryId=repository['registryId'],
                repositoryName=repository['repositoryName']):
            for image_detail in describe_image_response['imageDetails']:
                if image_detail.get('imageTags'):
                    tagged_images.append(image_detail)
                else:
                    untagged_digests.add(image_detail['imageDigest'])

        total_images_in_repo = len(tagged_images) + len(untagged_digests)
        logger.info(f"{repository['repositoryUri']}: found: {total_images_in_repo} images "
                    f"({len(tagged_images)} tagged + {len(untagged_digests)} untagged).")

        digests_to_delete_count = total_images_in_repo - inputs.images_to_keep
        if digests_to_delete_count <= 0 and len(untagged_digests) == 0:
            logger.info(f"No images to delete, as number of images found "
                        f"is no more than {inputs.images_to_keep=} and there are no untagged images.")
            continue

        sort_image_details_by_date(tagged_images, title="tagged", logger=logger)

        # Get ImageDigest from ImageURL for running images. Do this for every repository
        running_digests, running_images_details = get_running_digests(
            running_containers, repository, tagged_images, logger=logger)
        if running_digests:
            sort_image_details_by_date(running_images_details, title="running", logger=logger)
            logger.info(f"{repository['repositoryUri']}: "
                        f"found {len(running_digests)} running images, these won't be deleted:")
            for image_details in running_images_details:
                logger.info(image_details)
        else:
            logger.info(f"{repository['repositoryUri']}: found 0 running images.")

        delete_digests = untagged_digests - running_digests
        untagged_running_digests = untagged_digests - delete_digests
        if untagged_running_digests:
            logger.warning(f"{repository['repositoryUri']}: "
                           f"found {len(untagged_running_digests)} UNTAGGED RUNNING images.")

        delete_images_dict = defaultdict(list)
        for image_detail in tagged_images:
            if len(delete_digests) >= digests_to_delete_count:
                break
            digest = image_detail['imageDigest']
            if digest in running_digests:
                # don't delete running images
                continue
            image_date = get_image_date(image_detail, logger=logger)
            if inputs.older_than is not None and image_date > inputs.older_than:
                # image is too new to be deleted
                continue
            image_tags = image_detail['imageTags']
            del_image = True
            for tag in image_tags:
                if ((not inputs.protect_latest and "latest" in tag) or
                    (inputs.protect_tags_regex is not None and inputs.protect_tags_regex.search(tag) is not None)):
                    # this image has a protected tag
                    del_image = False
                    break
            if not del_image:
                continue
            delete_digests.add(digest)
            delete_images_details.append({
                # given digest can have multiple tags that point to the same image
                "imageUrl": [repository['repositoryUri'] + ":" + tag for tag in image_tags],
                "imagePushedAt": image_detail.get("imagePushedAt"),
                "lastRecordedPullTime": image_detail.get('lastRecordedPullTime'),
                "imageDigest": digest,
            })
            delete_images_dict[repository['repositoryUri']].extend(image_tags)

        if delete_digests:
            logger.info(f"Number of images to be deleted: {len(delete_digests)}")
            for uri, tags in delete_images_dict.items():
                logger.info(f"Image tags to delete in {uri} ({len(tags)}): {tags}")
            delete_images(
                inputs,
                ecr_client,
                delete_digests,
                delete_images_details,
                repository['registryId'],
                repository['repositoryName'],
            )
            deleted_digests.extend(delete_digests)
        else:
            logger.info(f"Nothing to delete in repository: {repository['repositoryName']}")
    logger.info(f"ECR cleanup in region {region_name} is completed! {len(deleted_digests)} images "
                f"were {f'pretending to be ' if inputs.dryrun else ''}deleted.")
    return deleted_digests


def get_running_containers(ecs_client, inputs: Inputs):
    logger = inputs.logger
    logger.info('Getting running containers...')

    if inputs.check_ecs:
        running_containers_ecs = get_running_containers_from_ecs(ecs_client, logger)
        logger.info(f"{len(running_containers_ecs)} running containers "
                    f"from ECS tasks: {sorted(running_containers_ecs)}...")
    else:
        running_containers_ecs = []
        logger.info('Skip checking running containers on ECS...')

    if inputs.check_eks:
        running_containers_eks = get_running_containers_from_eks(logger)
        logger.info(f"{len(running_containers_eks)} running containers "
                    f"from EKS: {sorted(running_containers_eks)}...")
    else:
        running_containers_eks = []
        logger.info('Skip checking running containers on EKS...')

    running_containers_ecs = sorted(running_containers_ecs | running_containers_eks)
    logger.info(f"{len(running_containers_ecs)} total running containers: {running_containers_ecs}...")
    return running_containers_ecs


def get_running_containers_from_ecs(ecs_client, logger: logging.Logger):
    running_containers = set()
    list_clusters_paginator = ecs_client.get_paginator('list_clusters')
    for response_clusters_list_paginator in list_clusters_paginator.paginate():
        for cluster_arn in response_clusters_list_paginator['clusterArns']:
            logger.debug(f"Processing ECS cluster {cluster_arn}...")

            list_tasks_paginator = ecs_client.get_paginator('list_tasks')
            for list_tasks_response in list_tasks_paginator.paginate(cluster=cluster_arn, desiredStatus='RUNNING'):
                if list_tasks_response['taskArns']:
                    describe_tasks_list = ecs_client.describe_tasks(
                        cluster=cluster_arn,
                        tasks=list_tasks_response['taskArns']
                    )
                    for tasks_list in describe_tasks_list['tasks']:
                        if tasks_list['taskDefinitionArn'] is not None:
                            response = ecs_client.describe_task_definition(
                                taskDefinition=tasks_list['taskDefinitionArn']
                            )
                            for container in response['taskDefinition']['containerDefinitions']:
                                if '.dkr.ecr.' in container['image'] and ":" in container['image']:
                                    running_containers.add(container['image'])

            # List services in the cluster
            services_response = ecs_client.list_services(cluster=cluster_arn)
            service_arns = services_response['serviceArns']
            if not service_arns:
                continue

            # Describe the service to get details
            services_details = ecs_client.describe_services(
                cluster=cluster_arn,
                services=service_arns
            )

            for service in services_details['services']:
                # Get task definition for the service
                task_definition_arn = service['taskDefinition']
                task_definition_response = ecs_client.describe_task_definition(taskDefinition=task_definition_arn)
                task_definition = task_definition_response['taskDefinition']

                # Extract container images from task definition
                container_definitions = task_definition['containerDefinitions']
                for container_definition in container_definitions:
                    container_image = container_definition['image']
                    logger.info(f"Service: {service['serviceName']}, Container Image: {container_image}")
                    running_containers.add(container_image)

    return running_containers


def get_k8s_config(logger: logging.Logger, cluster_name):
    # From Lambda function list pods in EKS cluster
    # https://github.com/aws-samples/amazon-eks-kubernetes-api-aws-lambda/blob/3d6c01864c2db120caea99343e6bbaf6c901a146/function/lambda_function.py

    STS_TOKEN_EXPIRES_IN = 60
    session = boto3.session.Session()
    sts = session.client('sts')
    service_id = sts.meta.service_model.service_id
    eks = boto3.client('eks')
    cluster_cache = {}

    def get_cluster_info():
        "Retrieve cluster endpoint and certificate"
        cluster_info = eks.describe_cluster(name=cluster_name)
        endpoint = cluster_info['cluster']['endpoint']
        cert_authority = cluster_info['cluster']['certificateAuthority']['data']
        cluster_info = {
            "endpoint": endpoint,
            "ca": cert_authority
        }
        return cluster_info

    def get_bearer_token():
        """Create authentication token"""
        signer = RequestSigner(
            service_id,
            session.region_name,
            'sts',
            'v4',
            session.get_credentials(),
            session.events
        )

        params = {
            'method': 'GET',
            'url': 'https://sts.{}.amazonaws.com/'
                   '?Action=GetCallerIdentity&Version=2011-06-15'.format(session.region_name),
            'body': {},
            'headers': {
                'x-k8s-aws-id': cluster_name
            },
            'context': {}
        }

        signed_url = signer.generate_presigned_url(
            params,
            region_name=session.region_name,
            expires_in=STS_TOKEN_EXPIRES_IN,
            operation_name=''
        )
        base64_url = base64.urlsafe_b64encode(signed_url.encode('utf-8')).decode('utf-8')

        # remove any base64 encoding padding:
        return 'k8s-aws-v1.' + re.sub(r'=*', '', base64_url)

    if cluster_name in cluster_cache:
        cluster = cluster_cache[cluster_name]
    else:
        # not present in cache retrieve cluster info from EKS service
        cluster = get_cluster_info()
        # store in cache for execution environment resuse
        cluster_cache[cluster_name] = cluster

    kubeconfig = {
        'apiVersion': 'v1',
        'clusters': [{
            'name': 'cluster1',
            'cluster': {
                'certificate-authority-data': cluster["ca"],
                'server': cluster["endpoint"]}
        }],
        'contexts': [{'name': 'context1', 'context': {'cluster': 'cluster1', "user": "user1"}}],
        'current-context': 'context1',
        'kind': 'Config',
        'preferences': {},
        'users': [{'name': 'user1', "user": {'token': get_bearer_token()}}]
    }

    return kubeconfig
    # logger.info(f"load kube config from dict: {kubeconfig=}")
    # config.load_kube_config_from_dict(config_dict=kubeconfig)
    # v1_api = client.CoreV1Api()  # api_client
    # ret = v1_api.list_namespaced_pod("default")
    # return f"There are {len(ret.items)} pods in the default namespace."


def get_running_containers_from_eks(logger: logging.Logger):
    from kubernetes import client, config

    try:
        # Initialize AWS EKS client
        eks_client = boto3.client("eks")

        # Fetch the list of EKS clusters
        response = eks_client.list_clusters()
        cluster_names = response.get("clusters", [])

        logger.info(f"Found {len(cluster_names)} EKS clusters: {cluster_names}")
    except Exception as e:
        e.add_note(f"Error fetching EKS clusters")
        logger.error(str(e))
        raise

    all_running_containers = set()
    for cluster_name in cluster_names:
        logger.info(f"Processing EKS cluster: {cluster_name}")
        running_containers = set()
        try:
            # Load in-cluster config or local kubeconfig

            # Load kubeconfig (ensures the correct context is used)
            # config.load_kube_config()  # For local development
            # config.load_kube_config(config_file="~/.kube/config")  # Ensure it points to your Lens kubeconfig
            kubeconfig = get_k8s_config(logger, cluster_name=cluster_name)
            logger.info(f"load kube config from dict: {kubeconfig=}")
            config.load_kube_config_from_dict(config_dict=kubeconfig)
            # config.load_kube_config(context=desired_context)  # Load the specified context

            # config.load_incluster_config()  # Use ServiceAccount authentication inside an EKS pod

            contexts, current_context = config.kube_config.list_kube_config_contexts()
            logger.info(f"{contexts=}")
            logger.info(f"{current_context=}")

            logger.info(f"Active user: {client.configuration.Configuration().username}")
            # assert k8s_config.Configuration().username

            # Initialize Kubernetes API client
            v1 = client.CoreV1Api()

            # Fetch all pods in all namespaces
            pods = v1.list_pod_for_all_namespaces(watch=False)
            # list_namespaced_pod

            logger.info(f"Found {len(pods.items)} pods in the cluster:")

            for pod in pods.items:
                logger.debug(
                    f"Pod: {pod.metadata.name}, Namespace: {pod.metadata.namespace}, Status: {pod.status.phase}")
                if pod.status.phase == "Running":
                    for container in pod.spec.containers:
                        logger.debug(f"EKS Pod: {pod.metadata.name}, Container Image: {container.image}")
                        running_containers.add(container.image)

            logger.info(
                f"{len(running_containers)} running containers in EKS cluster {cluster_name}: {running_containers}")

            all_running_containers = all_running_containers | running_containers
        except Exception as e:
            e.add_note(f"Failed to fetch running containers from EKS {cluster_name}")
            logger.error(str(e))
            raise

    return all_running_containers


def get_running_digests(running_containers, repository, tagged_images, logger: logging.Logger) -> tuple[set, list]:
    running_digests = set()
    running_images_details = []
    for running_image in running_containers:
        repository_uri = repository['repositoryUri']

        # get uri from running image - cut off the tag and digest
        # extract the base repository URI from running_image, excluding the tag or digest.
        uri = re.search(r"^[^@:]+", running_image).group(0)
        if uri != repository_uri:
            # Ensures that the base repository URI matches the current repository
            # (repository['repositoryUri']), filtering out irrelevant images.
            continue

        # Get the digest of the running image

        # check if image is directly referenced by digest e.g. @sha256:1234567890abcdef
        running_digest_match = re.search(r"@([^@]+)$", running_image)
        if running_digest_match:
            # In some cases, running containers may reference images directly by their digest instead of by a tag.
            digest = running_digest_match.group(1)
            running_digests.add(digest)
            running_images_details.append({
                "imageDigest": digest,
            })
            logger.warning(f"Running image is referenced by digest, not by by tag {running_image=}.")
        else:
            # the image is referenced by tag - lookup the digest for this tag
            tag = running_image.split(":")[1]
            image_details = [image_detail for image_detail in tagged_images if tag in image_detail['imageTags']]
            if image_details:
                if len(image_details) > 1:
                    logger.error(f"{running_image=}: {len(image_details)} digests match the same {tag=}: {image_details=}")
                for image_detail in image_details:
                    # there should be only one here, otherwise emit the error above
                    digest = image_detail['imageDigest']
                    running_digests.add(digest)
                    running_images_details.append({
                        "imageUrl": [repository['repositoryUri'] + ":" + tag],
                        "imagePushedAt": image_detail.get("imagePushedAt"),
                        "lastRecordedPullTime": image_detail.get('lastRecordedPullTime'),
                        "imageDigest": digest,
                    })
            else:
                # A container is using an image that does not exist anymore?
                logger.warning(f"Running image with '{tag=}' not found in tagged images, "
                               f"Is {running_image=} an image that no longer exist in ECR ?!")
                continue

    return running_digests, running_images_details


def get_running_digests_simple(running_containers, repository, tagged_images) -> set:
    running_digests_sha = set()
    for image in tagged_images:
        for tag in image['imageTags']:
            image_url = repository['repositoryUri'] + ":" + tag
            for running_images in running_containers:
                if image_url == running_images:
                    digest = image['imageDigest']
                    running_digests_sha.add(digest)
    return running_digests_sha


def delete_images(inputs: Inputs, ecr_client, delete_digests: set, delete_images_details: list, repo_id, name):
    logger = inputs.logger
    if not delete_digests:
        return

    if delete_images_details:
        sort_image_details_by_date(delete_images_details, title="delete", logger=logger)
        logger.info(f"{len(delete_images_details)} Image URLs that are marked for deletion:")
        for image_details in delete_images_details:
            logger.info(image_details)

    ## splitting list of images to delete on chunks with 100 images each
    ## http://docs.aws.amazon.com/AmazonECR/latest/APIReference/API_BatchDeleteImage.html#API_BatchDeleteImage_RequestSyntax
    i = 0
    delete_ids = [{'imageDigest': digest} for digest in delete_digests]
    logger.info(f"Registry Id: {repo_id}, Repository Name: {name}, "
                f"{f'Simulate deletion of' if inputs.dryrun else 'Deleting'} {len(delete_ids)} images...")
    for delete_ids_chunk in chunks(delete_ids, 100):
        i += 1
        logger.debug(f"Deleting chunk #{i} of {len(delete_ids_chunk)} images with image Ids: {delete_ids_chunk}")
        if not inputs.dryrun:
            delete_response = ecr_client.batch_delete_image(
                registryId=repo_id,
                repositoryName=name,
                imageIds=delete_ids_chunk
            )
            logger.info(delete_response)


def chunks(repo_list, chunk_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(repo_list), chunk_size):
        yield repo_list[i:i + chunk_size]


def get_default_logger(log_file='ecr_cleanup.log', error_file='ecr_cleanup-error.log'):
    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the desired logging level

    # Create handlers for stdout and stderr
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)

    # Set levels for handlers
    stdout_handler.setLevel(logging.INFO)  # Info and below go to stdout
    stderr_handler.setLevel(logging.ERROR)  # Error and above go to stderr

    # Define the log message format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file) # Log all levels to file
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if error_file:
        error_file_handler = logging.FileHandler(error_file) # Log errors separately
        error_file_handler.setLevel(logging.ERROR)  # Log errors and above to error file
        error_file_handler.setFormatter(formatter)
        logger.addHandler(error_file_handler)

    return logger


# Below is the test harness
if __name__ == '__main__':
    main(Inputs.from_parser())
