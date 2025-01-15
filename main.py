'''
Copyright 2016 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with
the License. A copy of the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and
limitations under the License.
'''
from __future__ import print_function

import argparse
import os
import re
import boto3

REGION = None
DRYRUN = None
IMAGES_TO_KEEP = None
IGNORE_TAGS_REGEX = None


def initialize():
    global REGION
    global DRYRUN
    global IMAGES_TO_KEEP
    global IGNORE_TAGS_REGEX

    REGION = os.environ.get('REGION', "None")
    DRYRUN = os.environ.get('DRYRUN', "false").lower()
    if DRYRUN == "false":
        DRYRUN = False
    else:
        DRYRUN = True
    IMAGES_TO_KEEP = int(os.environ.get('IMAGES_TO_KEEP', 100))
    IGNORE_TAGS_REGEX = os.environ.get('IGNORE_TAGS_REGEX', "^$")

def handler(event, context):
    initialize()
    if REGION == "None":
        ec2_client = boto3.client('ec2')
        available_regions = ec2_client.describe_regions()['Regions']
        for region in available_regions:
            discover_delete_images(region['RegionName'])
    else:
        discover_delete_images(REGION)


def get_running_digests_sha_old(running_containers, repository, tagged_images) -> set:
    running_digests_sha = set()
    for image in tagged_images:
        for tag in image['imageTags']:
            image_url = repository['repositoryUri'] + ":" + tag
            for running_images in running_containers:
                if image_url == running_images:
                    digest = image['imageDigest']
                    running_digests_sha.add(digest)
    return running_digests_sha


def get_running_digests_sha(running_containers, repository, tagged_images) -> set:
    running_digests_sha = set()
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
        else:
            # the image is referenced by tag - lookup the digest for this tag
            tag = running_image.split(":")[1]
            image_tags = [x for x in tagged_images if tag in x['imageTags']]
            if image_tags:
                digest = image_tags[0]['imageDigest']
            else:
                # A container is using an image that does not exist anymore?
                print(f"Error: Image with '{tag=}' not found in tagged images, "
                      f"Is {running_image=} is using an image that does not exist anymore? ")
                continue

        if digest:
            running_digests_sha.add(digest)

    return running_digests_sha


def get_running_containers(ecs_client):
    running_containers = set()  # Actually, used container images
    list_clusters_paginator = ecs_client.get_paginator('list_clusters')
    for response_clusters_list_paginator in list_clusters_paginator.paginate():
        for cluster_arn in response_clusters_list_paginator['clusterArns']:
            print("cluster " + cluster_arn)

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

    return sorted(running_containers)


def get_running_containers_from_services(ecs_client):
    running_containers = set()  # Actually, used container images
    list_clusters_paginator = ecs_client.get_paginator('list_clusters')
    for response_clusters_list_paginator in list_clusters_paginator.paginate():
        for cluster_arn in response_clusters_list_paginator['clusterArns']:
            print("cluster " + cluster_arn)
            # List services in the cluster
            services_response = ecs_client.list_services(cluster=cluster_arn)
            service_arns = services_response['serviceArns']

            # Describe the service to get details
            services_details = ecs_client.describe_services(cluster=cluster_arn, services=service_arns)[
                'services'] if service_arns else []

            for service in services_details:
                # Get task definition for the service
                task_definition_arn = service['taskDefinition']
                task_definition_response = ecs_client.describe_task_definition(taskDefinition=task_definition_arn)
                task_definition = task_definition_response['taskDefinition']

                # Extract container images from task definition
                container_definitions = task_definition['containerDefinitions']
                for container_definition in container_definitions:
                    container_image = container_definition['image']
                    print(f"Service: {service['serviceName']}, Container Image: {container_image}")
                    running_containers.add(container_image)

    return sorted(running_containers)


def discover_delete_images(region_name: str):
    print("Discovering images in " + region_name)
    ecr_client = boto3.client('ecr', region_name=region_name)

    repositories = []
    describe_repo_paginator = ecr_client.get_paginator('describe_repositories')
    for describe_repo_response in describe_repo_paginator.paginate():
        for repo in describe_repo_response['repositories']:
            repositories.append(repo)

    ecs_client = boto3.client('ecs', region_name=region_name)

    running_containers = get_running_containers(ecs_client)

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

    print("Images that are running:")
    for image in running_containers:
        print(image)

    for repository in repositories:
        print("------------------------")
        print("Starting with repository :" + repository['repositoryUri'])
        delete_sha = []
        delete_tag = []
        tagged_images = []

        describe_image_paginator = ecr_client.get_paginator('describe_images')
        for describe_image_response in describe_image_paginator.paginate(
                registryId=repository['registryId'],
                repositoryName=repository['repositoryName']):
            for image in describe_image_response['imageDetails']:
                if 'imageTags' in image:
                    tagged_images.append(image)
                else:
                    append_to_list(delete_sha, image['imageDigest'])

        print("Total number of images found: {}".format(len(tagged_images) + len(delete_sha)))
        print("Number of untagged images found {}".format(len(delete_sha)))

        tagged_images.sort(key=lambda k: k['imagePushedAt'], reverse=True)

        # Get ImageDigest from ImageURL for running images. Do this for every repository
        running_digests_sha = get_running_digests_sha(running_containers, repository, tagged_images)

        print("Number of running images found {}".format(len(running_digests_sha)))
        ignore_tags_regex = re.compile(IGNORE_TAGS_REGEX)
        for image in tagged_images:
            if tagged_images.index(image) >= IMAGES_TO_KEEP:
                for tag in image['imageTags']:
                    if "latest" not in tag and ignore_tags_regex.search(tag) is None:
                        if not running_digests_sha or image['imageDigest'] not in running_digests_sha:
                            append_to_list(delete_sha, image['imageDigest'])
                            append_to_tag_list(delete_tag, {"imageUrl": repository['repositoryUri'] + ":" + tag,
                                                           "pushedAt": image["imagePushedAt"]})
        if delete_sha:
            print("Number of images to be deleted: {}".format(len(delete_sha)))
            delete_images(
                ecr_client,
                delete_sha,
                delete_tag,
                repository['registryId'],
                repository['repositoryName']
            )
        else:
            print("Nothing to delete in repository : " + repository['repositoryName'])


def append_to_list(image_digest_list, repo_id):
    if not {'imageDigest': repo_id} in image_digest_list:
        image_digest_list.append({'imageDigest': repo_id})


def append_to_tag_list(tag_list, tag_id):
    if not tag_id in tag_list:
        tag_list.append(tag_id)


def chunks(repo_list, chunk_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(repo_list), chunk_size):
        yield repo_list[i:i + chunk_size]


def delete_images(ecr_client, deletesha, deletetag, repo_id, name):
    if len(deletesha) >= 1:
        ## spliting list of images to delete on chunks with 100 images each
        ## http://docs.aws.amazon.com/AmazonECR/latest/APIReference/API_BatchDeleteImage.html#API_BatchDeleteImage_RequestSyntax
        i = 0
        for deletesha_chunk in chunks(deletesha, 100):
            i += 1
            if not DRYRUN:
                delete_response = ecr_client.batch_delete_image(
                    registryId=repo_id,
                    repositoryName=name,
                    imageIds=deletesha_chunk
                )
                print(delete_response)
            else:
                print("registryId:" + repo_id)
                print("repositoryName:" + name)
                print("Deleting {} chank of images".format(i))
                print("imageIds:", end='')
                print(deletesha_chunk)
    if deletetag:
        print("Image URLs that are marked for deletion:")
        for ids in deletetag:
            print("- {} - {}".format(ids["imageUrl"], ids["pushedAt"]))


# Below is the test harness
if __name__ == '__main__':
    REQUEST = {"None": "None"}
    PARSER = argparse.ArgumentParser(description='Deletes stale ECR images')
    PARSER.add_argument('-dryrun', help='Prints the repository to be deleted without deleting them', default='true',
                        action='store', dest='dryrun')
    PARSER.add_argument('-imagestokeep', help='Number of image tags to keep', default='100', action='store',
                        dest='imagestokeep')
    PARSER.add_argument('-region', help='ECR/ECS region', default=None, action='store', dest='region')
    PARSER.add_argument('-ignoretagsregex', help='Regex of tag names to ignore', default="^$", action='store', dest='ignoretagsregex')

    ARGS = PARSER.parse_args()
    if ARGS.region:
        os.environ["REGION"] = ARGS.region
    else:
        os.environ["REGION"] = "None"
    os.environ["DRYRUN"] = ARGS.dryrun.lower()
    os.environ["IMAGES_TO_KEEP"] = ARGS.imagestokeep
    os.environ["IGNORE_TAGS_REGEX"] = ARGS.ignoretagsregex
    handler(REQUEST, None)
