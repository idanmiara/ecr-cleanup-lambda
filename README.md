# Automated Image Cleanup for Amazon ECR
The Python script and Lambda function described here help clean up images in [Amazon ECR](https://aws.amazon.com/ecr).
The script looks for images that are not used in running 
[Amazon ECS](https://aws.amazon.com/ecs) tasks, 
[Amazon EKS](https://aws.amazon.com/eks) tasks and 
[Amazon Lambda](https://aws.amazon.com/lambda) container images that can be deleted.
You can configure the script to print the image list first to confirm deletions, specify a region, 
or specify a number of images to keep for potential rollbacks.

## Authenticate with AWS
[Configuring the AWS Command Line Interface.](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)

## Use virtualenv for Python execution

To prevent any problems with your system Python version conflicting with the application, we recommend using virtualenv.
This code was tested with Python 3.13, but would probably work with any Python >= 3.6.

Install Python:
    `pip install python 3`

Install virtualenv:

    $ pip install virtualenv
    $ virtualenv -p PATH_TO_YOUR_PYTHON_3 cloudformtion
    $ virtualenv ~/.virtualenvs/cloudformtion
    $ source ~/.virtualenvs/cloudformtion/bin/activate
    
## Generate the Lambda package

1. CD to the folder that contains main.py.
1. Run the following command:
`pip install -r requirements.txt -t `pwd``
1. Compress the contents of folder (not the folder).
    
## Upload the package to Lambda

1. Run the following command:
`aws lambda create-function --function-name {NAME_OF_FUNCTION} --runtime python3.13 
--role {ARN_NUMBER} --handler main.handler --timeout 15 
--zip-file fileb://{ZIP_FILE_PATH}`
    
## Send the package update to Lambda

1. Run the following command:
    
    `aws lambda update-function-code --function-name {NAME_OF_FUNCTION} --zip-file fileb://{ZIP_FILE_PATH}`


## Examples
Prints the images that are not used by running tasks and which are older than the last 100 versions, in all regions:

`python main.py`


Deletes the images that are not used by running tasks and which are older than the last 100 versions, in all regions:

`python main.py -no-dryrun`


Deletes the images that are not used by running tasks and which are older than the last 20 versions (in each repository), in all regions:

`python main.py -no-dryrun –images_to_keep 20`


Deletes the images that are not used by running tasks and which are older than the last 20 versions (in each repository), in Oregon only:

`python main.py -no-dryrun –images_to_keep 20 –region us-west-2`


Deletes the images that are not used by running tasks and which are older than the last 20 versions (in each repository), in Oregon only, and ignore image tags that contains `release` or `archive`:

`python main.py -no-dryrun –images_to_keep 20 –region us-west-2 --protect_tags_regex release|archive`


For full option list, please refer to the help, by running:

`python main.py -h`

````
usage: main.py [-h] [-no-dryrun] [-region REGION]
               [-repo_name_regex REPO_NAME_REGEX]
               [-images_to_keep IMAGES_TO_KEEP] [-older_than OLDER_THAN]
               [-protect_tags_regex PROTECT_TAGS_REGEX] [-unprotect-latest]
               [-no-ecs] [-no-lambda] [-no-eks]
               [-connect_timeout CONNECT_TIMEOUT] [-read_timeout READ_TIMEOUT]
               [-max_attempts MAX_ATTEMPTS]

Deletes stale ECR images

options:
  -h, --help            show this help message and exit
  -no-dryrun            Don't just prints the repository to be deleted,
                        actually delete them
  -region REGION        ECR/ECS region
  -repo_name_regex REPO_NAME_REGEX
                        Regex of repo names to search
  -images_to_keep IMAGES_TO_KEEP
                        Number of image tags to keep
  -older_than OLDER_THAN
                        Only delete images older than a specified amount of
                        days
  -protect_tags_regex PROTECT_TAGS_REGEX
                        Regex of tag names to protect (not delete)
  -unprotect-latest     Allow deletion images with `latest` tag
  -no-ecs               Don't search ECS for running images
  -no-lambda            Don't search Lambda for running images
  -no-eks               Don't search EKS for running images
  -connect_timeout CONNECT_TIMEOUT
                        ECS connection timeout (in seconds)
  -read_timeout READ_TIMEOUT
                        ECS read timeout (in seconds)
  -max_attempts MAX_ATTEMPTS
                        ECS maximum number of attempts

Deletion logic: In each ECR repository that contains more than
`images_to_keep` non running images (running images are images that currently
deployed on a container), Iterate through a list of images, sorted from oldest
to newest (image date: if an image was pulled - last pull date, otherwise push
date), Mark images for deletion, until less then `images_to_keep` are left, or
until iterated through the entire list. An image is marked for deletion if it
is older than `older_than`; AND is not tagged `latest`, AND does not have has
any tags that matches `protect_tags_regex`. Note that if not enough images are
marked for deletion, more than `images_to_keep` may be left untouched.
```