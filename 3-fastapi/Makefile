BASE_IMAGE_URI:=chck/container4ml-fastapi
IMAGE_URI:=gcr.io/$(GCP_PROJECT)/container4ml/fastapi
TAG:=1.0

.PHONY: all
all: help

.PHONY: build ## Build an image from a Dockerfile
build:
	docker build . -t $(IMAGE_URI):$(TAG)

.PHONY: push ## Push an image or a repository to a registry
push:
	docker tag $(BASE_IMAGE_URI):$(TAG) $(IMAGE_URI):$(TAG)
	docker tag $(IMAGE_URI):$(TAG) $(IMAGE_URI):latest
	docker push $(IMAGE_URI):$(TAG)
	docker push $(IMAGE_URI):latest

.PHONY: help ## View help
help:
	@grep -E '^.PHONY: [a-zA-Z_-]+.*?## .*$$' $(MAKEFILE_LIST) | sed 's/^.PHONY: //g' | awk 'BEGIN {FS = "## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
