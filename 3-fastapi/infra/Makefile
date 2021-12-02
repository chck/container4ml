.PHONY: all
all: help

.PHONY: bucket  ## Initialize the bucket for terraform
bucket:
	gcloud auth login
	gcloud config set project $(GCP_PROJECT)
	gsutil mb -c NEARLINE -l asia-northeast1 gs://$(GCP_BUCKET) || true
	gsutil versioning set on $(GCP_BUCKET)

.PHONY: terraform  ## Generate your terraform files via .env
terraform:
	@sed -e 's|_YOUR_PROJECT|${GCP_PROJECT}|g ; s|_YOUR_BUCKET|${GCP_BUCKET}|g' 01_setting.tf.tmpl > 01_setting.tf
	@sed -e 's|_YOUR_DOMAIN|${CLOUDRUN_DOMAIN}|g' 02_cloudrun.tf.tmpl > 02_cloudrun.tf
	@echo Generated!:
	@ls *.tf

.PHONY: help ## View help
help:
	@grep -E '^.PHONY: [a-zA-Z_-]+.*?## .*$$' $(MAKEFILE_LIST) | sed 's/^.PHONY: //g' | awk 'BEGIN {FS = "## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
