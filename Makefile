# Define all the targets that are not files/directories
.PHONY: test


test:
	@pytest ./tests/ --cov-report=term-missing --cov=graphgen

install-torch:
	# @pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchtext==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
	@pip install torch==1.7.0 torchvision==0.8.1 torchtext==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

install-torch-geometric:
	$(eval CUDA_VERSION = $(shell python -c "import torch; print(torch.version.cuda)"))
	$(eval CUDA_STRING = $(if $(findstring None,$(CUDA_VERSION)),cpu,cu$(subst .,,$(CUDA_VERSION))))
	$(eval TORCH_VERSION = $(word 1,$(subst +, ,$(shell python -c "import torch; print(torch.__version__)"))))
	@echo "torch: $(TORCH_VERSION)"
	@echo "cuda: $(CUDA_VERSION) ($(CUDA_STRING))"
	@pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f "https://pytorch-geometric.com/whl/torch-$(TORCH_VERSION)+$(CUDA_STRING).html"
	@pip install torch-geometric

install-other:
	@pip install -r requirements.txt

install: install-torch install-torch-geometric install-other
