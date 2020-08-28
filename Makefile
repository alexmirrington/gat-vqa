# Define all the targets that are not files/directories
.PHONY: test


test:
	@pytest ./tests/ --cov-report term-missing --cov=graphgen

requirements:
	@pip install -r requirements.txt

install: requirements
	# Refer to: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
	$(eval CUDA_VERSION = $(shell python -c "import torch; print(torch.version.cuda)"))
	$(eval CUDA_STRING = $(if $(findstring None,$(CUDA_VERSION)),cpu,cu$(subst .,,$(CUDA_VERSION))))
	$(eval TORCH_VERSION = $(subst +cpu,,$(shell python -c "import torch; print(torch.__version__)")))
	@echo "torch: $(TORCH_VERSION)"
	@echo "cuda: $(CUDA_VERSION) ($(CUDA_STRING))"
	@pip install torch-scatter==latest+$(CUDA_STRING) -f "https://pytorch-geometric.com/whl/torch-$(TORCH_VERSION).html"
	@pip install torch-sparse==latest+$(CUDA_STRING) -f https://pytorch-geometric.com/whl/torch-$(TORCH_VERSION).html
	@pip install torch-cluster==latest+$(CUDA_STRING) -f https://pytorch-geometric.com/whl/torch-$(TORCH_VERSION).html
	@pip install torch-spline-conv==latest+$(CUDA_STRING) -f https://pytorch-geometric.com/whl/torch-$(TORCH_VERSION).html
