VENV_DIR = .venv
PYTHON = python3
PIP = $(VENV_DIR)/bin/pip
PYTHON_VENV = $(VENV_DIR)/bin/python
VERSION_TYPE ?= patch

.PHONY: setup
setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: install
install:
	$(PIP) install -r requirements.txt


install-editable:
	pip install -e  ../cjlapao/pd-ai-agent-core

uninstall-editable:
	pip uninstall -y pd-ai-agent-core

# Version increment targets
.PHONY: version-increment frontend-version-increment backend-version-increment

version-increment: frontend-version-increment backend-version-increment
	@echo "Incremented both frontend and backend versions ($(VERSION_TYPE))"

frontend-version-increment:
	@echo "Incrementing frontend version ($(VERSION_TYPE))"
	cd ../cjlapao/pd-ai-agents/prl-ai-agent/frontend && bash scripts/increment-version.sh $(VERSION_TYPE)

backend-version-increment:
	@echo "Incrementing backend version ($(VERSION_TYPE))"
	cd ../cjlapao/pd-ai-agents/prl-ai-agent/backend/src && python3 -c "from version import set_version, get_version; major, minor, build = get_version(); version_type='$(VERSION_TYPE)'; print(set_version(major+1 if version_type == 'major' else major, minor+1 if version_type == 'minor' else (0 if version_type == 'major' else minor), build+1 if version_type == 'patch' else 0))"