QUARTO ?= quarto
PROJECT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: html pdf all clean preview

html:
	$(QUARTO) render $(PROJECT_DIR) --to html

pdf:
	$(QUARTO) render $(PROJECT_DIR) --to pdf

all: html pdf

preview:
	$(QUARTO) preview $(PROJECT_DIR)

clean:
	rm -rf $(PROJECT_DIR)_output
