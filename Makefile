SHELL = /bin/bash
.DEFAULT_GOAL = help

.PHONY: pluto

pluto: ## Starts Pluto server (and allow for local packages)
	julia -e '\
		import Pkg; \
		Pkg.activate("."); \
		Pkg.instantiate(); \
		import Pluto; \
		Pluto.run() \
	'