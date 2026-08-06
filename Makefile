PYEXEC ?= "python3"

lint:
	$(PYEXEC) -m flake8 --ignore E251,E501 src

format:
	ruff check src/ --select F401 --fix
	yapf -i -r -vv src/
	isort --ls --ds src/

tail:
	tail -f $(ls -1 slurm-*.out | head -n 1)

clean:
	rm -I slurm-*.out
	
