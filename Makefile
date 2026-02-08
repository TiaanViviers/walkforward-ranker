.PHONY: clean clean-dry

clean-dry:
	python3 scripts/clean_generated.py --all --dry-run

clean:
	python3 scripts/clean_generated.py --all
