.PHONY: demo test reproduce
demo:        ## judge mode: run the dashboard with the shipped models and demo patients
	streamlit run app/app.py
test:        ## run the test suite (works without the raw data)
	python -m pytest -q tests
reproduce:   ## rebuild every result from raw PhysioNet files (~50 min, needs data/raw)
	./run_all.sh
