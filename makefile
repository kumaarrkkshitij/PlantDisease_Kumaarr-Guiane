# Makefile for Plant Disease Classification

install train eval app

install:
	pip install -r requirements.txt

train:
	python src/train_model.py

eval:
	python src/eval.py

app:
	python src/app.py