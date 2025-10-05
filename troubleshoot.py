from transformers import pipeline
nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print(nlp("Microsoft stock surges after earnings beat expectations"))
print(nlp("Apple shares drop as sales disappoint"))
