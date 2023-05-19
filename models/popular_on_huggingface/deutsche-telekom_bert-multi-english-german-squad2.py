# labels: test_group::monthly author::deutsche-telekom name::bert-multi-english-german-squad2 downloads::1,615 license::mit task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="deutsche-telekom/bert-multi-english-german-squad2",
    tokenizer="deutsche-telekom/bert-multi-english-german-squad2"
)

contexts = ["Die Allianz Arena ist ein Fußballstadion im Norden von München und bietet bei Bundesligaspielen 75.021 Plätze, zusammengesetzt aus 57.343 Sitzplätzen, 13.794 Stehplätzen, 1.374 Logenplätzen, 2.152 Business Seats und 966 Sponsorenplätzen. In der Allianz Arena bestreitet der FC Bayern München seit der Saison 2005/06 seine Heimspiele. Bis zum Saisonende 2017 war die Allianz Arena auch Spielstätte des TSV 1860 München.",
            "Harvard is a large, highly residential research university. It operates several arts, cultural, and scientific museums, alongside the Harvard Library, which is the world's largest academic and private library system, comprising 79 individual libraries with over 18 million volumes. "]
questions = ["Wo befindet sich die Allianz Arena?", 
            "What is the worlds largest academic and private library system?"]
 
qa_pipeline(context=contexts, question=questions)
