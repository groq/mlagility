# labels: test_group::monthly author::deutsche-telekom name::electra-base-de-squad2 downloads::217 license::mit task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="deutsche-telekom/electra-base-de-squad2",
    tokenizer="deutsche-telekom/electra-base-de-squad2"
)

contexts = ['''Die Robert Bosch GmbH ist ein im Jahr 1886 von Robert Bosch gegründetes multinationales deutsches Unternehmen. 
Es ist tätig als Automobilzulieferer, Hersteller von Gebrauchsgütern und Industrie- und Gebäudetechnik und darüber hinaus 
in der automatisierten Verpackungstechnik, wo Bosch den führenden Platz einnimmt. Die Robert Bosch GmbH und ihre rund 460 
Tochter- und Regionalgesellschaften in mehr als 60 Ländern bilden die Bosch-Gruppe. Der Sitz der Geschäftsführung befindet 
sich auf der Schillerhöhe in Gerlingen, der Firmensitz in Stuttgart. Seit dem 1. Juli 2012 ist Volkmar Denner Vorsitzender 
der Geschäftsführung. Im Jahr 2015 konnte Bosch die Spitzenposition zurückgewinnen. Die Automobilsparte war im Jahr 2018 
für 61 % des Konzernumsatzes von Bosch verantwortlich. Das Unternehmen hatte im Jahr 2018 in Deutschland an 85 Standorten 
139.400 Mitarbeiter.''']*2

questions = ["Wer leitet die Robert Bosch GmbH?", 
            "Wer begründete die Robert Bosch GmbH?"]

qa_pipeline(context=contexts, question=questions)
