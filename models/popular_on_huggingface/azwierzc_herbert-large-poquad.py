# labels: test_group::monthly author::azwierzc name::herbert-large-poquad downloads::283 license::gpl-3.0 task::Natural_Language_Processing sub_task::Question_Answering
import transformers
from transformers import pipeline

question_answerer = pipeline("question-answering", model='azwierzc/herbert-large-poquad', handle_impossible_answer=True)

context = "Pod koniec życia Carducci, choć nadal budził kontrowersje, był szanowanym autorytetem literackim i naukowym. Z okazji 35-lecia pracy profesorskiej został obdarowany gałązką wawrzynu z drzewa rosnącego przy grobie Dantego w Rawennie oraz księgą pamiątkową zawierającą nazwiska wszystkich jego studentów. Kiedy ze względu na hemiplegię musiał w roku 1904 zaniechać prowadzenia wykładów, parlament przyznał mu – podobnie jak kiedyś Alessandrowi Manzoniemu – dożywotnią emeryturę. Katedrę objął po nim Giovanni Pascoli. W roku 1906 Giosuè Carducci został wyróżniony Nagrodą Nobla w dziedzinie literatury. Komitet Noblowski czuł się w obowiązku podkreślić, że pogańskie motywy w jego wierszach wcale nie oznaczały odrzucenia chrześcijaństwa, a jedynie krytykę błędnych posunięć Kościoła. Schorowany Carducci nie zdołał pojechać do Sztokholmu, żeby osobiście odebrać nagrodę, gościł natomiast u siebie ambasadora Szwecji. Zmarł na marskość wątroby dwa miesiące później, 16 lutego 1907 roku. Jego śmierć upamiętnił wierszem Per la tomba di G. Carducci (Na grób G. Carducciego) Gabriele D’Annunzio, główny przedstawiciel włoskiego dekadentyzmu. Utwory Carducciego często inspirowały kompozytorów, którzy wykorzystywali je jako teksty swoich pieśni (Alfredo Casella – Notte de maggio, Guido Alberto Fano – Vere novo, Stanislao Gastaldon – Viva il Re)."
question = "Jakie wyróżnienie otrzymał Carducci?"

result = question_answerer({
    'context': context,
    'question': question})

print(result)
