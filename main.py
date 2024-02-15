# pip install git+https://github.com/huggingface/transformers.git

from deep_translator import GoogleTranslator
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import re

MAX_QUESTION = 250

app = FastAPI()

class QuestionModel(BaseModel):
    question: str
    temperature: int = 0

rutranslator = GoogleTranslator(source='auto', target='ru')
entranslator = GoogleTranslator(source='auto', target='en')

def format_text_advanced(text):
    trimmed_text = text.strip()
    if trimmed_text:
        formatted_text = trimmed_text[0].upper() + trimmed_text[1:]
    else:
        formatted_text = ""
    return rutranslator.translate(formatted_text)

@app.get("/")
async def read_root():
    return {"message": "This TinyLlama-1.1B-Chat-v1.0 server version 1 beta"}

@app.post("/gpt/question")
async def gpt_ask(questionModel: QuestionModel):
    try:

        l = len(questionModel.question)

        if (l==0):
          raise HTTPException(status_code=422, detail='Не задан вопрос')
        
        if (l>MAX_QUESTION):
          raise HTTPException(status_code=422, detail=f'Максимальная длина вопроса {MAX_QUESTION} символов')
        
        messages = [
            {
                "role": "system",
                "content": "You are a walking encyclopedia, answer users questions briefly",
            },
            {   
                "role": "user", 
                "content": entranslator.translate(questionModel.question)
            },
        ]

        pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device='cpu')
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        response = pipe(prompt, truncation=True, temperature=questionModel.temperature)

        print(response[0]['generated_text'])

        pattern = r"\<\|assistant\|\>\n(.*)"
        match = re.search(pattern, response[0]['generated_text'], re.DOTALL)

        if match:
            return { 
                "question": format_text_advanced(questionModel.question),
                "answer": format_text_advanced(match.group(1))
                }
        else:
            raise HTTPException(status_code=500, detail="Ошибка формата ответа GPT.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Внутренняя ошибка сервера: {e}')
    
