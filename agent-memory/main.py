import google.generativeai as genai
from dotenv import load_dotenv
from dotenv import find_dotenv
import os
from utils.record_audio import record_audio
from tools.daily_events import DailyEvents
from datetime import datetime
import json

load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings,
                              tools=[DailyEvents])

while True:
    memory = {
        "events": [],
        "interactions": []
    } if not os.path.exists("memory.json") else json.load(open("memory.json"))

    filename_audio = record_audio()

    audio_file = genai.upload_file(path=filename_audio)

    os.remove(filename_audio)

    actual_date = datetime.now().strftime("%d/%m/%Y")
    
    prompt = f"""You are a helpful assistant. You are responsible for remembering events of my life. Today is {actual_date} use this as a reference to remember events. If the event occurred in the past, you should use the date to remember the event using today's date as a reference.
    
    {json.dumps(memory)}
    
    """

    convo = model.start_chat(history=[])
    
    convo.send_message(prompt)
    response = convo.send_message(audio_file)

    if response.candidates[0].content.parts[0].function_call:
        tool_call = response.candidates[0].content.parts[0].function_call
        if tool_call.name == "DailyEvents":
            daily_events = DailyEvents(**tool_call.args)
            memory["events"].append(f"Day: {daily_events.date} - {daily_events.events}")

            memory['interactions'].append(f"Human: {audio_file.name}")
            memory['interactions'].append(f"Assistant: Evento do dia {daily_events.date} registrado com sucesso, posso te ajudar com mais alguma coisa?")
            print(f"Evento do dia {daily_events.date} registrado com sucesso, posso te ajudar com mais alguma coisa?")

    if response.text:
        memory['interactions'].append(f"Human: {audio_file.name}")
        memory['interactions'].append(f"Assistant: {response.text}")
        print(response.text)

    with open("memory.json", "w") as f:
        json.dump(memory, f)
