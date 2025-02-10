import whisper
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from langgraph.graph import StateGraph
from typing import Dict
import os

ffmpeg_path="C:\\Users\\338573\\Documents\\Capstone\\Multimodel\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe"
os.environ["PATH"]+=os.pathsep+os.path.dirname(ffmpeg_path)

MODEL_NAME = "base"
model = whisper.load_model(MODEL_NAME)

AgentState=Dict[str,str]

def record_audio(duration=10, sample_rate=44100):
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    file_path = "temp_audio.wav"
    wav.write(file_path, sample_rate, audio)
    print("Recording finished!")
    return file_path

def whisper_speech_to_text(state: AgentState) -> AgentState:
    audio_path = record_audio()
    result = model.transcribe(audio_path)
    transcribed_text = result["text"]
    state["transcribed_text"] = transcribed_text
    return state

graph = StateGraph(AgentState)
graph.add_node("speech_to_text", whisper_speech_to_text)
graph.set_entry_point("speech_to_text")
graph.set_finish_point("speech_to_text")
speech_to_text_agent = graph.compile()

def main():
    print("Starting the Speech-to-Text Agent using Whisper and LangGraph")
    result = speech_to_text_agent.invoke({})
    transcribed_text = result.get("transcribed_text", "")
    
    if transcribed_text:
        print(f"Transcribed Text: {transcribed_text}")
    else:
        print("No valid speech input detected.")

if __name__ == "__main__":
    main()
